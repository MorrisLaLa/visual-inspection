"""Kaggle 視覺檢測共用核心（Ultralytics YOLO）。

由 medical_yolo.py 與 semiconductor_yolo.py 匯入，負責兩者共用的流程：

    讀取資料 → 依分組雜湊切 train/val/test（測試集長期固定）→（可選）大圖切塊
    → 訓練 → 在 val 上逐類調門檻 → 在 test 上評估 → 與舊模型比較（上線關卡）
    → 對未標註影像產生複判佇列（主動學習）→ 匯出部署格式
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
ON_KAGGLE = Path("/kaggle/working").is_dir()
CONF_FLOOR = 0.01  # 推論先用極低門檻收集所有候選，之後再逐類套用調好的門檻
DATASET_MARKER = ".vi_dataset"  # 標記「這個資料夾是本程式建立的」，重建時才敢刪除


# ───────────────────────── 資料結構 ─────────────────────────


@dataclass
class Item:
    """一張影像與它的標註（像素座標）。"""

    image: Path
    group: str  # 分組 key（病人 / 批號 / 晶圓）；同一組一定落在同一個 split，避免資料洩漏
    boxes: list = field(default_factory=list)  # [(cls, x1, y1, x2, y2)]
    polygons: list = field(default_factory=list)  # [(cls, [x1, y1, x2, y2, ...])]
    label: str | None = None  # 分類任務的類別名稱
    split: str | None = None  # 來源資料已指定的 split


@dataclass
class Dataset:
    items: list
    names: list
    task: str  # detect / segment / classify


@dataclass
class TileCfg:
    size: int = 640
    overlap: float = 0.2
    empty_ratio: float = 0.3  # 訓練集保留多少比例的無缺陷切塊（負樣本，用來壓低過殺）
    min_visible: float = 0.4  # 缺陷被切到剩不到這個比例就不標，避免標到殘缺碎片


@dataclass
class Domain:
    name: str  # 輸出資料夾名稱
    escape_label: str  # 「漏掉」指標的顯示名稱
    false_alarm_label: str  # 「誤報」指標的顯示名稱
    normal_names: tuple  # 自動辨識「正常 / 良品」類別用
    aug: dict  # 預設資料增強
    disclaimer: str = ""


# ───────────────────────── 環境 ─────────────────────────


def default_dirs(name: str) -> tuple[Path, Path]:
    """(輸出目錄, 轉換後資料集目錄)。Kaggle 上資料集放 /tmp，不佔 /kaggle/working 的 20GB 輸出空間。"""
    if ON_KAGGLE:
        return Path("/kaggle/working") / name, Path("/tmp/vi-datasets") / name
    return Path("runs_vi") / name, Path("runs_vi") / name / "dataset"


def cache_dir(args) -> Path:
    """轉檔快取（DICOM→PNG、晶圓圖→PNG）；放在資料集目錄外，重建資料集時不會被刪。"""
    d = args.data_dir.parent / f"{args.data_dir.name}_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_package(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError:
        print(f"安裝 {pip_name or module} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or module])
        return __import__(module)


def setup_ultralytics(datasets_dir: Path) -> None:
    ensure_package("ultralytics")
    os.environ.setdefault("WANDB_MODE", "disabled")  # Kaggle 預裝 wandb，不關掉會卡在登入提示
    from ultralytics import settings

    updates = {k: False for k in ("wandb", "comet", "clearml", "neptune", "dvc", "mlflow", "raytune") if k in settings}
    updates["datasets_dir"] = str(datasets_dir.resolve())  # 內建資料集（例如 brain-tumor.yaml）下載到這裡
    settings.update(updates)


def load_model(weights):
    from ultralytics import YOLO

    try:
        return YOLO(str(weights))
    except Exception as e:  # 舊版 ultralytics 沒有 YOLO26
        w = str(weights)
        if "yolo26" not in w:
            raise
        fallback = w.replace("yolo26", "yolo11")
        print(f"⚠️ 無法載入 {w}（{e}）；改用 {fallback}。要用 YOLO26 請先 pip install -U ultralytics")
        return YOLO(fallback)


def default_weights(task: str, size: str) -> str:
    return f"yolo26{size}{ {'detect': '', 'segment': '-seg', 'classify': '-cls'}[task] }.pt"


def device_kwargs(args) -> dict:
    return {"device": args.device} if args.device is not None else {}


# ───────────────────────── 檔案工具 ─────────────────────────


def iter_images(root, exts=IMG_EXTS) -> list[Path]:
    """遞迴列出影像，略過 macOS 壓縮殘留（__MACOSX、._ 檔）。"""
    return sorted(
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in exts and p.is_file() and not p.name.startswith(".") and "__MACOSX" not in p.parts
    )


def image_size(path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(Path(src).resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def unique_stem(path: Path) -> str:
    """同名檔案來自不同資料夾時加上路徑雜湊，避免互相覆蓋。"""
    return f"{Path(path).stem}_{hashlib.md5(str(path).encode()).hexdigest()[:6]}"


def split_from_path(path: Path, root: Path) -> str | None:
    try:
        parts = Path(path).relative_to(root).parts[:-1]
    except ValueError:
        parts = Path(path).parts[:-1]
    for part in reversed(parts):
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return None


def hash_fraction(key: str) -> float:
    """把字串穩定地對應到 [0, 1)；同一個 key 每次結果都一樣。"""
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


class ImageIndex:
    """用檔名或主檔名找影像，處理標註檔與影像放在不同資料夾的情況。"""

    def __init__(self, root, exts=IMG_EXTS):
        self.all = iter_images(root, exts)
        self.by_name, self.by_stem = {}, {}
        for p in self.all:
            self.by_name.setdefault(p.name.lower(), p)
            self.by_stem.setdefault(p.stem.lower(), p)

    def find(self, name) -> Path | None:
        name = Path(str(name).strip()).name.lower()
        if not name:
            return None
        # 先比完整檔名，再把整串當主檔名（DICOM UID 本身就含「.」），最後才去掉副檔名比對
        return self.by_name.get(name) or self.by_stem.get(name) or self.by_stem.get(Path(name).stem)


def item_boxes(it: Item) -> list:
    """取得偵測框；只有多邊形標註時用多邊形外框。"""
    if it.boxes:
        return it.boxes
    out = []
    for c, pts in it.polygons:
        xs, ys = pts[0::2], pts[1::2]
        out.append((c, min(xs), min(ys), max(xs), max(ys)))
    return out


def apply_group_regex(ds: Dataset, pattern: str | None) -> bool:
    """用正規表示式從檔名取出分組 key（病人 ID、批號等）。回傳是否有套用。"""
    if not pattern:
        return False
    rx = re.compile(pattern)
    hit = 0
    for it in ds.items:
        m = rx.search(it.image.name)
        if m:
            it.group = m.group(1) if m.groups() else m.group(0)
            hit += 1
    print(f"分組規則 {pattern!r}：{hit}/{len(ds.items)} 張影像比對成功，共 {len({it.group for it in ds.items})} 組")
    return True


# ───────────────────────── 讀取各種標註格式 ─────────────────────────


def detect_format(src, exts=IMG_EXTS) -> str:
    p = Path(src)
    if p.is_file():
        return {".yaml": "yolo-yaml", ".yml": "yolo-yaml", ".json": "coco", ".csv": "csv", ".pkl": "pickle"}.get(
            p.suffix.lower(), "unknown"
        )
    if not p.exists():
        return "yolo-yaml" if p.suffix.lower() in (".yaml", ".yml") else "missing"  # 可能是 Ultralytics 內建資料集
    for name in ("data.yaml", "dataset.yaml"):
        if (p / name).is_file():
            return "yolo-yaml"
    if any(_is_voc(x) for x in _first(p.rglob("*.xml"), 20)):
        return "voc"
    if any(_is_coco(x) for x in _first(p.rglob("*.json"), 20)):
        return "coco"
    if next(p.glob("**/labels/*.txt"), None) is not None:
        return "yolo"
    if iter_images(p, exts):
        return "classify"
    return "unknown"


def _first(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out


def _is_voc(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return b"<annotation" in f.read(512)
    except OSError:
        return False


def _is_coco(p: Path) -> bool:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(d, dict) and "images" in d and "annotations" in d


def load_auto(src, exts=IMG_EXTS) -> Dataset:
    """自動判斷格式：YOLO（yaml 或資料夾）、Pascal VOC XML、COCO JSON、分類資料夾。"""
    fmt = detect_format(src, exts)
    print(f"資料格式：{fmt}（{src}）")
    p = Path(src)
    if fmt == "yolo-yaml":
        if p.is_dir():
            p = next(p / n for n in ("data.yaml", "dataset.yaml") if (p / n).is_file())
        return load_yolo_yaml(p, exts)
    if fmt == "yolo":
        return load_yolo_dir(p, exts=exts)
    if fmt == "voc":
        return load_voc(p, exts)
    if fmt == "coco":
        return load_coco(p, exts)
    if fmt == "classify":
        return load_class_folders(p, exts)
    raise SystemExit(f"無法辨識資料格式：{src}（支援 YOLO yaml/資料夾、VOC XML、COCO JSON、分類資料夾）")


def finalize_named(items: list, task: str, names: list | None = None) -> Dataset:
    """把以類別名稱標記的框 / 多邊形轉成索引；類別依名稱排序，讓每次執行順序一致。"""
    labels = {b[0] for it in items for b in it.boxes} | {p[0] for it in items for p in it.polygons}
    names = list(names) if names else sorted(labels, key=str)
    idx = {n: i for i, n in enumerate(names)}
    for it in items:
        it.boxes = [(idx[b[0]], *b[1:]) for b in it.boxes if b[0] in idx]
        it.polygons = [(idx[p[0]], p[1]) for p in it.polygons if p[0] in idx]
    return Dataset(items, [str(n) for n in names], task)


def _label_path(img: Path) -> Path:
    """與 Ultralytics 相同的規則：把最後一個 /images/ 換成 /labels/。"""
    s = os.fspath(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def read_yolo_label(img: Path, w: int, h: int) -> tuple[list, list]:
    boxes, polys = [], []
    lp = _label_path(img)
    if not lp.is_file():
        return boxes, polys
    for line in lp.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c, xs = int(float(v[0])), [float(x) for x in v[1:]]
        if len(xs) == 4:
            cx, cy, bw, bh = xs
            boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        elif len(xs) >= 6 and len(xs) % 2 == 0:
            polys.append((c, [x * (w if i % 2 == 0 else h) for i, x in enumerate(xs)]))
    return boxes, polys


def _yolo_items(images: list[tuple[Path, str | None]]) -> list[Item]:
    items, seen = [], set()
    for img, split in images:
        if img in seen:  # 有些資料集 val 直接指向 train，重複的只留第一次
            continue
        seen.add(img)
        w, h = image_size(img)
        boxes, polys = read_yolo_label(img, w, h)
        items.append(Item(image=img, group=img.stem, boxes=boxes, polygons=polys, split=split))
    return items


def _yolo_task(items) -> str:
    return "segment" if any(it.polygons for it in items) else "detect"


def load_yolo_yaml(src, exts=IMG_EXTS) -> Dataset:
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(str(src))  # 內建名稱（如 brain-tumor.yaml）會自動下載
    names = [d["names"][i] for i in sorted(d["names"])]
    images = []
    for split in SPLITS:
        specs = d.get(split) or []
        for spec in specs if isinstance(specs, list) else [specs]:
            images += [(p, split) for p in _images_from_spec(spec, Path(d["path"]), exts)]
    items = _yolo_items(images)
    for it in items:  # 圖檔沒有標註 → 視為負樣本（良品 / 正常）
        it.boxes = [b for b in it.boxes if b[0] < len(names)]
        it.polygons = [p for p in it.polygons if p[0] < len(names)]
    return Dataset(items, names, _yolo_task(items))


def _images_from_spec(spec, root: Path, exts) -> list[Path]:
    spec = Path(spec)
    if spec.is_dir():
        return iter_images(spec, exts)
    if spec.suffix == ".txt" and spec.is_file():
        out = []
        for line in spec.read_text().splitlines():
            if line.strip():
                p = Path(line.strip())
                out.append(p if p.is_absolute() else (root / p).resolve())
        return [p for p in out if p.suffix.lower() in exts]
    return [spec] if spec.is_file() else []


def load_yolo_dir(root: Path, exts=IMG_EXTS) -> Dataset:
    items = _yolo_items([(p, split_from_path(p, root)) for p in iter_images(root, exts)])
    names = None
    for fn in ("classes.txt", "obj.names"):
        f = next(root.rglob(fn), None)
        if f:
            names = [n.strip() for n in f.read_text().splitlines() if n.strip()]
            break
    n = 1 + max([b[0] for it in items for b in it.boxes] + [p[0] for it in items for p in it.polygons], default=0)
    names = names or [f"class_{i}" for i in range(n)]
    return Dataset(items, names, _yolo_task(items))


def load_voc(root: Path, exts=IMG_EXTS) -> Dataset:
    index = ImageIndex(root, exts)
    items, labeled, missing = [], set(), 0
    for xml in sorted(root.rglob("*.xml")):
        try:
            r = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        if r.tag != "annotation":
            continue
        img = index.find(r.findtext("filename") or "") or index.find(xml.stem)
        if img is None:
            missing += 1
            continue
        boxes = []
        for obj in r.iter("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append(((obj.findtext("name") or "").strip(), x1, y1, x2, y2))
        items.append(Item(image=img, group=img.stem, boxes=boxes, split=split_from_path(img, root)))
        labeled.add(img)
    negatives = [Item(image=p, group=p.stem, split=split_from_path(p, root)) for p in index.all if p not in labeled]
    if missing:
        print(f"⚠️ {missing} 個 XML 找不到對應影像")
    print(f"VOC：{len(items)} 張有標註，{len(negatives)} 張沒有 XML（視為無缺陷）")
    return finalize_named(items + negatives, "detect")


def load_coco(src: Path, exts=IMG_EXTS) -> Dataset:
    jsons = [src] if src.is_file() else [p for p in sorted(src.rglob("*.json")) if _is_coco(p)]
    root = src.parent if src.is_file() else src
    index = ImageIndex(root, exts)
    items, missing = {}, 0
    for jp in jsons:
        d = json.loads(jp.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        token = re.search(r"(train|valid|val|test)", jp.stem.lower())
        json_split = split_from_path(jp, root) or (SPLIT_ALIASES[token.group(1)] if token else None)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            p = jp.parent / im["file_name"]
            p = p if p.is_file() else index.find(im["file_name"])
            if p is None:
                missing += 1
                continue
            it = items.setdefault(str(p), Item(image=p, group=p.stem, split=json_split or split_from_path(p, root)))
            for a in anns[im["id"]]:
                if a.get("iscrowd"):
                    continue
                name = cats.get(a["category_id"], str(a["category_id"]))
                x, y, w, h = a["bbox"]
                it.boxes.append((name, x, y, x + w, y + h))
                seg = a.get("segmentation")
                if isinstance(seg, list):
                    it.polygons += [(name, [float(v) for v in poly]) for poly in seg if len(poly) >= 6]
    if missing:
        print(f"⚠️ COCO：{missing} 張影像找不到檔案")
    items = list(items.values())
    return finalize_named(items, "segment" if any(it.polygons for it in items) else "detect")


def load_class_folders(root: Path, exts=IMG_EXTS) -> Dataset:
    """分類資料夾：.../<類別>/<影像>，可選擇外層有 train/val/test。"""
    items = []
    for p in iter_images(root, exts):
        if p.parent == root or p.parent.name.lower() in SPLIT_ALIASES:
            continue  # 影像直接放在根目錄或 split 資料夾：沒有類別
        items.append(Item(image=p, group=p.stem, label=p.parent.name, split=split_from_path(p, root)))
    return Dataset(items, sorted({it.label for it in items}), "classify")


def mask_to_polygons(mask_path: Path, n_classes: int, img_wh: tuple[int, int], min_area: float = 4.0) -> list:
    """分割遮罩 → YOLO 多邊形。二值遮罩（0/255）視為單一類別；多值遮罩的像素值 v 對應類別 v-1。"""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    values = [v for v in np.unique(m) if v > 0]
    binary = n_classes == 1 or set(values) <= {255} or max(values, default=0) > n_classes
    layers = [(0, m > 127)] if binary else [(int(v) - 1, m == v) for v in values]
    sx, sy = img_wh[0] / m.shape[1], img_wh[1] / m.shape[0]
    polys = []
    for c, layer in layers:
        contours, _ = cv2.findContours(layer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2)
            if len(cnt) >= 3:
                polys.append((c, [float(v) for x, y in cnt for v in (x * sx, y * sy)]))
    return polys


# ───────────────────────── 切分 ─────────────────────────


def assign_splits(ds: Dataset, val_pct: float, test_pct: float, resplit: bool = False) -> None:
    """依 group 的雜湊分配 train/val/test。

    雜湊切分的好處：之後加入新資料時，舊資料的分組完全不變，測試集可以長期固定當「黃金測試集」，
    新舊模型的比較才公平。來源已完整提供 train/val/test 時沿用來源分組（--resplit 可強制重切）。
    """
    have = {it.split for it in ds.items}
    if not resplit and set(SPLITS) <= have and None not in have:
        print("沿用來源資料的 train/val/test 分組")
        return
    for it in ds.items:
        h = hash_fraction(it.group)
        it.split = "test" if h < test_pct else "val" if h < test_pct + val_pct else "train"
    _fill_empty_splits(ds)


def _fill_empty_splits(ds: Dataset) -> None:
    """小資料集時雜湊可能讓某個類別（或無缺陷影像）在 val/test 一張都沒有，這時從 train 移一整組過去。"""
    by_group = defaultdict(list)
    keys = {}
    for it in ds.items:
        by_group[it.group].append(it)
        if ds.task == "classify":
            keys[id(it)] = {it.label}
        else:  # 負樣本也要分到 val/test，才算得出過殺率 / 特異度
            keys[id(it)] = ({b[0] for b in item_boxes(it)} or {"__negative__"}) | {"__all__"}
    groups_of = defaultdict(set)
    for it in ds.items:
        for k in keys[id(it)]:
            groups_of[k].add(it.group)
    for key in sorted(groups_of, key=str):
        groups = sorted(groups_of[key], key=hash_fraction)
        for split in ("val", "test"):
            if any(by_group[g][0].split == split for g in groups):
                continue
            movable = [g for g in groups if by_group[g][0].split == "train"]
            if len(movable) < 2:
                name = ds.names[key] if isinstance(key, int) else key
                print(f"⚠️ 類別 {name} 樣本太少，{split} 裡沒有它")
                continue
            for it in by_group[movable[0]]:
                it.split = split


def print_split_summary(ds: Dataset) -> None:
    print(f"\n任務：{ds.task}，類別（{len(ds.names)}）：{ds.names}")
    for split in SPLITS:
        its = [it for it in ds.items if it.split == split]
        if ds.task == "classify":
            cnt = Counter(it.label for it in its)
            detail = ", ".join(f"{n}={cnt[n]}" for n in ds.names)
        else:
            cnt = Counter(b[0] for it in its for b in item_boxes(it))
            neg = sum(1 for it in its if not item_boxes(it))
            detail = ", ".join(f"{n}={cnt[i]}" for i, n in enumerate(ds.names)) + f"；無標註（負樣本）{neg} 張"
        print(f"  {split:5s} {len(its):6d} 張（{len({it.group for it in its})} 組）  {detail}")
    print()


# ───────────────────────── 切塊 ─────────────────────────


def tile_windows(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """把 w×h 的大圖切成 size×size、重疊 overlap 的視窗，邊緣視窗貼齊影像邊界。"""

    def starts(n):
        if n <= size:
            return [0]
        step = max(1, int(size * (1 - overlap)))
        s = list(range(0, n - size + 1, step))
        if s[-1] != n - size:
            s.append(n - size)
        return s

    return [(x, y, min(x + size, w), min(y + size, h)) for y in starts(h) for x in starts(w)]


def clip_boxes(boxes, win, min_visible: float) -> tuple[list, int]:
    """把框裁進視窗並轉成視窗座標。回傳 (保留的框, 被切到太小而捨棄的數量)。"""
    x0, y0, x1, y1 = win
    kept, dropped = [], 0
    for c, a, b, cc, d in boxes:
        ix1, iy1, ix2, iy2 = max(a, x0), max(b, y0), min(cc, x1), min(d, y1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        area = max((cc - a) * (d - b), 1e-9)
        if (ix2 - ix1) * (iy2 - iy1) / area < min_visible:
            dropped += 1
            continue
        kept.append((c, ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0))
    return kept, dropped


def should_tile(ds: Dataset, mode: str, imgsz: int) -> bool:
    if ds.task != "detect" or mode == "off":
        if mode == "on" and ds.task != "detect":
            print("⚠️ 切塊目前只支援 detect 任務，已關閉")
        return False
    if mode == "on":
        return True
    sample = [it.image for it in ds.items[:: max(1, len(ds.items) // 30)]][:30]
    med = float(np.median([max(image_size(p)) for p in sample])) if sample else 0
    on = med > 1.5 * imgsz
    print(f"影像長邊中位數 {med:.0f}px，模型輸入 {imgsz}px → {'自動開啟切塊' if on else '不切塊'}")
    return on


# ───────────────────────── 寫出 Ultralytics 格式 ─────────────────────────


def _reset_dir(out: Path) -> None:
    if out.exists():
        if not (out / DATASET_MARKER).exists() and any(out.iterdir()):
            raise SystemExit(f"{out} 已存在且不是本程式建立的資料夾，請換一個 --data-dir")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / DATASET_MARKER).touch()


def _box_lines(boxes, w, h) -> list[str]:
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def _seg_lines(it: Item, w, h) -> list[str]:
    polys = it.polygons or [(c, [x1, y1, x2, y1, x2, y2, x1, y2]) for c, x1, y1, x2, y2 in it.boxes]
    lines = []
    for c, pts in polys:
        norm = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(pts)]
        lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in norm))
    return lines


def _write_tiles(it: Item, boxes, out: Path, stem: str, tile: TileCfg) -> int:
    img = cv2.imread(str(it.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 讀不到影像：{it.image}")
        return 0
    h, w = img.shape[:2]
    ext = ".jpg" if it.image.suffix.lower() in (".jpg", ".jpeg") else ".png"  # 原圖無損就維持無損
    n = 0
    for win in tile_windows(w, h, tile.size, tile.overlap):
        kept, dropped = clip_boxes(boxes, win, tile.min_visible)
        name = f"{stem}__{win[0]}_{win[1]}"
        if not kept and it.split == "train":
            # 只含殘缺缺陷的切塊不能當負樣本；純背景切塊只保留一部分
            if dropped or hash_fraction(name) >= tile.empty_ratio:
                continue
        x0, y0, x1, y1 = win
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else []
        cv2.imwrite(str(out / "images" / it.split / f"{name}{ext}"), img[y0:y1, x0:x1], params)
        (out / "labels" / it.split / f"{name}.txt").write_text("\n".join(_box_lines(kept, x1 - x0, y1 - y0)))
        n += 1
    return n


def write_dataset(ds: Dataset, out: Path, tile: TileCfg | None = None) -> Path:
    """寫成 Ultralytics 格式（影像用符號連結，不複製）。回傳 data.yaml（分類任務回傳資料夾）。

    另外寫出 eval.json：val/test 的「原始整張影像 + 像素座標標註」，評估與調門檻都用它，
    所以即使訓練用的是切塊，評估仍是在整張影像上做（和實際上線時一樣）。
    """
    _reset_dir(out)
    eval_sets = {s: [] for s in SPLITS}
    if ds.task == "classify":
        present = {s: {it.label for it in ds.items if it.split == s} for s in SPLITS}
        keep = [n for n in ds.names if all(n in present[s] for s in SPLITS)]
        if len(keep) < len(ds.names):
            print(f"⚠️ 以下類別無法同時出現在 train/val/test，已排除：{sorted(set(ds.names) - set(keep))}")
        ds.names = keep
        for it in ds.items:
            if it.label in keep:
                link_or_copy(it.image, out / it.split / it.label / f"{unique_stem(it.image)}{it.image.suffix.lower()}")
                eval_sets[it.split].append({"image": str(it.image), "label": it.label})
        data_ref = out
    else:
        for s in SPLITS:
            (out / "images" / s).mkdir(parents=True)
            (out / "labels" / s).mkdir(parents=True)
        n_units = Counter()
        for it in ds.items:
            boxes = item_boxes(it)
            stem = unique_stem(it.image)
            if tile:
                n_units[it.split] += _write_tiles(it, boxes, out, stem, tile)
            else:
                link_or_copy(it.image, out / "images" / it.split / f"{stem}{it.image.suffix.lower()}")
                w, h = image_size(it.image)
                lines = _seg_lines(it, w, h) if ds.task == "segment" else _box_lines(boxes, w, h)
                (out / "labels" / it.split / f"{stem}.txt").write_text("\n".join(lines))
                n_units[it.split] += 1
            eval_sets[it.split].append({"image": str(it.image), "boxes": [[float(v) for v in b] for b in boxes]})
        if tile:
            print("切塊數量：" + "，".join(f"{s} {n_units[s]}" for s in SPLITS))
        data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val"}
        if eval_sets["test"]:
            data["test"] = "images/test"
        data["names"] = dict(enumerate(ds.names))
        data_ref = out / "data.yaml"
        data_ref.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    meta = {"task": ds.task, "names": ds.names, "tile": asdict(tile) if tile else None, **eval_sets}
    (out / "eval.json").write_text(json.dumps(meta, ensure_ascii=False))
    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "split"])
        w.writerows([str(it.image), it.group, it.split] for it in ds.items)
    return data_ref


# ───────────────────────── 訓練 ─────────────────────────


def train_model(args, data_ref: Path, task: str, aug: dict) -> Path:
    model = load_model(args.model or default_weights(task, args.size))
    train_args = dict(
        data=str(data_ref), epochs=args.epochs, imgsz=args.train_imgsz, batch=args.batch, patience=args.patience,
        project=str(args.out.resolve()), name="train", exist_ok=True, seed=args.seed, plots=True,
    )
    train_args.update(aug)
    train_args.update(args.extra)  # --set key=value 可覆蓋任何 Ultralytics 訓練參數
    train_args.update(device_kwargs(args))
    for k in ("workers", "time"):
        if getattr(args, k) is not None:
            train_args[k] = getattr(args, k)
    model.train(**train_args)
    weights = Path(model.trainer.save_dir) / "weights"
    del model  # 釋放 trainer（資料載入器、優化器、EMA），否則評估時記憶體會疊加到 OOM
    free_memory()
    return weights / "best.pt" if (weights / "best.pt").exists() else weights / "last.pt"


def free_memory() -> None:
    gc.collect()
    try:  # glibc 不會主動把釋放的記憶體還給系統，訓練後 RSS 會一直停在高點
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # 非 glibc（macOS / Windows）
        pass
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ───────────────────────── 推論 ─────────────────────────


def _class_map(model_names: dict, names: list) -> dict:
    """模型類別索引 → 資料集類別索引（依名稱對應；新舊模型類別不同時也能比較）。"""
    idx = {n: i for i, n in enumerate(names)}
    return {int(mi): idx[n] for mi, n in model_names.items() if n in idx}


def _boxes_from_result(r, offset, m2d) -> np.ndarray:
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6))
    cls = b.cls.cpu().numpy().astype(int)
    keep = np.array([c in m2d for c in cls])
    if not keep.any():
        return np.zeros((0, 6))
    ox, oy = offset
    xyxy = b.xyxy.cpu().numpy()[keep] + np.array([ox, oy, ox, oy])
    return np.column_stack([[m2d[c] for c in cls[keep]], b.conf.cpu().numpy()[keep], xyxy])


def nms_ios(preds: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """依類別做 NMS，重疊度用「交集 / 較小框面積」，能合併切塊邊界上重複或被切開的框（同 SAHI 的 IOS）。"""
    if len(preds) == 0:
        return preds
    keep = []
    for c in np.unique(preds[:, 0]):
        p = preds[preds[:, 0] == c]
        p = p[np.argsort(-p[:, 1])]
        area = np.clip(p[:, 4] - p[:, 2], 0, None) * np.clip(p[:, 5] - p[:, 3], 0, None)
        removed = np.zeros(len(p), bool)
        for i in range(len(p)):
            if removed[i]:
                continue
            keep.append(p[i])
            rest = slice(i + 1, None)
            iw = np.clip(np.minimum(p[i, 4], p[rest, 4]) - np.maximum(p[i, 2], p[rest, 2]), 0, None)
            ih = np.clip(np.minimum(p[i, 5], p[rest, 5]) - np.maximum(p[i, 3], p[rest, 3]), 0, None)
            ios = iw * ih / np.clip(np.minimum(area[i], area[rest]), 1e-9, None)
            removed[rest] |= ios > thr
    return np.array(keep)


def predict_boxes(model, entries: list, args, tile: TileCfg | None, m2d: dict) -> list[np.ndarray]:
    """每張影像回傳 (N, 6) 陣列：[類別, 信心, x1, y1, x2, y2]（資料集類別索引、原圖像素座標）。"""
    if not entries:
        return []
    kw = dict(conf=CONF_FLOOR, verbose=False, **device_kwargs(args))
    if tile is None:
        paths = [e["image"] for e in entries]
        results = model.predict(paths, imgsz=args.train_imgsz, stream=True, batch=args.pred_batch, **kw)
        return [_boxes_from_result(r, (0, 0), m2d) for r in results]
    out = []
    for e in entries:
        img = cv2.imread(e["image"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ 讀不到影像：{e['image']}")
            out.append(np.zeros((0, 6)))
            continue
        h, w = img.shape[:2]
        wins = tile_windows(w, h, tile.size, tile.overlap)
        parts = []
        for i in range(0, len(wins), args.pred_batch):
            chunk = wins[i : i + args.pred_batch]
            crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]
            for r, win in zip(model.predict(crops, imgsz=tile.size, **kw), chunk):
                parts.append(_boxes_from_result(r, win[:2], m2d))
        out.append(nms_ios(np.concatenate(parts)) if parts else np.zeros((0, 6)))
    return out


def predict_probs(model, entries: list, names: list, args) -> np.ndarray:
    out = np.zeros((len(entries), len(names)), np.float32)
    if not entries:
        return out
    m2d = _class_map(model.names, names)
    results = model.predict([e["image"] for e in entries], imgsz=args.train_imgsz, stream=True, verbose=False,
                            batch=args.pred_batch, **device_kwargs(args))
    for i, r in enumerate(results):
        p = r.probs.data.cpu().numpy()
        for mi, di in m2d.items():
            out[i, di] = p[mi]
    return out


# ───────────────────────── 指標與門檻 ─────────────────────────


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (M,4)、b: (N,4) xyxy → (M,N) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def gt_array(entry) -> np.ndarray:
    return np.asarray(entry.get("boxes", []), float).reshape(-1, 5)


def collect_scores(preds, gts, nc: int, iou: float, level: str) -> list[list[float]]:
    """每個正樣本「最好被抓到的信心值」（0 = 完全沒抓到）。

    level="object"：每個標註缺陷/病灶各算一次（同類別且 IoU ≥ iou 的預測中最高信心）。
    level="image" ：每張含該類別的影像算一次（該類別預測的最高信心，不看位置）。
    """
    scores = [[] for _ in range(nc)]
    for p, g in zip(preds, gts):
        if level == "image":
            for c in sorted(set(g[:, 0].astype(int))):
                pc = p[p[:, 0] == c] if len(p) else p
                scores[c].append(float(pc[:, 1].max()) if len(pc) else 0.0)
            continue
        ious = iou_matrix(g[:, 1:5], p[:, 2:6]) if len(p) else np.zeros((len(g), 0))
        for j, (c, *_) in enumerate(g):
            ok = (p[:, 0] == c) & (ious[j] >= iou) if len(p) else np.zeros(0, bool)
            scores[int(c)].append(float(p[ok, 1].max()) if ok.any() else 0.0)
    return scores


def threshold_for_recall(scores, target: float) -> tuple[float, bool]:
    """最大的門檻 t，使 scores 中 ≥ t 的比例達到 target。回傳 (t, 是否達得到)。"""
    s = np.sort(np.asarray(scores, float))[::-1]
    k = max(1, math.ceil(target * len(s) - 1e-9))
    t = float(s[k - 1])
    if t <= 0:
        return CONF_FLOOR, False
    return math.floor(t * 1e4) / 1e4, True  # 無條件捨去，避免四捨五入後反而漏掉邊界樣本


def clamp_threshold(t: float, ok: bool | None, min_t: float, max_t: float) -> tuple[float, bool | None]:
    """把門檻限制在 [min_t, max_t]。達標需要低於 min_t 的門檻時，停在 min_t 並視為未達標，
    不為了召回率把門檻降到誤報暴增的程度；是否上線交給人判斷。"""
    if t < min_t:
        return min_t, False
    return min(t, max_t), ok


def tune_thresholds(preds, gts, names, target: float, iou: float, level: str, max_t: float,
                    min_t: float = 0.0) -> dict:
    """逐類調門檻。上限 max_t：val 樣本少時，「剛好達標的最大門檻」很容易過度貼合 val，
    到 test 反而漏掉；設上限讓調門檻只會讓系統更敏感、不會比預設更不敏感。下限 min_t 見 clamp_threshold。"""
    out = {}
    for c, s in enumerate(collect_scores(preds, gts, len(names), iou, level)):
        if not s:
            out[names[c]] = {"threshold": max(min(0.25, max_t), min_t), "reachable": None, "val_recall": None, "n_val": 0}
            continue
        t, ok = clamp_threshold(*threshold_for_recall(s, target), min_t, max_t)
        out[names[c]] = {"threshold": t, "reachable": ok, "val_recall": float(np.mean(np.array(s) >= t)), "n_val": len(s)}
    return out


def object_metrics(preds, gts, thr: np.ndarray, iou: float, names: list) -> dict:
    nc = len(names)
    hit, tot = np.zeros(nc), np.zeros(nc)
    tp_pred = fp_pred = 0
    for p, g in zip(preds, gts):
        if len(p):
            p = p[p[:, 1] >= thr[p[:, 0].astype(int)]]
        if len(g):
            tot += np.bincount(g[:, 0].astype(int), minlength=nc)
        tp = 0
        if len(p) and len(g):
            ok = (iou_matrix(g[:, 1:5], p[:, 2:6]) >= iou) & (g[:, [0]] == p[None, :, 0])
            hit += np.bincount(g[ok.any(1), 0].astype(int), minlength=nc)
            tp = int(ok.any(0).sum())
        tp_pred += tp
        fp_pred += len(p) - tp
    recall = float(hit.sum() / tot.sum()) if tot.sum() else None
    return {
        "n_objects": int(tot.sum()),
        "recall": recall,
        "escape_rate": None if recall is None else 1 - recall,
        "precision": tp_pred / (tp_pred + fp_pred) if tp_pred + fp_pred else None,
        "fp_per_image": fp_pred / len(preds) if preds else None,
        "per_class": {names[c]: {"n": int(tot[c]), "recall": float(hit[c] / tot[c]) if tot[c] else None} for c in range(nc)},
    }


def binary_metrics(y_pos: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int((y_pos & y_pred).sum())
    fn = int((y_pos & ~y_pred).sum())
    fp = int((~y_pos & y_pred).sum())
    tn = int((~y_pos & ~y_pred).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "n_pos": tp + fn, "n_neg": tn + fp, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "specificity": spec,
        "escape_rate": None if sens is None else 1 - sens,
        "overkill_rate": None if spec is None else 1 - spec,
    }


def image_metrics(preds, gts, thr: np.ndarray) -> dict:
    """整張影像判 NG / 異常：任何一個預測超過該類門檻即判陽性。"""
    y_pos = np.array([len(g) > 0 for g in gts])
    y_pred = np.array([bool(len(p)) and bool((p[:, 1] >= thr[p[:, 0].astype(int)]).any()) for p in preds])
    return binary_metrics(y_pos, y_pred)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann–Whitney 形式的 ROC AUC（同分取平均名次）。"""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.bincount(inv, weights=ranks)[inv] / counts[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pick_normal(names: list, explicit: str | None, candidates: tuple) -> str | None:
    if explicit:
        if explicit in names:
            return explicit
        print(f"⚠️ --normal-class {explicit!r} 不在類別 {names} 中，忽略")
        return None
    norm = {re.sub(r"[\s\-]+", "_", n.lower()): n for n in names}
    return next((norm[c] for c in candidates if c in norm), None)


# ───────────────────────── 評估 ─────────────────────────


def _ultralytics_val(model, data_ref, args, imgsz: int, split: str) -> dict:
    try:
        m = model.val(data=str(data_ref), split=split, imgsz=imgsz, batch=args.pred_batch, plots=True, verbose=False,
                      project=str(args.out.resolve()), name=f"val_{split}", exist_ok=True, **device_kwargs(args))
    except Exception as e:  # 評估失敗不影響後續流程
        print(f"⚠️ Ultralytics val 失敗：{e}")
        return {}
    res = {}
    if hasattr(m, "top1"):
        res.update(top1=float(m.top1), top5=float(m.top5))
    box, seg = getattr(m, "box", None), getattr(m, "seg", None)
    if box is not None:
        res.update(map50=float(box.map50), map50_95=float(box.map))
    if seg is not None:
        res.update(mask_map50=float(seg.map50), mask_map50_95=float(seg.map))
    return res


def evaluate(model_path, data_ref: Path, meta: dict, args, domain: Domain) -> dict:
    """在 val 上逐類調門檻（達到 --target-recall），再用同一組門檻在 test 上報告。"""
    model = load_model(model_path)
    test_split = "test" if meta["test"] else "val"
    if test_split == "val":
        print("⚠️ 沒有 test split，改用 val 報告（數字會偏樂觀）")
    if meta["task"] == "classify":
        report = _evaluate_cls(model, meta, args, domain, test_split)
        report["ultralytics"] = _ultralytics_val(model, data_ref, args, args.train_imgsz, test_split)
        report["key"]["quality"] = report["accuracy"]
    else:
        report = _evaluate_det(model, meta, args, test_split)
        tile = meta["tile"]
        report["ultralytics"] = _ultralytics_val(model, data_ref, args, tile["size"] if tile else args.train_imgsz, test_split)
        report["key"]["quality"] = report["ultralytics"].get("map50")
    report.update(model=str(model_path), task=meta["task"], names=meta["names"], test_split=test_split,
                  n_val=len(meta["val"]), n_test=len(meta[test_split]))
    return report


def _evaluate_det(model, meta, args, test_split) -> dict:
    names = meta["names"]
    tile = TileCfg(**meta["tile"]) if meta["tile"] else None
    m2d = _class_map(model.names, names)
    val_gts = [gt_array(e) for e in meta["val"]]
    val_preds = predict_boxes(model, meta["val"], args, tile, m2d)
    thresholds = tune_thresholds(val_preds, val_gts, names, args.target_recall, args.match_iou, args.tune_level,
                                 args.max_threshold, args.min_threshold)
    thr = np.array([thresholds[n]["threshold"] for n in names])
    test_gts = [gt_array(e) for e in meta[test_split]]
    test_preds = val_preds if test_split == "val" else predict_boxes(model, meta[test_split], args, tile, m2d)
    obj = object_metrics(test_preds, test_gts, thr, args.match_iou, names)
    img = image_metrics(test_preds, test_gts, thr)
    use_overkill = img["n_neg"] > 0
    return {
        "thresholds": thresholds,
        "object": obj,
        "image": img,
        "key": {
            "escape": obj["escape_rate"],
            "false_alarm": img["overkill_rate"] if use_overkill else obj["fp_per_image"],
            "false_alarm_kind": "overkill_rate" if use_overkill else "fp_per_image",
        },
    }


def _evaluate_cls(model, meta, args, domain, test_split) -> dict:
    names = meta["names"]
    nc = len(names)
    y_val = np.array([names.index(e["label"]) for e in meta["val"]], int)
    y_test = np.array([names.index(e["label"]) for e in meta[test_split]], int)
    p_val = predict_probs(model, meta["val"], names, args)
    p_test = p_val if test_split == "val" else predict_probs(model, meta[test_split], names, args)
    pred = p_test.argmax(1)
    cm = np.zeros((nc, nc), int)
    np.add.at(cm, (y_test, pred), 1)
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(nc)]
    precision = [float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else None for i in range(nc)]
    valid_r = [r for r in recall if r is not None]
    report = {
        "accuracy": float((pred == y_test).mean()) if len(y_test) else None,
        "per_class": {n: {"n": int(cm[i].sum()), "recall": recall[i], "precision": precision[i]} for i, n in enumerate(names)},
        "confusion_matrix": cm.tolist(),
        "key": {"escape": 1 - float(np.mean(valid_r)) if valid_r else None, "false_alarm": None,
                "false_alarm_kind": None},
    }
    normal = pick_normal(names, args.normal_class, domain.normal_names)
    if normal is None:
        print("未指定 / 找不到「正常」類別，只計算多類別指標（可用 --normal-class 指定）")
        report["thresholds"] = {}
        return report
    ni = names.index(normal)
    s_val, s_test = 1 - p_val[:, ni], 1 - p_test[:, ni]  # 異常分數 = 1 − P(正常)
    t, ok = threshold_for_recall(s_val[y_val != ni], args.target_recall) if (y_val != ni).any() else (0.5, None)
    t, ok = clamp_threshold(t, ok, args.min_threshold, args.max_threshold)
    b = binary_metrics(y_test != ni, s_test >= t)
    b["auc"] = roc_auc(y_test != ni, s_test)
    report["binary"] = b
    report["thresholds"] = {"normal_class": normal, "abnormal_threshold": t, "reachable": ok}
    report["key"] = {"escape": b["escape_rate"], "false_alarm": b["overkill_rate"], "false_alarm_kind": "overkill_rate"}
    return report


def compare(new: dict, old: dict, tol: float = 0.002) -> tuple[bool, list]:
    """上線關卡：漏檢不能變多、誤報不能變多、整體品質不能明顯下降。"""
    rows, passed = [], True
    for key, better in (("escape", "lower"), ("false_alarm", "lower"), ("quality", "higher")):
        a, b = new["key"].get(key), old["key"].get(key)
        if a is None or b is None:
            continue
        worse = a > b + tol if better == "lower" else a < b - max(tol, 0.01)
        passed &= not worse
        rows.append((key, b, a, not worse))
    return passed, rows


# ───────────────────────── 報告 ─────────────────────────


MIN_RELIABLE_N = 30  # 樣本數低於此值的比率只供參考（例如 2 張算出的 50%）


def _small(n: int) -> str:
    """樣本太少時附加在指標名稱後的提示。"""
    return f"，⚠️ 少於 {MIN_RELIABLE_N} 張，僅供參考" if 0 < n < MIN_RELIABLE_N else ""


def _floor_note(t: float, args) -> str:
    return f"（停在下限 {args.min_threshold:g}）" if t <= args.min_threshold else ""


def _fmt(v, pct=True) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%" if pct else f"{v:.4f}"


def write_report(args, domain: Domain, report: dict, baseline: dict | None, gate) -> Path:
    out = args.out
    key = report["key"]
    fa_kind = key.get("false_alarm_kind")
    fa_label = {"overkill_rate": domain.false_alarm_label, "fp_per_image": "誤報數 / 每張影像"}.get(fa_kind)
    esc_label = domain.escape_label
    if report["task"] == "classify" and not report.get("binary"):
        esc_label = "1 − 各類別平均召回率"
    lines = [f"# {domain.name} 檢測報告", ""]
    if domain.disclaimer:
        lines += [f"> {domain.disclaimer}", ""]
    lines += [
        f"- 模型：`{report['model']}`",
        f"- 任務：{report['task']}；類別：{', '.join(report['names'])}",
        f"- 調門檻：val {report['n_val']} 張，目標召回率 {args.target_recall:.0%}"
        + (f"（{'每個缺陷/病灶' if args.tune_level == 'object' else '每張影像'}）" if report["task"] != "classify" else ""),
        f"- 報告：{report['test_split']} {report['n_test']} 張",
        "",
        "## 主要指標（test）",
        "",
        "| 指標 | 數值 |",
        "|---|---|",
        f"| {esc_label} | {_fmt(key['escape'])} |",
    ]
    if fa_label:
        lines.append(f"| {fa_label} | {_fmt(key['false_alarm'], pct=fa_kind == 'overkill_rate')} |")
    u = report.get("ultralytics", {})
    for k, label in (("map50", "mAP50"), ("map50_95", "mAP50-95"), ("mask_map50", "Mask mAP50"),
                     ("top1", "Top-1 準確率"), ("top5", "Top-5 準確率")):
        if k in u:
            lines.append(f"| {label} | {_fmt(u[k], pct=False)} |")
    if report["task"] == "classify":
        lines += _cls_report_lines(report, args)
    else:
        lines += _det_report_lines(report, args, show_fp=fa_kind != "fp_per_image")
    if baseline is not None:
        lines += ["", "## 上線關卡（與舊模型比較，同一份 test）", "", f"舊模型：`{baseline['model']}`", "",
                  "| 指標 | 舊模型 | 新模型 | 結果 |", "|---|---|---|---|"]
        labels = {"escape": esc_label, "false_alarm": fa_label, "quality": "整體品質（mAP50 / 準確率）"}
        for k, b, a, ok in gate[1]:
            pct = k == "escape" or (k == "false_alarm" and fa_kind == "overkill_rate")
            lines.append(f"| {labels[k]} | {_fmt(b, pct)} | {_fmt(a, pct)} | {'✅' if ok else '❌'} |")
        lines += ["", f"**{'✅ 通過：可以進入影子模式 / 灰度上線' if gate[0] else '❌ 未通過：維持舊模型'}**"]
    lines += [
        "", "## 輸出檔案", "",
        "- `best.pt`：最佳權重；`thresholds.json`：上線用的每類門檻",
        "- `train/`：訓練曲線、混淆矩陣；`val_*/`：Ultralytics 評估圖表",
        "- `split_manifest.csv`：每張影像的分組與 split（黃金測試集清單）",
        "- `review_queue/`：需要人工複判的影像與預標註（有給 --unlabeled 時）",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"report": report, "baseline": baseline, "gate_passed": gate[0] if gate else None, "args": _jsonable(args)}
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _det_report_lines(report, args, show_fp: bool) -> list[str]:
    img = report["image"]
    lines = [
        f"| 缺陷/病灶層級召回率（IoU ≥ {args.match_iou}） | {_fmt(report['object']['recall'])} |",
        f"| 預測精確率 | {_fmt(report['object']['precision'])} |",
    ]
    if show_fp:
        lines.append(f"| 誤報數 / 每張影像 | {_fmt(report['object']['fp_per_image'], pct=False)} |")
    lines += [
        f"| 影像層級敏感度（{img['n_pos']} 張有標註{_small(img['n_pos'])}） | {_fmt(img['sensitivity'])} |",
        f"| 影像層級特異度（{img['n_neg']} 張無標註{_small(img['n_neg'])}） | {_fmt(img['specificity'])} |",
        "", "## 每類門檻（val 調出）與 test 召回率", "",
        "| 類別 | 門檻 | val 樣本 | val 召回 | test 樣本 | test 召回 |", "|---|---|---|---|---|---|",
    ]
    for n in report["names"]:
        t, pc = report["thresholds"][n], report["object"]["per_class"][n]
        flag = f" ⚠️ 達不到目標{_floor_note(t['threshold'], args)}" if t["reachable"] is False else (" （val 無樣本，用預設）" if t["reachable"] is None else "")
        lines.append(f"| {n} | {t['threshold']:.4f}{flag} | {t['n_val']} | {_fmt(t['val_recall'])} | {pc['n']} | {_fmt(pc['recall'])} |")
    if img["n_neg"] == 0:
        lines += ["", "> test 中沒有無缺陷 / 正常影像，無法計算特異度與過殺率；建議加入良品 / 正常影像。"]
    elif img["n_neg"] < MIN_RELIABLE_N:
        lines += ["", f"> test 中只有 {img['n_neg']} 張無缺陷 / 正常影像，特異度與過殺率（偽陽性率）的統計誤差很大，"
                  f"不能拿來判斷模型好壞；建議至少 {MIN_RELIABLE_N} 張，最好數百張。"]
    return lines


def _cls_report_lines(report, args) -> list[str]:
    lines = [] if "top1" in report.get("ultralytics", {}) else [f"| 準確率 | {_fmt(report['accuracy'])} |"]
    b = report.get("binary")
    if b:
        t = report["thresholds"]
        flag = f" ⚠️ 達不到目標召回率{_floor_note(t['abnormal_threshold'], args)}" if t["reachable"] is False else ""
        lines += [
            f"| 敏感度（異常判為異常{_small(b.get('n_pos', 0))}） | {_fmt(b['sensitivity'])} |",
            f"| 特異度（正常判為正常{_small(b.get('n_neg', 0))}） | {_fmt(b['specificity'])} |",
            f"| ROC AUC | {_fmt(b['auc'], pct=False)} |",
            "", f"正常類別：`{t['normal_class']}`；異常門檻（1 − P(正常) ≥ t）：**{t['abnormal_threshold']:.4f}**{flag}",
        ]
    names = report["names"]
    lines += ["", "## 每類結果（argmax）", "", "| 類別 | 樣本 | 召回率 | 精確率 |", "|---|---|---|---|"]
    for n in names:
        pc = report["per_class"][n]
        lines.append(f"| {n} | {pc['n']} | {_fmt(pc['recall'])} | {_fmt(pc['precision'])} |")
    lines += ["", "## 混淆矩陣（列 = 真實，欄 = 預測）", "", "| | " + " | ".join(names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for n, row in zip(names, report["confusion_matrix"]):
        lines.append(f"| **{n}** | " + " | ".join(map(str, row)) + " |")
    return lines


def _jsonable(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if not k.startswith("_")}


def write_thresholds(args, report: dict) -> None:
    if report["task"] == "classify":
        data = report["thresholds"]
    else:
        data = {n: v["threshold"] for n, v in report["thresholds"].items()}
    (args.out / "thresholds.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ───────────────────────── 主動學習：複判佇列 ─────────────────────────


def build_review_queue(model_path, src, meta: dict, report: dict, args) -> Path | None:
    """對未標註影像推論，挑出「最接近門檻、模型最沒把握」的影像給人複判，再加一些隨機抽樣。

    隨機抽樣用來發現模型「很有把握但其實錯了」的盲點，不能省。
    輸出 YOLO 格式預標註，可直接匯入 CVAT / Label Studio，人只需要修正。
    """
    images = iter_images(src)
    if not images:
        print(f"⚠️ {src} 裡沒有影像，略過複判佇列")
        return None
    q = args.out / "review_queue"
    if q.exists():
        shutil.rmtree(q)
    model = load_model(model_path)
    names = meta["names"]
    entries = [{"image": str(p)} for p in images]
    rows = []
    if meta["task"] == "classify":
        probs = predict_probs(model, entries, names, args)
        t = report["thresholds"]
        if t.get("normal_class"):
            score = 1 - probs[:, names.index(t["normal_class"])]
            margin = np.abs(score - t["abnormal_threshold"])
        else:
            top2 = np.sort(probs, 1)[:, -2:] if probs.shape[1] > 1 else np.zeros((len(probs), 2))
            margin = top2[:, 1] - top2[:, 0]
        preds = [names[i] for i in probs.argmax(1)]
    else:
        tile = TileCfg(**meta["tile"]) if meta["tile"] else None
        boxes = predict_boxes(model, entries, args, tile, _class_map(model.names, names))
        thr = np.array([report["thresholds"][n]["threshold"] for n in names])
        margin = np.array([np.abs(b[:, 1] - thr[b[:, 0].astype(int)]).min() if len(b) else 1.0 for b in boxes])
        preds = ["NG" if len(b) and (b[:, 1] >= thr[b[:, 0].astype(int)]).any() else "OK" for b in boxes]
    order = np.argsort(margin, kind="stable")
    k = min(args.review_k, len(images))
    picks = [(int(i), "uncertain") for i in order[:k]]
    rest = [int(i) for i in order[k:]]
    picks += [(i, "random") for i in random.Random(args.seed).sample(rest, min(max(k // 10, 1), len(rest)))]
    for i, reason in picks:
        p = images[i]
        stem = unique_stem(p)
        if meta["task"] == "classify":
            copy_file(p, q / preds[i] / f"{stem}{p.suffix.lower()}")
        else:
            copy_file(p, q / "images" / f"{stem}{p.suffix.lower()}")
            w, h = image_size(p)
            b = boxes[i]
            b = b[b[:, 1] >= 0.5 * thr[b[:, 0].astype(int)]] if len(b) else b  # 灰色地帶也放進預標註讓人確認
            lines = _box_lines([(int(r[0]), *r[2:6]) for r in b], w, h)
            if meta["task"] == "segment" and meta["tile"] is None:
                lines = _seg_prelabels(model, p, names, thr, args)
            (q / "labels").mkdir(parents=True, exist_ok=True)
            (q / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        rows.append([str(p), reason, f"{margin[i]:.4f}", preds[i]])
    with open(q / "queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "reason", "distance_to_threshold", "prediction"])
        w.writerows(rows)
    if meta["task"] != "classify":
        (q / "classes.txt").write_text("\n".join(names))
        (q / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images", "val": "images",
                                                     "names": dict(enumerate(names))}, allow_unicode=True))
    shutil.make_archive(str(q), "zip", q)
    print(f"複判佇列：{len(picks)} 張（{k} 張最不確定 + {len(picks) - k} 張隨機）→ {q}.zip")
    return q


def copy_file(src: Path, dst: Path) -> None:
    """複判佇列要能下載，所以複製實體檔案而不是連結。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _seg_prelabels(model, path, names, thr, args) -> list[str]:
    r = model.predict(str(path), imgsz=args.train_imgsz, conf=CONF_FLOOR, verbose=False, **device_kwargs(args))[0]
    if r.masks is None:
        return []
    m2d = _class_map(model.names, names)
    lines = []
    for poly, c, conf in zip(r.masks.xyn, r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        if c in m2d and conf >= 0.5 * thr[m2d[c]] and len(poly) >= 3:
            lines.append(f"{m2d[c]} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
    return lines


# ───────────────────────── 匯出 ─────────────────────────


def export_model(best: Path, fmt: str, imgsz: int, args) -> str | None:
    if not fmt or fmt.lower() == "none":
        return None
    try:
        return str(load_model(best).export(format=fmt, imgsz=imgsz, **device_kwargs(args)))
    except Exception as e:
        print(f"⚠️ 匯出 {fmt} 失敗：{e}")
        return None


# ───────────────────────── 命令列與主流程 ─────────────────────────


def add_common_args(p: argparse.ArgumentParser, *, target_recall: float, tile_default: str) -> None:
    g = p.add_argument_group("訓練")
    g.add_argument("--task", choices=["auto", "detect", "segment", "classify"], default="auto")
    g.add_argument("--model", default=None, help="起始權重；預設 yolo26<size>(-seg/-cls).pt。接續上一輪時給上一輪的 best.pt")
    g.add_argument("--size", default="s", choices=list("nsmlx"), help="模型大小（n 最快、x 最準）")
    g.add_argument("--imgsz", type=int, default=None, help="模型輸入尺寸；預設 detect/segment 640、classify 224")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch", type=int, default=-1, help="-1 = 依 GPU 記憶體自動決定")
    g.add_argument("--patience", type=int, default=30, help="val 連續幾個 epoch 沒進步就提早停止")
    g.add_argument("--time", type=float, default=None, help="訓練時數上限（小時，會覆蓋 epochs）；Kaggle 單次最長約 12 小時")
    g.add_argument("--device", default=None, help="例如 0、0,1、cpu；預設自動")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="覆蓋任何 Ultralytics 訓練參數，例如 --set lr0=0.005 mosaic=0.5")
    g.add_argument("--skip-train", action="store_true", help="不訓練，直接用 --model 評估 / 產生複判佇列")
    g = p.add_argument_group("資料切分與切塊")
    g.add_argument("--val-pct", type=float, default=0.15)
    g.add_argument("--test-pct", type=float, default=0.15)
    g.add_argument("--resplit", action="store_true", help="忽略來源資料的 train/val/test，依分組重新切")
    g.add_argument("--tile", choices=["auto", "on", "off"], default=tile_default, help="大圖切塊（只用於 detect）")
    g.add_argument("--tile-size", type=int, default=None, help="切塊尺寸，預設等於 --imgsz")
    g.add_argument("--tile-overlap", type=float, default=0.2, help="切塊重疊比例；缺陷尺寸應小於重疊寬度")
    g.add_argument("--empty-tile-ratio", type=float, default=0.3, help="訓練集保留多少比例的無缺陷切塊")
    g = p.add_argument_group("評估與閉環")
    g.add_argument("--target-recall", type=float, default=target_recall, help="調門檻的目標召回率（漏檢率 = 1 − 這個值）")
    g.add_argument("--max-threshold", type=float, default=0.5,
                   help="門檻上限；避免小樣本的 val 調出過高門檻，讓系統比預設更不敏感")
    g.add_argument("--min-threshold", type=float, default=0.1,
                   help="門檻下限；達不到目標召回率時停在這裡並標示未達標，而不是一路降低讓誤報暴增")
    g.add_argument("--match-iou", type=float, default=0.3, help="預測框與標註框算「抓到」的 IoU 門檻")
    g.add_argument("--tune-level", choices=["object", "image"], default="object",
                   help="object：每個缺陷都要抓到；image：只看整張影像判 OK/NG")
    g.add_argument("--normal-class", default=None, help="分類任務中代表正常 / 良品的類別名稱（預設自動辨識）")
    g.add_argument("--baseline", default=None, help="舊模型權重；會在同一份 test 上比較，判斷能否上線")
    g.add_argument("--unlabeled", default=None, help="未標註影像資料夾；產生複判佇列（主動學習）")
    g.add_argument("--review-k", type=int, default=200, help="複判佇列挑幾張最不確定的影像")
    g.add_argument("--export", default="onnx", help="匯出格式：onnx / engine / openvino / tflite / coreml / none")
    g = p.add_argument_group("路徑")
    g.add_argument("--out", default=None, help="輸出資料夾（Kaggle 預設 /kaggle/working/<領域>）")
    g.add_argument("--data-dir", default=None, help="轉換後資料集的位置（Kaggle 預設 /tmp/vi-datasets/<領域>）")


def parse_args(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args, unknown = parser.parse_known_args(argv)
    unknown = [u for u in unknown if not (u.startswith("-f") or u.endswith(".json"))]  # Jupyter 會塞 -f kernel.json
    if unknown:
        print(f"⚠️ 忽略不認得的參數：{unknown}")
    if not 0 <= args.min_threshold <= args.max_threshold <= 1:
        raise SystemExit(f"門檻範圍不合理：--min-threshold {args.min_threshold} 必須介於 0 和 --max-threshold {args.max_threshold} 之間")
    args.extra = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        args.extra[k] = yaml.safe_load(v)
    import torch

    args.pred_batch = 16 if torch.cuda.is_available() else 4  # CPU 推論大批次很吃記憶體
    return args


def run(args, domain: Domain, load_dataset) -> dict:
    """完整流程。load_dataset(args) 由各領域程式提供，回傳 Dataset。"""
    out, data_dir = default_dirs(domain.name)
    args.out = Path(args.out or out)
    args.data_dir = Path(args.data_dir or data_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    setup_ultralytics(args.data_dir.parent / "_downloads")

    ds = load_dataset(args)
    if not ds.items:
        raise SystemExit("沒有讀到任何影像，請確認 --data 路徑")
    if args.task != "auto" and args.task != ds.task:
        if "classify" in (args.task, ds.task):
            raise SystemExit(f"資料是 {ds.task} 格式，無法用 {args.task} 任務訓練")
        print(f"任務改為 {args.task}（資料原為 {ds.task}）")
        ds.task = args.task
    args.imgsz = args.imgsz or (224 if ds.task == "classify" else 640)
    assign_splits(ds, args.val_pct, args.test_pct, args.resplit)
    print_split_summary(ds)

    tile = None
    if should_tile(ds, args.tile, args.imgsz):
        tile = TileCfg(args.tile_size or args.imgsz, args.tile_overlap, args.empty_tile_ratio)
    args.train_imgsz = tile.size if tile else args.imgsz
    data_ref = write_dataset(ds, args.data_dir, tile)
    meta = json.loads((args.data_dir / "eval.json").read_text())
    shutil.copy2(args.data_dir / "split_manifest.csv", args.out / "split_manifest.csv")

    if args.skip_train:
        if not args.model:
            raise SystemExit("--skip-train 需要搭配 --model 指定權重")
        best = Path(args.model)
    else:
        best = train_model(args, data_ref, ds.task, getattr(args, "aug", None) or domain.aug)
        shutil.copy2(best, args.out / "best.pt")
        best = args.out / "best.pt"

    report = evaluate(best, data_ref, meta, args, domain)
    baseline = gate = None
    if args.baseline:
        print(f"\n評估舊模型 {args.baseline} ...")
        baseline = evaluate(args.baseline, data_ref, meta, args, domain)
        gate = compare(report, baseline)
    write_thresholds(args, report)
    report_path = write_report(args, domain, report, baseline, gate)

    unlabeled = args.unlabeled or getattr(args, "default_unlabeled", None)
    if unlabeled:
        build_review_queue(best, unlabeled, meta, report, args)
    exported = export_model(best, args.export, args.train_imgsz, args)

    print("\n" + report_path.read_text(encoding="utf-8"))
    if exported:
        print(f"匯出：{exported}")
    print(f"所有輸出在：{args.out.resolve()}")
    return report
