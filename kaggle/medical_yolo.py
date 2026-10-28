#!/usr/bin/env python3
"""醫學影像 YOLO 檢測（Kaggle）。

支援：
  - DICOM（自動套用 Modality/VOI LUT 或指定窗位窗寬、MONOCHROME1 反相、可選 CLAHE）
  - CSV 框標註（VinDr-CXR、RSNA Pneumonia 這類「一列一個框」格式，自動辨識欄位）
  - 分割遮罩（Kvasir-SEG、ISIC 等，自動轉成 YOLO 多邊形）
  - YOLO / VOC / COCO / 分類資料夾（例如 Chest X-Ray Pneumonia）
  - 以「病人」分組切 train/val/test，避免同一病人同時出現在訓練與測試

不給 --data 時用 Ultralytics 內建的 brain-tumor 資料集示範。

⚠️ 僅供研究與教學，不可作為臨床診斷依據。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi_core as core  # noqa: E402

DICOM_EXTS = {".dcm", ".dicom"}
EXTS = core.IMG_EXTS | DICOM_EXTS
BOX_COLUMNS = [  # 自動辨識的框欄位：(欄位, 格式)
    (("x_min", "y_min", "x_max", "y_max"), "xyxy"),  # VinDr-CXR
    (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
    (("x1", "y1", "x2", "y2"), "xyxy"),
    (("x", "y", "width", "height"), "xywh"),  # RSNA Pneumonia
    (("x", "y", "w", "h"), "xywh"),
]
EMPTY = {"", "nan", "none", "null"}

DOMAIN = core.Domain(
    name="medical",
    escape_label="漏診率（1 − 敏感度）",
    false_alarm_label="偽陽性率（1 − 特異度）",
    normal_names=("normal", "no_finding", "negative", "healthy", "nofinding"),
    # 解剖構造有左右之分（例如心臟在左），預設不翻轉；灰階影像不做色相 / 飽和度增強；
    # 隨機擦除可能把小病灶整個擦掉、造成錯誤標籤，所以關閉。
    aug=dict(fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=5.0, translate=0.05, scale=0.2,
             mosaic=0.5, erasing=0.0),
    disclaimer="⚠️ 本結果僅供研究與教學，不可作為臨床診斷依據。",
)


# ───────────────────────── 影像讀取與前處理 ─────────────────────────


def _lut_funcs():
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut  # pydicom >= 3
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    return apply_modality_lut, apply_voi_lut


def _scale_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def read_dicom(path, window: tuple[float, float] | None = None) -> np.ndarray:
    """DICOM → 8-bit 影像。灰階回傳 (H, W)，彩色（內視鏡、超音波）回傳 BGR。"""
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if int(getattr(ds, "SamplesPerPixel", 1)) == 3:
        if arr.ndim == 4:  # 多張彩色影格：取中間那張
            arr = arr[arr.shape[0] // 2]
        if arr.dtype != np.uint8:
            arr = _scale_to_uint8(arr.astype(np.float32), *np.percentile(arr, (0.5, 99.5)))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.ndim == 3:  # 多切片（CT/MR 多影格）：取中間那張
        arr = arr[arr.shape[0] // 2]
    apply_modality_lut, apply_voi_lut = _lut_funcs()
    arr = apply_modality_lut(arr, ds).astype(np.float32)  # 例如 CT 轉成 HU
    if window:
        c, w = window
        img = _scale_to_uint8(arr, c - w / 2, c + w / 2)
    else:
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float32)  # 用 DICOM 內建的窗位窗寬 / VOI LUT
        except Exception:
            pass
        img = _scale_to_uint8(arr, *np.percentile(arr, (0.5, 99.5)))
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = 255 - img  # MONOCHROME1 是「數值越大越暗」，反相成一般的顯示方式
    return img


def read_image_8bit(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("讀不到影像")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:  # 16-bit PNG/TIFF
        f = img.astype(np.float32)
        img = _scale_to_uint8(f, *np.percentile(f, (0.5, 99.5)))
    return img


def apply_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _convert_one(job):
    """（子行程）轉一張影像；回傳 (來源, (sx, sy) 縮放比例, 錯誤訊息)。"""
    src, dst, clahe, max_size, window = job
    try:
        img = read_dicom(src, window) if Path(src).suffix.lower() in DICOM_EXTS else read_image_8bit(src)
        h, w = img.shape[:2]
        if clahe:
            img = apply_clahe(img)
        if max_size and max(h, w) > max_size:
            r = max_size / max(h, w)
            img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img)
        return src, (img.shape[1] / w, img.shape[0] / h), None
    except Exception as e:
        return src, None, f"{type(e).__name__}: {e}"


def _check_dicom_decoder(sample: Path) -> None:
    """壓縮的 DICOM（JPEG2000 / JPEG-LS）需要額外解碼器；第一次失敗就自動安裝再試一次。"""
    core.ensure_package("pydicom")
    try:
        read_dicom(sample)
        return
    except Exception as e:
        print(f"DICOM 解碼失敗（{e}），安裝 pylibjpeg 解碼器後重試 ...")
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"])
    read_dicom(sample)


def preprocess(ds: core.Dataset, args) -> None:
    """DICOM 一律轉 PNG；其他影像在指定 --clahe / --max-size 時才轉。標註座標同步縮放。轉檔結果會快取。"""
    window = tuple(args.window) if args.window else None
    needs = lambda p: p.suffix.lower() in DICOM_EXTS or args.clahe or args.max_size  # noqa: E731
    todo = [it for it in ds.items if needs(it.image)]
    if not todo:
        return
    dicoms = [it.image for it in todo if it.image.suffix.lower() in DICOM_EXTS]
    if dicoms:
        _check_dicom_decoder(dicoms[0])
    out_dir = core.cache_dir(args) / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    keyed = []
    for it in todo:
        key = f"{it.image}|clahe={args.clahe}|max={args.max_size}|win={window}"
        keyed.append((it, key, out_dir / f"{core.unique_stem(it.image)}.png"))
    jobs = [(str(it.image), str(dst), args.clahe, args.max_size, window)
            for it, key, dst in keyed if key not in index or not dst.exists()]
    if jobs:
        print(f"轉換 {len(jobs)} 張影像（{len(dicoms)} 張 DICOM）→ {out_dir}")
        errors = []
        with ProcessPoolExecutor(max_workers=args.convert_workers) as pool:
            done = dict()
            for n, (src, scale, err) in enumerate(pool.map(_convert_one, jobs, chunksize=8), 1):
                if err:
                    errors.append((src, err))
                done[src] = scale
                if n % 500 == 0:
                    print(f"  {n}/{len(jobs)}")
        for it, key, _ in keyed:
            if str(it.image) in done and done[str(it.image)] is not None:
                index[key] = done[str(it.image)]
        index_path.write_text(json.dumps(index))
        if errors:
            print(f"⚠️ {len(errors)} 張轉換失敗，已排除。例如：{errors[0][0]}：{errors[0][1]}")
    failed = set()
    for it, key, dst in keyed:
        if key not in index:
            failed.add(id(it))
            continue
        sx, sy = index[key]
        it.image = dst
        it.boxes = [(c, x1 * sx, y1 * sy, x2 * sx, y2 * sy) for c, x1, y1, x2, y2 in it.boxes]
        it.polygons = [(c, [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(pts)]) for c, pts in it.polygons]
    ds.items = [it for it in ds.items if id(it) not in failed]


# ───────────────────────── 資料讀取 ─────────────────────────


def _box_spec(header: list[str], args) -> tuple[list[str], str]:
    if args.box_cols:
        cols = [c.strip() for c in args.box_cols.split(",")]
        missing = [c for c in cols if c not in header]
        if missing:
            raise SystemExit(f"CSV 沒有欄位 {missing}；現有欄位：{header}")
        return cols, args.box_format
    for cols, fmt in BOX_COLUMNS:
        if all(c in header for c in cols):
            print(f"框欄位：{', '.join(cols)}（{fmt}）")
            return list(cols), fmt
    raise SystemExit(f"找不到框座標欄位，請用 --box-cols 指定；現有欄位：{header}")


def merge_duplicate_boxes(boxes: list, iou: float) -> list:
    """多位醫師對同一病灶各標一次（例如 VinDr-CXR 有 3 位判讀者）時，把同類別且重疊的框平均成一個。"""
    out = []
    for c in sorted({b[0] for b in boxes}, key=str):
        rest = np.array([b[1:] for b in boxes if b[0] == c], float)
        while len(rest):
            ious = core.iou_matrix(rest[:1], rest)[0]
            group = ious >= iou
            out.append((c, *rest[group].mean(0)))
            rest = rest[~group]
    return out


def load_csv(args) -> core.Dataset:
    """一列一個框的 CSV；框欄位空白、或類別為 --negative-label 的列代表「正常影像」。"""
    csv_path = Path(args.csv or args.data)
    if args.images:
        img_root = Path(args.images)
    elif args.csv and args.data and Path(args.data).is_dir():
        img_root = Path(args.data)
    else:
        img_root = csv_path.parent
    index = core.ImageIndex(img_root, EXTS)
    print(f"在 {img_root} 找到 {len(index.all)} 張影像")
    negatives = {n.strip().lower() for n in args.negative_label}
    items, missing = {}, set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        cols, fmt = _box_spec(header, args)
        label_col = args.label_col if args.label_col in header else None
        if label_col is None:
            print(f"CSV 沒有 {args.label_col!r} 欄位，所有框視為單一類別 {args.class_name!r}")
        patient_col = args.patient_col if args.patient_col in header else None
        if args.patient_col and patient_col is None:
            print(f"⚠️ CSV 沒有病人欄位 {args.patient_col!r}，改用影像 ID 分組")
        for row in reader:
            iid = row[args.image_col].strip()
            it = items.get(iid)
            if it is None:
                path = index.find(iid)
                if path is None:
                    missing.add(iid)
                    continue
                group = row[patient_col].strip() if patient_col else iid
                it = items[iid] = core.Item(image=path, group=group)
            label = row[label_col].strip() if label_col else args.class_name
            vals = [row.get(c, "").strip() for c in cols]
            if label.lower() in negatives or any(v.lower() in EMPTY for v in vals):
                continue
            x1, y1, a, b = map(float, vals)
            x2, y2 = (x1 + a, y1 + b) if fmt == "xywh" else (a, b)
            it.boxes.append((label, x1, y1, x2, y2))
    if missing:
        print(f"⚠️ {len(missing)} 個影像 ID 找不到檔案，例如：{sorted(missing)[:3]}")
    items = list(items.values())
    if args.merge_iou:
        for it in items:
            it.boxes = merge_duplicate_boxes(it.boxes, args.merge_iou)
    n_neg = sum(1 for it in items if not it.boxes)
    print(f"CSV：{len(items)} 張影像，其中 {n_neg} 張沒有病灶（正常）")
    return core.finalize_named(items, "detect")


def load_masks(args) -> core.Dataset:
    """影像資料夾 + 遮罩資料夾（檔名相同；ISIC 這類遮罩檔名有後綴時用 --mask-suffix）。"""
    img_root, mask_root = Path(args.data), Path(args.masks)
    masks = {}
    for m in core.iter_images(mask_root):
        stem = m.stem[: -len(args.mask_suffix)] if args.mask_suffix and m.stem.endswith(args.mask_suffix) else m.stem
        masks[stem] = m
    names = [n.strip() for n in args.mask_classes.split(",")]
    items, missing = [], 0
    mask_root_r = mask_root.resolve()
    for img in core.iter_images(img_root, EXTS):
        if mask_root_r in img.resolve().parents:  # 遮罩資料夾在影像資料夾裡面時跳過遮罩本身
            continue
        m = masks.get(img.stem)
        if m is None:
            missing += 1
            continue
        # DICOM 讀不到尺寸，用遮罩尺寸（兩者通常相同；轉 PNG 時座標會再同步縮放）
        size = core.image_size(m if img.suffix.lower() in DICOM_EXTS else img)
        polys = core.mask_to_polygons(m, len(names), size)
        items.append(core.Item(image=img, group=img.stem, polygons=polys, split=core.split_from_path(img, img_root)))
    if missing:
        print(f"⚠️ {missing} 張影像找不到對應遮罩，已略過")
    return core.Dataset(items, names, "segment")


def load(args) -> core.Dataset:
    if args.data is None and args.csv is None:
        print("未指定 --data：使用 Ultralytics 內建 brain-tumor 資料集示範（需開啟 Kaggle Internet）")
        args.data = "brain-tumor.yaml"
    if args.csv or str(args.data).lower().endswith(".csv"):
        ds = load_csv(args)
    elif args.masks:
        ds = load_masks(args)
    else:
        ds = core.load_auto(args.data, EXTS)
    grouped = core.apply_group_regex(ds, args.group_regex) or bool(args.patient_col)
    if not grouped and args.data != "brain-tumor.yaml":
        print("⚠️ 未指定病人分組（--patient-col 或 --group-regex）：同一病人的多張影像可能同時出現在訓練與測試，"
              "指標會偏樂觀")
    preprocess(ds, args)
    if args.allow_flip:
        args.aug = {**DOMAIN.aug, "fliplr": 0.5}
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("資料")
    g.add_argument("--data", default=None, help="資料路徑：資料夾 / data.yaml / CSV；不給則用 brain-tumor 示範")
    g.add_argument("--csv", default=None, help="CSV 框標註檔（一列一個框）")
    g.add_argument("--images", default=None, help="CSV 對應的影像資料夾（預設為 CSV 所在資料夾）")
    g.add_argument("--image-col", default="image_id")
    g.add_argument("--label-col", default="class_name", help="類別欄位；不存在時所有框視為 --class-name")
    g.add_argument("--class-name", default="lesion")
    g.add_argument("--box-cols", default=None, help="框欄位，例如 x,y,width,height；預設自動辨識")
    g.add_argument("--box-format", choices=["xyxy", "xywh"], default="xyxy")
    g.add_argument("--negative-label", nargs="*", default=["No finding"], help="代表「正常、沒有病灶」的類別名稱")
    g.add_argument("--patient-col", default=None, help="CSV 中的病人 ID 欄位（用來分組，避免資料洩漏）")
    g.add_argument("--merge-iou", type=float, default=0.0, help="合併多位判讀者重複標註的框（例如 0.4）；0 = 不合併")
    g.add_argument("--masks", default=None, help="分割遮罩資料夾（與影像同檔名）")
    g.add_argument("--mask-suffix", default="", help="遮罩檔名的後綴，例如 ISIC 的 _segmentation")
    g.add_argument("--mask-classes", default="lesion", help="遮罩類別名稱，逗號分隔（像素值 1、2… 依序對應）")
    g.add_argument("--group-regex", default=None, help=r"從檔名取病人 ID 的正規表示式，例如 '^(person\d+)_'")
    g = p.add_argument_group("醫學影像前處理")
    g.add_argument("--window", nargs=2, type=float, metavar=("CENTER", "WIDTH"), default=None,
                   help="DICOM 窗位 / 窗寬，例如肺窗 -600 1500；預設用 DICOM 內建值")
    g.add_argument("--clahe", action="store_true", help="CLAHE 局部對比強化（X 光常用）")
    g.add_argument("--max-size", type=int, default=None, help="轉檔時把長邊縮到這個尺寸（例如 1024），節省空間與時間")
    g.add_argument("--allow-flip", action="store_true", help="允許左右翻轉增強（左右對稱的部位才開）")
    g.add_argument("--convert-workers", type=int, default=4)
    core.add_common_args(p, target_recall=0.95, tile_default="off")
    args = core.parse_args(p, argv)
    return core.run(args, DOMAIN, load)


if __name__ == "__main__":
    main()
