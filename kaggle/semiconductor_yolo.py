#!/usr/bin/env python3
"""工業 / 半導體 YOLO 檢測（Kaggle）。

支援：
  - 大圖小缺陷：自動切塊訓練 + 切塊推論 + 合併（PCB、晶圓 AOI、SEM 影像）
  - YOLO / Pascal VOC XML（例如 Kaggle「PCB Defects」）/ COCO JSON / 分類資料夾
  - WM-811K 晶圓圖（LSWMD.pkl）：轉成影像做缺陷圖樣分類，依批號（lot）分組切分，
    未標註的晶圓自動拿來做主動學習的複判佇列
  - 指標用產線語言：漏檢率、過殺率，並可與舊模型比較決定能否上線
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi_core as core  # noqa: E402

DOMAIN = core.Domain(
    name="semiconductor",
    escape_label="漏檢率",
    false_alarm_label="過殺率",
    normal_names=("none", "normal", "good", "ok", "pass", "no_defect", "non_defect", "nodefect"),
    # 缺陷方向通常沒有意義，上下左右翻轉都開；灰階 / 單色影像不做色相、飽和度增強；
    # 隨機擦除可能擦掉小缺陷，關閉。
    aug=dict(fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=0.0, translate=0.1, scale=0.3,
             mosaic=1.0, erasing=0.0),
)
# 晶圓圖的灰階值代表晶粒狀態（背景 / 良品 / 不良），任何改變亮度或顏色的增強都會破壞標籤
WAFER_AUG = dict(fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, auto_augment="")


# ───────────────────────── WM-811K 晶圓圖 ─────────────────────────


def read_pickle_compat(path: Path):
    """LSWMD.pkl 是用很舊的 pandas 存的，新版 pandas 讀取時找不到 pandas.indexes，補上別名再讀。"""
    import pandas as pd

    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        print(f"pandas 版本相容處理（{e}）...")
        sys.modules.setdefault("pandas.indexes", importlib.import_module("pandas.core.indexes"))
        for sub in ("base", "range", "multi", "numeric", "category", "frozen"):
            try:
                sys.modules.setdefault(f"pandas.indexes.{sub}", importlib.import_module(f"pandas.core.indexes.{sub}"))
            except ImportError:
                pass
        return pd.read_pickle(path)


def wafer_label(value) -> str | None:
    """failureType 欄位是 [['Center']] 這種巢狀陣列，未標註的是空陣列。"""
    a = np.asarray(value, dtype=object).ravel()
    return str(a[0]).strip() if a.size else None


def wafer_to_image(wafer_map, size: int) -> np.ndarray:
    """0 = 晶圓外、1 = 良品晶粒、2 = 不良晶粒 → 0 / 128 / 255 灰階；用最近鄰放大，保留每顆晶粒的邊界。"""
    m = np.asarray(wafer_map)
    img = np.zeros(m.shape, np.uint8)
    img[m == 1] = 128
    img[m == 2] = 255
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def load_wm811k(args) -> core.Dataset:
    df = read_pickle_compat(Path(args.data))
    print(f"WM-811K：{len(df)} 片晶圓")
    labels = [wafer_label(v) for v in df["failureType"]]
    lots = df["lotName"].astype(str).tolist() if "lotName" in df else [str(i) for i in range(len(df))]
    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab:
            by_class[lab].append(i)
    print("各類別數量：" + "，".join(f"{k}={len(v)}" for k, v in sorted(by_class.items())))
    out = core.cache_dir(args) / f"wafer_{args.wafer_size}"
    items = []
    for lab, idx in sorted(by_class.items()):
        if args.wafer_max_per_class and len(idx) > args.wafer_max_per_class:
            idx = sorted(rng.sample(idx, args.wafer_max_per_class))  # 類別極不平衡（none 佔大多數），設上限
        for i in idx:
            p = out / lab / f"wafer_{i}.png"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), wafer_to_image(df["waferMap"].iat[i], args.wafer_size))
            items.append(core.Item(image=p, group=lots[i], label=lab))  # 同一批號的晶圓高度相關，整批放同一個 split
    unlabeled = [i for i, lab in enumerate(labels) if not lab]
    if args.wafer_unlabeled and unlabeled:
        pool = out / "unlabeled"
        for i in sorted(rng.sample(unlabeled, min(args.wafer_unlabeled, len(unlabeled)))):
            p = pool / f"wafer_{i}.png"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), wafer_to_image(df["waferMap"].iat[i], args.wafer_size))
        args.default_unlabeled = str(pool)  # 沒給 --unlabeled 時，用未標註晶圓產生複判佇列
        print(f"另外取 {min(args.wafer_unlabeled, len(unlabeled))} 片未標註晶圓做複判佇列")
    args.imgsz = args.imgsz or args.wafer_size
    args.aug = WAFER_AUG
    return core.Dataset(items, sorted(by_class), "classify")


# ───────────────────────── 主程式 ─────────────────────────


def load(args) -> core.Dataset:
    if str(args.data).lower().endswith(".pkl"):
        return load_wm811k(args)
    ds = core.load_auto(args.data)
    if not core.apply_group_regex(ds, args.group_regex):
        print("ℹ️ 未指定 --group-regex：每張影像各自一組。同一片晶圓 / 同一批號的多張影像建議用它分組，避免資料洩漏")
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("資料")
    g.add_argument("--data", required=True, help="資料夾（YOLO / VOC / COCO / 分類）、data.yaml，或 WM-811K 的 LSWMD.pkl")
    g.add_argument("--group-regex", default=None, help=r"從檔名取批號 / 晶圓 ID 的正規表示式，例如 '^(LOT\w+)_'")
    g = p.add_argument_group("WM-811K 晶圓圖")
    g.add_argument("--wafer-size", type=int, default=128, help="晶圓圖轉成影像的尺寸")
    g.add_argument("--wafer-max-per-class", type=int, default=3000, help="每類最多取幾片（0 = 不限）")
    g.add_argument("--wafer-unlabeled", type=int, default=2000, help="取幾片未標註晶圓做複判佇列（0 = 不取）")
    core.add_common_args(p, target_recall=0.98, tile_default="auto")
    args = core.parse_args(p, argv)
    return core.run(args, DOMAIN, load)


if __name__ == "__main__":
    main()
