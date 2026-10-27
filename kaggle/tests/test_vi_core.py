"""不需要 GPU / Ultralytics 的單元測試：pytest kaggle/tests"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import medical_yolo  # noqa: E402
import semiconductor_yolo  # noqa: E402
import vi_core as core  # noqa: E402


def _items(n, prefix="g"):
    return [core.Item(image=Path(f"/x/{prefix}{i}.png"), group=f"{prefix}{i // 2}", boxes=[(0, 0, 0, 5, 5)])
            for i in range(n)]


def test_hash_split_keeps_groups_together_and_is_stable():
    ds = core.Dataset(_items(400), ["a"], "detect")
    core.assign_splits(ds, 0.15, 0.15)
    by_group = {}
    for it in ds.items:
        assert by_group.setdefault(it.group, it.split) == it.split
    before = {str(it.image): it.split for it in ds.items}
    grown = core.Dataset(_items(400) + _items(100, prefix="new"), ["a"], "detect")
    core.assign_splits(grown, 0.15, 0.15)
    assert all(before[str(it.image)] == it.split for it in grown.items if str(it.image) in before)
    counts = {s: sum(it.split == s for it in ds.items) for s in core.SPLITS}
    assert 0.08 < counts["test"] / 400 < 0.22


def test_small_dataset_gets_negatives_into_val_and_test():
    items = _items(12) + [core.Item(image=Path(f"/x/ok{i}.png"), group=f"ok{i}") for i in range(4)]
    ds = core.Dataset(items, ["a"], "detect")
    core.assign_splits(ds, 0.2, 0.2)
    for split in ("val", "test"):
        assert any(it.split == split and not it.boxes for it in ds.items)


def test_source_splits_are_kept_unless_resplit():
    items = [core.Item(image=Path(f"/x/{s}{i}.png"), group=f"{s}{i}", label="a", split=s)
             for s in core.SPLITS for i in range(3)]
    ds = core.Dataset(items, ["a"], "classify")
    core.assign_splits(ds, 0.15, 0.15)
    assert [it.split for it in ds.items] == [s for s in core.SPLITS for _ in range(3)]


def test_tile_windows_cover_image():
    wins = core.tile_windows(1000, 700, 320, 0.2)
    cover = np.zeros((700, 1000), bool)
    for x0, y0, x1, y1 in wins:
        assert x1 - x0 <= 320 and y1 - y0 <= 320
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    assert core.tile_windows(200, 100, 320, 0.2) == [(0, 0, 200, 100)]


def test_clip_boxes_drops_mostly_hidden_boxes():
    kept, dropped = core.clip_boxes([(0, 90, 10, 110, 30), (1, 95, 50, 200, 60)], (0, 0, 100, 100), 0.4)
    assert kept == [(0, 90, 10, 100, 30)]
    assert dropped == 1


def test_nms_ios_merges_fragments_but_keeps_other_classes():
    preds = np.array([
        [0, 0.9, 0, 0, 100, 100],
        [0, 0.6, 10, 10, 40, 40],  # 被大框包住的碎片
        [1, 0.7, 10, 10, 40, 40],  # 不同類別要保留
        [0, 0.5, 300, 300, 320, 320],
    ])
    out = core.nms_ios(preds)
    assert len(out) == 3
    assert not any((r[0] == 0 and r[1] == 0.6) for r in out)


def test_threshold_for_recall():
    t, ok = core.threshold_for_recall([0.9, 0.8, 0.7, 0.2], 0.75)
    assert ok and t == 0.7
    t, ok = core.threshold_for_recall([0.9, 0.0], 1.0)
    assert not ok and t == core.CONF_FLOOR


def test_metrics_and_tuning():
    gts = [np.array([[0, 10, 10, 50, 50]]), np.zeros((0, 5)), np.array([[0, 100, 100, 150, 150]])]
    preds = [
        np.array([[0, 0.8, 12, 12, 50, 50]]),
        np.array([[0, 0.3, 0, 0, 20, 20]]),  # 正常影像上的誤報
        np.array([[0, 0.4, 100, 100, 150, 150], [0, 0.35, 0, 0, 5, 5]]),
    ]
    th = core.tune_thresholds(preds, gts, ["d"], 1.0, 0.3, "object", 0.5)
    assert th["d"]["threshold"] == 0.4 and th["d"]["val_recall"] == 1.0
    thr = np.array([0.4])
    obj = core.object_metrics(preds, gts, thr, 0.3, ["d"])
    assert obj["recall"] == 1.0 and obj["fp_per_image"] == 0
    img = core.image_metrics(preds, gts, thr)
    assert img["sensitivity"] == 1.0 and img["specificity"] == 1.0
    img = core.image_metrics(preds, gts, np.array([0.3]))
    assert img["overkill_rate"] == 1.0
    capped = core.tune_thresholds(preds, gts, ["d"], 0.5, 0.3, "object", 0.5)
    assert capped["d"]["threshold"] == 0.5


def test_roc_auc_with_ties():
    assert core.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert core.roc_auc([1, 0], [0.5, 0.5]) == 0.5
    assert core.roc_auc([1, 1], [0.5, 0.6]) is None


def test_pick_normal_and_compare():
    assert core.pick_normal(["Center", "none"], None, semiconductor_yolo.DOMAIN.normal_names) == "none"
    assert core.pick_normal(["No Finding", "Mass"], None, medical_yolo.DOMAIN.normal_names) == "No Finding"
    new = {"key": {"escape": 0.01, "false_alarm": 0.05, "quality": 0.9}}
    old = {"key": {"escape": 0.02, "false_alarm": 0.05, "quality": 0.88}}
    assert core.compare(new, old)[0]
    assert not core.compare(old | {"key": {**old["key"], "escape": 0.05}}, new)[0]


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def test_voc_roundtrip_to_yolo(tmp_path):
    _img(tmp_path / "src/images/a.jpg")
    _img(tmp_path / "src/images/ok.jpg")
    (tmp_path / "src/ann").mkdir()
    (tmp_path / "src/ann/a.xml").write_text(
        "<annotation><filename>a.jpg</filename><object><name>spur</name>"
        "<bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>")
    assert core.detect_format(tmp_path / "src") == "voc"
    ds = core.load_auto(tmp_path / "src")
    assert ds.names == ["spur"] and len(ds.items) == 2
    for it in ds.items:
        it.split = "train" if it.boxes else "test"
    core.write_dataset(ds, tmp_path / "out")
    label = next((tmp_path / "out/labels/train").glob("*.txt")).read_text().split()
    assert label == ["0", "0.200000", "0.300000", "0.200000", "0.400000"]
    assert next((tmp_path / "out/labels/test").glob("*.txt")).read_text() == ""
    meta = json.loads((tmp_path / "out/eval.json").read_text())
    assert meta["test"][0]["boxes"] == [] and meta["train"][0]["boxes"] == [[0, 20, 10, 60, 50]]
    with pytest.raises(SystemExit):  # 不是本程式建立的資料夾不能被清掉
        core.write_dataset(ds, tmp_path / "src")


def test_tiled_dataset_writes_tiles_with_local_boxes(tmp_path):
    _img(tmp_path / "big.png", 600, 300)
    it = core.Item(image=tmp_path / "big.png", group="b", boxes=[(0, 500, 200, 520, 220)], split="val")
    ds = core.Dataset([it], ["d"], "detect")
    core.write_dataset(ds, tmp_path / "out", core.TileCfg(size=256, overlap=0.25))
    labels = {p.stem: p.read_text() for p in (tmp_path / "out/labels/val").glob("*.txt")}
    hits = sorted(k.split("__")[1] for k, v in labels.items() if v)
    assert hits == ["344_0", "344_44"]  # 貼齊右緣的那一欄，上下兩個重疊切塊都含這個缺陷
    local = next(v for k, v in labels.items() if k.endswith("__344_0")).split()
    assert local == ["0", "0.648438", "0.820312", "0.078125", "0.078125"]
    meta = json.loads((tmp_path / "out/eval.json").read_text())
    assert meta["tile"]["size"] == 256 and meta["val"][0]["boxes"] == [[0, 500, 200, 520, 220]]


def test_coco_with_polygons_is_segment(tmp_path):
    _img(tmp_path / "train/img1.jpg")
    coco = {"images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 3, "name": "crack"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 30, 20],
                             "segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]]}]}
    (tmp_path / "train/_annotations.coco.json").write_text(json.dumps(coco))
    ds = core.load_auto(tmp_path)
    assert ds.task == "segment" and ds.names == ["crack"]
    assert ds.items[0].split == "train" and ds.items[0].boxes == [(0, 10, 10, 40, 30)]


def test_mask_to_polygons(tmp_path):
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 40), 20, 255, -1)
    cv2.imwrite(str(tmp_path / "m.png"), m)
    polys = core.mask_to_polygons(tmp_path / "m.png", 1, (200, 200))  # 影像是遮罩的兩倍大
    assert len(polys) == 1
    xs, ys = polys[0][1][0::2], polys[0][1][1::2]
    assert abs(min(xs) - 60) <= 2 and abs(max(xs) - 140) <= 2 and abs(min(ys) - 40) <= 2


def test_medical_csv_loader_and_merge(tmp_path):
    _img(tmp_path / "imgs/a.png")
    _img(tmp_path / "imgs/b.png")
    (tmp_path / "train.csv").write_text(
        "image_id,class_name,x_min,y_min,x_max,y_max,patient\n"
        "a,Mass,10,10,30,30,p1\n"
        "a,Mass,12,11,31,29,p1\n"
        "b,No finding,,,,,p2\n"
        "missing,Mass,1,1,2,2,p3\n")
    args = argparse.Namespace(csv=str(tmp_path / "train.csv"), data=None, images=str(tmp_path / "imgs"),
                              image_col="image_id", label_col="class_name", class_name="lesion", box_cols=None,
                              box_format="xyxy", negative_label=["No finding"], patient_col="patient", merge_iou=0.4)
    ds = medical_yolo.load_csv(args)
    by_name = {it.image.name: it for it in ds.items}
    assert ds.names == ["Mass"] and set(by_name) == {"a.png", "b.png"}
    assert len(by_name["a.png"].boxes) == 1 and by_name["a.png"].group == "p1"
    assert np.allclose(by_name["a.png"].boxes[0][1:], (11, 10.5, 30.5, 29.5))
    assert by_name["b.png"].boxes == []


def test_wafer_helpers_and_loader(tmp_path):
    pd = pytest.importorskip("pandas")
    assert semiconductor_yolo.wafer_label(np.array([["Center"]], dtype=object)) == "Center"
    assert semiconductor_yolo.wafer_label(np.zeros((0, 0))) is None
    img = semiconductor_yolo.wafer_to_image(np.array([[0, 1], [2, 1]]), 4)
    assert img.shape == (4, 4) and set(np.unique(img)) == {0, 128, 255}
    rows = [{"waferMap": np.ones((5, 5), np.uint8), "lotName": f"lot{i // 2}",
             "failureType": np.array([["none" if i % 2 else "Loc"]], dtype=object) if i < 8 else np.zeros((0, 0))}
            for i in range(10)]
    pd.DataFrame(rows).to_pickle(tmp_path / "w.pkl")
    args = argparse.Namespace(data=str(tmp_path / "w.pkl"), seed=0, wafer_max_per_class=3, wafer_size=32,
                              wafer_unlabeled=5, imgsz=None, data_dir=tmp_path / "ds")
    ds = semiconductor_yolo.load_wm811k(args)
    assert ds.task == "classify" and ds.names == ["Loc", "none"] and len(ds.items) == 6
    assert args.imgsz == 32 and len(list(Path(args.default_unlabeled).glob("*.png"))) == 2


def test_small_sample_warning():
    from vi_core import _small, MIN_RELIABLE_N
    assert _small(0) == ""
    assert "僅供參考" in _small(2)
    assert _small(MIN_RELIABLE_N) == ""
