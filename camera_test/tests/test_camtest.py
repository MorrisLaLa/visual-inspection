"""不需要模型 / 相機的單元測試：pytest camera_test/tests"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camtest_core as core  # noqa: E402

W, H, FPS = 400, 120, 30


def make_cfg(**over):
    raw = {
        "line": {"part_length_mm": 60, "part_gap_mm": 80},
        "camera": {"type": "iphone", "resolution": [W, H], "fps": FPS, "shutter": "1/1000", "mm_per_px": 1.0},
        "inspect": {"model": None, "thresholds": None, "background_frames": 20, "vote_min_frames": 2,
                    "min_valid_frames": 3},
    }
    for dotted, v in over.items():
        sec, _, key = dotted.partition(".")
        raw.setdefault(sec, {})[key] = v
    return core.build_config(raw)


def conveyor(n_parts, px_per_frame, ng=(), part_len=60, gap=80, empty=25, defect_frames=None):
    """素色輸送帶上一排亮色工件由左往右移動；ng 中的工件中央有黑點。"""
    pitch = part_len + gap
    frames = [np.full((H, W, 3), 90, np.uint8) for _ in range(empty)]
    total = int((W + n_parts * pitch) / px_per_frame) + 3
    for f in range(total):
        img = np.full((H, W, 3), 90, np.uint8)
        for i in range(n_parts):
            x = int(f * px_per_frame) - (i + 1) * pitch + gap  # 第 i 件的前緣位置
            x0, x1 = max(x - part_len, 0), min(x, W)
            if x1 <= x0:
                continue
            img[20:100, x0:x1] = 190
            cx = x - part_len // 2
            seen = defect_frames is None or f in defect_frames
            if i + 1 in ng and 0 <= cx < W and seen:
                cv2.circle(img, (cx, 60), 6, (20, 20, 20), -1)
        frames.append(img)
    return frames


def fake_detector(roi):
    """把黑點當缺陷（取代 YOLO）。"""
    mask = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) < 60).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    dets = [core.Detection("spot", 0.9, (x, y, x + w, y + h)) for x, y, w, h, a in stats[1:] if a > 5]
    return dets, []


def run(cfg, frames):
    insp = core.Inspector(cfg, fake_detector, {"spot": 0.5})
    parts = []
    for i, f in enumerate(frames):
        parts += insp.feed(f, i / FPS)
    return sorted(parts + insp.finish(), key=lambda p: p.t_first), insp


def test_counts_parts_and_votes_ng():
    parts, insp = run(make_cfg(), conveyor(5, 8, ng={2, 4}))
    assert [p.verdict for p in parts] == ["OK", "NG", "OK", "NG", "OK"]
    assert all(abs(p.speed_mm_s - 8 * FPS) < 10 for p in parts)  # 從影片量出速度 ≈ 240 mm/s
    assert insp.stray_detections == 0


def test_too_fast_is_never_ok():
    """每件只拍到 1~2 幀時，必須判「需人工確認」，不能放行。"""
    cfg = make_cfg(**{"camera.motion_direction": "+", "line.part_gap_mm": 300})
    parts, _ = run(cfg, conveyor(4, 180, ng={2}, gap=300))  # (400 - 60) / 180 → 每件最多完整 2 幀
    assert len(parts) == 4 and all(p.verdict == "CHECK" for p in parts)
    assert all("幀" in p.reason for p in parts)
    assert all(p.speed_mm_s is None for p in parts)  # 幀數不夠時不回報（會錯的）速度


def test_stroboscopic_speed_is_flagged():
    """每幀剛好走一個間距時，工件在影片上看起來靜止：必須判需人工確認。"""
    parts, _ = run(make_cfg(), conveyor(4, 140, ng={2}))
    assert parts and all(p.verdict == "CHECK" and "沒在動" in p.reason for p in parts)


def test_direction_setting_and_aliasing_plan():
    with pytest.raises(core.ConfigError, match="motion_direction"):
        make_cfg(**{"camera.motion_direction": "sideways"})
    cfg = make_cfg(**{"camera.motion_direction": "left"})
    assert cfg.direction == -1
    row = core.plan_row(cfg, 140 * FPS)  # 每幀走 140 mm = 一個間距
    assert row.verdict == "不行" and any("分不出" in r for r in row.reasons)


def test_single_frame_defect_needs_human():
    frames = conveyor(1, 8, ng={1}, defect_frames={30})
    parts, _ = run(make_cfg(), frames)
    assert [p.verdict for p in parts] == ["CHECK"]
    assert "只有 1 幀" in parts[0].reason


def test_merged_parts_flagged():
    parts, _ = run(make_cfg(), conveyor(2, 8, gap=1))
    assert len(parts) == 1 and parts[0].verdict == "CHECK" and "黏在一起" in parts[0].reason


def test_part_in_view_during_background_is_rejected():
    frames = conveyor(3, 8, empty=0)
    insp = core.Inspector(make_cfg(), fake_detector, {"spot": 0.5})
    with pytest.raises(ValueError, match="背景不穩定"):
        for i, f in enumerate(frames):
            insp.feed(f, i / FPS)


def test_config_validation_lists_all_errors():
    with pytest.raises(core.ConfigError) as e:
        core.build_config({
            "line": {"part_length_mm": -1},
            "camera": {"type": "phone", "resolution": [1920], "fps": 60, "shutter": "abc"},
            "inspect": {"vote_min_frames": 3, "min_valid_frames": 2},
        })
    msg = str(e.value)
    for key in ("part_length_mm", "camera.type", "camera.resolution", "camera.shutter", "min_valid_frames"):
        assert key in msg


def test_unit_mistakes_become_warnings():
    cfg = make_cfg(**{"line.speed_mm_s": 0.5, "camera.shutter": 2000})
    assert cfg.shutter_s == pytest.approx(1 / 2000)
    text = " ".join(cfg.warnings)
    assert "m/s" in text and "1/2000" in text


def test_shutter_longer_than_frame_is_error():
    with pytest.raises(core.ConfigError, match="比每幀時間"):
        make_cfg(**{"camera.shutter": "1/15"})


def test_roi_shorter_than_part_is_error():
    with pytest.raises(core.ConfigError, match="永遠無法完整入鏡"):
        make_cfg(**{"inspect.roi": [0, 0, 50, 120]})


def test_plan_math():
    cfg = make_cfg(**{"camera.shutter": "1/1000"})
    r = core.plan_row(cfg, 500)  # 500 mm/s、1 mm/px、1/1000 秒 → 模糊 0.5 px
    assert r.blur_px == pytest.approx(0.5)
    assert r.frames_full == pytest.approx((400 - 60) / 500 * FPS)
    assert r.parts_per_min == pytest.approx(60 * 500 / 140)
    v, limit = core.max_speed(cfg)
    assert v == pytest.approx(min(1.0 * 1.0 / 0.001, (400 - 60) * FPS / 3))
    assert "模糊" in core.plan_row(cfg, 5000).reasons[0]
    report = core.plan_report(cfg)
    assert "最高可用速度" in report and "| 1000 |" in report


def test_manifest_validation(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "m.csv").write_text("video,speed_mm_s,n_parts,ng_parts\na.mp4,100,5,\"2,9\"\nmissing.mp4,x,0,\n")
    with pytest.raises(core.ConfigError) as e:
        core.load_manifest(tmp_path / "m.csv")
    msg = str(e.value)
    assert "超出 1~5" in msg and "找不到影片" in msg and "n_parts" in msg


def test_score_and_speed_limit(tmp_path):
    row = core.ManifestRow(tmp_path / "a.mp4", 100, 4, {2})

    def part(i, verdict):
        return core.PartResult(i, i, i + 1, 5, 4, 0, 60, 100, {}, verdict, "")

    good = core.score_run([part(1, "OK"), part(2, "NG"), part(3, "OK"), part(4, "CHECK")], row)
    assert good["valid"] and good["escape_rate"] == 0 and good["overkill_rate"] == pytest.approx(1 / 3)
    bad = core.score_run([part(1, "OK"), part(2, "OK"), part(3, "OK"), part(4, "OK")], row)
    assert bad["escape_rate"] == 1 and bad["escaped_positions"] == [2]
    short = core.score_run([part(1, "OK")], row)
    assert not short["valid"] and "計數不符" in short["problem"]
    cfg = make_cfg(**{"criteria.max_overkill_rate": 0.5})
    results = [dict(speed=100, valid=True, escape_rate=0, overkill_rate=0),
               dict(speed=200, valid=True, escape_rate=0, overkill_rate=0.4),
               dict(speed=300, valid=True, escape_rate=0.5, overkill_rate=0),
               dict(speed=400, valid=True, escape_rate=0, overkill_rate=0)]
    assert core.speed_limit(results, cfg) == 200  # 300 失敗之後，400 合格也不採信


def test_load_thresholds_checks(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"normal_class": "none", "abnormal_threshold": 0.3}))
    with pytest.raises(core.ConfigError, match="分類模型"):
        core.load_thresholds(str(p), ["a"], None, "part", "background")
    p.write_text(json.dumps({"scratch": 0.3, "crack": 0.2}))
    with pytest.raises(core.ConfigError, match="不在模型裡"):
        core.load_thresholds(str(p), ["scratch", "dent"], None, "part", "background")
    p.write_text(json.dumps({"scratch": 0.3}))
    thr, warns = core.load_thresholds(str(p), ["scratch", "dent", "part"], None, "part", "model")
    assert thr == {"scratch": 0.3, "dent": 0.25} and warns
    with pytest.raises(core.ConfigError, match="presence: model"):
        core.load_thresholds(None, ["scratch"], None, "part", "model")


def test_example_config_loads_and_plans(tmp_path):
    """隨附的範例設定檔一定要能直接用。"""
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    text = example.read_text(encoding="utf-8").replace("dir: camtest_runs", f"dir: {tmp_path}")
    (tmp_path / "c.yaml").write_text(text, encoding="utf-8")
    cfg = core.load_config(tmp_path / "c.yaml")
    assert cfg.readout_ms == core.CAMERA_PRESETS["iphone"] and cfg.direction is None
    assert "最高可用速度" in core.plan_report(cfg)
