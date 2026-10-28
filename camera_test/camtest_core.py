"""相機實測核心邏輯（不依賴 Ultralytics，可單獨測試）。

- 設定檔讀取與防呆檢查（單位、範圍、互相矛盾的設定）
- 規劃計算：動態模糊、每件拍到幾幀、滾動快門歪斜、最高可用速度
- 工件偵測（背景相減或模型）→ 沿運動方向追蹤 → 多幀投票判 OK / NG / 需人工確認
- 速度測試：與實際答案（manifest）比對，算漏檢率 / 過殺率
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

CAMERA_PRESETS = {"iphone": 20.0, "cctv": 30.0, "industrial": 0.0}  # 預設滾動快門讀出時間（ms）；工業全域快門 = 0
YOLO_MIN_DEFECT_PX = 10  # 缺陷在模型輸入影像上至少約 10 px，YOLO 才抓得穩（經驗值）
MIN_DEFECT_PX = 3  # 缺陷在原始影像上至少 3 px 才看得到
DEFAULT_SPEEDS = (20, 50, 100, 200, 300, 500, 800, 1000)  # 速度未知時，規劃表列出的速度（mm/s）
VERDICTS = ("OK", "NG", "CHECK")
VERDICT_ZH = {"OK": "良品", "NG": "不良", "CHECK": "需人工確認"}


class ConfigError(Exception):
    """設定檔有錯；訊息會列出所有問題，一次改完。"""


# ───────────────────────── 設定 ─────────────────────────


@dataclass
class Config:
    speed_mm_s: float | None
    part_length_mm: float
    part_gap_mm: float | None
    camera_type: str
    source: str | int
    resolution: tuple[int, int]
    fps: float
    shutter_s: float
    readout_ms: float
    motion_axis: str
    direction: int | None  # +1 = 座標增加方向（往右 / 往下），-1 = 反方向，None = 自動判斷
    mm_per_px: float
    model: str | None
    thresholds: str | None
    defect_classes: list | None
    imgsz: int
    roi: tuple[int, int, int, int] | None
    min_defect_mm: float
    presence: str
    part_class: str
    background_frames: int
    vote_min_frames: int
    min_valid_frames: int
    max_blur_px: float
    device: str | None
    max_escape_rate: float
    max_overkill_rate: float
    output_dir: Path
    save_ok_ratio: float
    warnings: list = field(default_factory=list)

    @property
    def roi_box(self) -> tuple[int, int, int, int]:
        return self.roi or (0, 0, self.resolution[0], self.resolution[1])

    @property
    def roi_len_px(self) -> int:
        x1, y1, x2, y2 = self.roi_box
        return (x2 - x1) if self.motion_axis == "x" else (y2 - y1)

    @property
    def roi_len_mm(self) -> float:
        return self.roi_len_px * self.mm_per_px

    @property
    def part_len_px(self) -> float:
        return self.part_length_mm / self.mm_per_px


def parse_shutter(value, warnings: list) -> float:
    """接受 "1/2000"、0.0005；填 2000 這種大於 1 的數字時視為 1/2000 秒。"""
    if isinstance(value, str):
        s = value.replace(" ", "")
        m = re.fullmatch(r"1/(\d+(?:\.\d+)?)", s)
        if m:
            return 1 / float(m.group(1))
        value = float(s)
    value = float(value)
    if value <= 0:
        raise ValueError("快門必須大於 0")
    if value >= 1:
        warnings.append(f"快門填了 {value:g}，視為 1/{value:g} 秒")
        return 1 / value
    return value


def fmt_shutter(s: float) -> str:
    return f"1/{1 / s:.0f} 秒"


def load_config(path, overrides: dict | None = None) -> Config:
    """讀設定檔並做防呆檢查。所有錯誤一次列出；警告放在 cfg.warnings。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"找不到設定檔 {path}（可以複製 config.example.yaml 來改）")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for dotted, v in (overrides or {}).items():
        if v is None:
            continue
        section, _, key = dotted.partition(".")
        raw.setdefault(section, {})[key] = v
    return build_config(raw, base_dir=path.parent)


def build_config(raw: dict, base_dir: Path = Path(".")) -> Config:
    errors, warns = [], []
    line, cam, ins, crit = (raw.get(k) or {} for k in ("line", "camera", "inspect", "criteria"))

    def number(section, d, key, *, default=None, required=False, lo=None, hi=None, integer=False):
        v = d.get(key)
        if v is None:  # 沒寫或寫 null 都用預設值
            v = default
        if v is None:
            if required:
                errors.append(f"{section}.{key} 必須填寫")
            return None
        try:
            v = int(v) if integer and float(v).is_integer() else float(v)
        except (TypeError, ValueError):
            errors.append(f"{section}.{key} = {v!r} 不是數字")
            return None
        if integer and not isinstance(v, int):
            errors.append(f"{section}.{key} = {v} 必須是整數")
            return None
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            errors.append(f"{section}.{key} = {v:g} 超出合理範圍 {lo:g} ~ {hi:g}")
            return None
        return v

    speed = number("line", line, "speed_mm_s", lo=0.1, hi=20000)
    if speed is not None and speed < 5:
        warns.append(f"line.speed_mm_s = {speed:g} mm/s 非常慢；如果你填的是 m/s，請乘以 1000（例如 0.5 m/s = 500）")
    part_len = number("line", line, "part_length_mm", required=True, lo=0.1, hi=10000)
    gap = number("line", line, "part_gap_mm", lo=0, hi=100000)

    cam_type = str(cam.get("type", "iphone")).lower()
    if cam_type not in CAMERA_PRESETS:
        errors.append(f"camera.type = {cam_type!r}，只能是 {' / '.join(CAMERA_PRESETS)}")
        cam_type = "iphone"
    res = cam.get("resolution")
    resolution = None
    if not (isinstance(res, (list, tuple)) and len(res) == 2 and all(isinstance(x, int) and x > 0 for x in res)):
        errors.append(f"camera.resolution = {res!r}，要寫成 [寬, 高]，例如 [1920, 1080]")
    else:
        resolution = (int(res[0]), int(res[1]))
    fps = number("camera", cam, "fps", required=True, lo=1, hi=2000)
    if fps and cam_type == "iphone" and fps > 240:
        warns.append(f"camera.fps = {fps:g}：iPhone 錄影最高約 240 FPS，請確認")
    shutter = None
    try:
        shutter = parse_shutter(cam.get("shutter", "1/1000"), warns)
    except (TypeError, ValueError) as e:
        errors.append(f"camera.shutter = {cam.get('shutter')!r} 看不懂（{e}）；請寫成 \"1/2000\" 這種格式")
    if shutter and fps and shutter > 1 / fps:
        errors.append(f"快門 {fmt_shutter(shutter)} 比每幀時間 1/{fps:g} 秒還長，相機做不到；請把快門調快")
    elif shutter and shutter > 1 / 250:
        warns.append(f"快門 {fmt_shutter(shutter)} 偏慢，移動中的工件容易模糊（建議 1/1000 以上）")
    readout = number("camera", cam, "readout_ms", default=CAMERA_PRESETS[cam_type], lo=0, hi=200)
    axis = str(cam.get("motion_axis", "x")).lower()
    if axis not in ("x", "y"):
        errors.append(f"camera.motion_axis = {axis!r}，只能是 x（左右移動）或 y（上下移動）")
        axis = "x"
    dir_raw = str(cam.get("motion_direction", "auto")).strip().lower()
    direction = {"auto": None, "+": 1, "right": 1, "down": 1, "-": -1, "left": -1, "up": -1}.get(dir_raw, "bad")
    if direction == "bad":
        errors.append(f"camera.motion_direction = {dir_raw!r}，只能是 auto、+（往右 / 往下）或 -（往左 / 往上）")
        direction = None

    mm_per_px = number("camera", cam, "mm_per_px", lo=1e-5, hi=100)
    fov = number("camera", cam, "fov_mm", lo=0.1, hi=100000)
    if resolution and fov:
        from_fov = fov / (resolution[0] if axis == "x" else resolution[1])
        if mm_per_px is None:
            mm_per_px = from_fov
        elif abs(from_fov - mm_per_px) / mm_per_px > 0.1:
            warns.append(f"camera.fov_mm 換算是 {from_fov:.4f} mm/px，和 mm_per_px = {mm_per_px:.4f} 差超過 10%；"
                         "以 mm_per_px 為準（建議用 calib 指令重新量）")
    if mm_per_px is None and resolution:
        errors.append("camera.mm_per_px 和 camera.fov_mm 至少要填一個（建議用 calib 指令放尺量）")

    roi = ins.get("roi")
    if roi is not None:
        ok = isinstance(roi, (list, tuple)) and len(roi) == 4 and all(isinstance(v, int) for v in roi)
        if not ok or not resolution:
            errors.append(f"inspect.roi = {roi!r}，要寫成 [x1, y1, x2, y2] 整數像素座標，或 null")
            roi = None
        else:
            x1, y1, x2, y2 = roi
            if not (0 <= x1 < x2 <= resolution[0] and 0 <= y1 < y2 <= resolution[1]):
                errors.append(f"inspect.roi = {roi} 超出畫面 {resolution[0]}×{resolution[1]}，或座標順序錯誤")
                roi = None
            else:
                roi = tuple(roi)

    imgsz = number("inspect", ins, "imgsz", default=640, lo=32, hi=4096, integer=True)
    if imgsz and imgsz % 32:
        new = int(math.ceil(imgsz / 32) * 32)
        warns.append(f"inspect.imgsz = {imgsz} 不是 32 的倍數，改用 {new}")
        imgsz = new
    min_defect = number("inspect", ins, "min_defect_mm", default=0.5, lo=0.001, hi=1000)
    presence = str(ins.get("presence", "background")).lower()
    if presence not in ("background", "model"):
        errors.append(f"inspect.presence = {presence!r}，只能是 background 或 model")
    bg_frames = number("inspect", ins, "background_frames", default=30, lo=6, hi=10000, integer=True)
    vote = number("inspect", ins, "vote_min_frames", default=2, lo=1, hi=100, integer=True)
    valid = number("inspect", ins, "min_valid_frames", default=3, lo=1, hi=100, integer=True)
    if vote and valid and valid < vote:
        errors.append(f"inspect.min_valid_frames（{valid}）不能小於 vote_min_frames（{vote}）")
    max_blur = number("inspect", ins, "max_blur_px", default=1.0, lo=0.1, hi=20)
    esc = number("criteria", crit, "max_escape_rate", default=0.0, lo=0, hi=1)
    ovk = number("criteria", crit, "max_overkill_rate", default=0.02, lo=0, hi=1)
    out = raw.get("output") or {}
    save_ok = number("output", out, "save_ok_ratio", default=0.05, lo=0, hi=1)

    defect_classes = ins.get("defect_classes")
    if defect_classes is not None and not (isinstance(defect_classes, list) and defect_classes):
        errors.append("inspect.defect_classes 要寫成類別名稱清單，例如 [scratch, dent]，或 null")
        defect_classes = None

    if errors:
        raise ConfigError("設定檔有以下問題：\n  - " + "\n  - ".join(errors))

    def resolve(p):
        if p in (None, ""):
            return None
        p = Path(str(p)).expanduser()
        return str(p if p.is_absolute() else (base_dir / p))

    source = cam.get("source", 0)
    cfg = Config(
        speed_mm_s=speed, part_length_mm=part_len, part_gap_mm=gap, camera_type=cam_type,
        source=source if isinstance(source, int) else str(source), resolution=resolution, fps=fps,
        shutter_s=shutter, readout_ms=readout, motion_axis=axis, direction=direction, mm_per_px=mm_per_px,
        model=resolve(ins.get("model")), thresholds=resolve(ins.get("thresholds")), defect_classes=defect_classes,
        imgsz=imgsz, roi=roi, min_defect_mm=min_defect, presence=presence,
        part_class=str(ins.get("part_class", "part")), background_frames=bg_frames, vote_min_frames=vote,
        min_valid_frames=valid, max_blur_px=max_blur, device=None if ins.get("device") is None else str(ins["device"]),
        max_escape_rate=esc, max_overkill_rate=ovk, output_dir=Path(resolve(out.get("dir", "camtest_runs"))),
        save_ok_ratio=save_ok, warnings=warns,
    )
    _cross_checks(cfg)
    return cfg


def _cross_checks(cfg: Config) -> None:
    """設定之間互相矛盾、但還不到錯誤的情況。"""
    if cfg.roi_len_mm < cfg.part_length_mm:
        raise ConfigError(
            f"設定檔有以下問題：\n  - 檢測區沿移動方向只有 {cfg.roi_len_mm:.1f} mm，比工件 {cfg.part_length_mm:g} mm 短，"
            "工件永遠無法完整入鏡；請把相機拉遠、加大 roi，或改用多台相機分段拍")
    if cfg.part_gap_mm is not None and cfg.part_gap_mm / cfg.mm_per_px < 6:
        cfg.warnings.append(f"工件間距 {cfg.part_gap_mm:g} mm 在畫面上只有 {cfg.part_gap_mm / cfg.mm_per_px:.1f} px，"
                            "前後工件會黏在一起，無法分開計數；請加大間距")
    if cfg.part_gap_mm is None:
        cfg.warnings.append("line.part_gap_mm 沒填：無法估計每分鐘幾件，也無法檢查工件會不會黏在一起")


# ───────────────────────── 規劃計算 ─────────────────────────


@dataclass
class PlanRow:
    speed: float
    blur_px: float
    frames_full: float
    skew_mm: float
    shutter_needed_s: float
    parts_per_min: float | None
    verdict: str  # 可以 / 注意 / 不行
    reasons: list


def static_checks(cfg: Config) -> list[tuple[str, str]]:
    """和速度無關的檢查：缺陷在影像上夠不夠大。回傳 [(等級, 訊息)]。"""
    out = []
    defect_px = cfg.min_defect_mm / cfg.mm_per_px
    x1, y1, x2, y2 = cfg.roi_box
    scale = min(1.0, cfg.imgsz / max(x2 - x1, y2 - y1))  # YOLO 會把檢測區縮到 imgsz
    model_px = defect_px * scale
    if defect_px < MIN_DEFECT_PX:
        out.append(("不行", f"最小缺陷 {cfg.min_defect_mm:g} mm 在影像上只有 {defect_px:.1f} px（至少要 {MIN_DEFECT_PX} px）："
                           "相機要更靠近、或用更高解析度"))
    elif model_px < YOLO_MIN_DEFECT_PX:
        need = math.ceil(YOLO_MIN_DEFECT_PX / defect_px * max(x2 - x1, y2 - y1) / 32) * 32
        fix = (f"把 roi 縮小只框住工件，或 imgsz 提高到約 {need}" if need <= 1280
               else f"把 roi 縮小只框住工件，或把相機拉近（只靠 imgsz 要到約 {need}，太大跑不動）")
        out.append(("注意", f"最小缺陷縮到模型輸入後只剩 {model_px:.1f} px（建議 ≥ {YOLO_MIN_DEFECT_PX} px）：{fix}"))
    else:
        out.append(("可以", f"最小缺陷在原圖 {defect_px:.1f} px、模型輸入 {model_px:.1f} px"))
    if cfg.readout_ms > 0:
        out.append(("說明", f"{cfg.camera_type} 是滾動快門（讀出約 {cfg.readout_ms:g} ms）：高速時影像會歪斜，"
                           "外觀瑕疵影響較小，尺寸量測會不準"))
    return out


def plan_row(cfg: Config, v: float) -> PlanRow:
    blur = v * cfg.shutter_s / cfg.mm_per_px
    t_full = max(cfg.roi_len_mm - cfg.part_length_mm, 0.0) / v
    frames_full = t_full * cfg.fps
    skew_mm = v * cfg.readout_ms / 1000
    shutter_needed = cfg.max_blur_px * cfg.mm_per_px / v
    ppm = 60 * v / (cfg.part_length_mm + cfg.part_gap_mm) if cfg.part_gap_mm is not None else None
    verdict, reasons = "可以", []
    if blur > cfg.max_blur_px:
        verdict = "不行"
        reasons.append(f"模糊 {blur:.1f}px，快門要快於 {fmt_shutter(shutter_needed)}")
    if frames_full < cfg.min_valid_frames:
        verdict = "不行"
        reasons.append(f"每件只拍到 {frames_full:.1f} 幀（需要 {cfg.min_valid_frames}）")
    elif frames_full < cfg.min_valid_frames * 1.5 and verdict == "可以":
        verdict = "注意"
        reasons.append("幀數餘裕很小")
    if cfg.part_gap_mm is not None:  # 每幀移動接近一個間距時，影片上分不出是哪一件（車輪倒轉的錯覺）
        step = v / cfg.fps
        pitch = cfg.part_length_mm + cfg.part_gap_mm
        if step > pitch - 0.25 * cfg.part_length_mm:
            verdict = "不行"
            reasons.append(f"每幀移動 {step:.0f} mm，接近工件間距 {pitch:g} mm，影片上分不出是哪一件")
        elif step > 0.5 * pitch:
            if verdict == "可以":
                verdict = "注意"
            reasons.append("每幀移動超過間距一半：camera.motion_direction 一定要設定，不能用 auto")
    if skew_mm > 0.1 * cfg.part_length_mm and verdict == "可以":  # 形狀明顯變形；只影響尺寸量測時不必擋
        verdict = "注意"
        reasons.append(f"滾動快門歪斜 {skew_mm:.1f} mm（超過工件長 10%，尺寸量測會不準）")
    return PlanRow(v, blur, frames_full, skew_mm, shutter_needed, ppm, verdict, reasons)


def max_speed(cfg: Config) -> tuple[float, str]:
    """目前設定下能用的最高速度（mm/s）與限制因素。"""
    v_blur = cfg.max_blur_px * cfg.mm_per_px / cfg.shutter_s
    v_frames = (cfg.roi_len_mm - cfg.part_length_mm) * cfg.fps / cfg.min_valid_frames
    if v_blur <= v_frames:
        return v_blur, f"動態模糊（快門 {fmt_shutter(cfg.shutter_s)}）"
    return v_frames, f"每件幀數（{cfg.fps:g} FPS、檢測區 {cfg.roi_len_mm:.0f} mm）"


def plan_report(cfg: Config, speeds=None) -> str:
    speeds = sorted(set(speeds or ([cfg.speed_mm_s] if cfg.speed_mm_s else DEFAULT_SPEEDS)))
    v_max, limit = max_speed(cfg)
    light = (1 / 60) / cfg.shutter_s
    lines = ["# 相機實測規劃", "", "## 目前設定", "",
             f"- 相機：{cfg.camera_type}，{cfg.resolution[0]}×{cfg.resolution[1]}、{cfg.fps:g} FPS、快門 {fmt_shutter(cfg.shutter_s)}",
             f"- 每像素 {cfg.mm_per_px:.4f} mm；檢測區沿移動方向 {cfg.roi_len_mm:.1f} mm；工件長 {cfg.part_length_mm:g} mm",
             f"- 判定：至少 {cfg.vote_min_frames} 幀看到缺陷判 NG；每件至少完整拍到 {cfg.min_valid_frames} 幀",
             f"- 快門 {fmt_shutter(cfg.shutter_s)} 需要的光量約是一般 1/60 秒的 {light:.0f} 倍，要準備強光源",
             "", "## 和速度無關的檢查", ""]
    lines += [f"- **{lvl}**：{msg}" for lvl, msg in static_checks(cfg)]
    lines += ["", f"## 最高可用速度：約 **{v_max:.0f} mm/s**（{v_max * 60 / 1000:.1f} m/min），限制因素：{limit}", "",
              "| 速度 mm/s | 動態模糊 px | 每件完整幀數 | 滾動快門歪斜 mm | 需要的快門 | 每分鐘件數 | 結果 |",
              "|---|---|---|---|---|---|---|"]
    for v in sorted(set(speeds) | {round(v_max)}):
        if v <= 0:
            continue
        r = plan_row(cfg, v)
        ppm = f"{r.parts_per_min:.0f}" if r.parts_per_min is not None else "—"
        note = f"（{'；'.join(r.reasons)}）" if r.reasons else ""
        lines.append(f"| {v:g} | {r.blur_px:.2f} | {r.frames_full:.1f} | {r.skew_mm:.1f} | "
                     f"{fmt_shutter(r.shutter_needed_s)} | {ppm} | {r.verdict}{note} |")
    if cfg.warnings:
        lines += ["", "## 設定提醒", ""] + [f"- {w}" for w in cfg.warnings]
    return "\n".join(lines) + "\n"


# ───────────────────────── 工件偵測與追蹤 ─────────────────────────


def occupancy_segments(mask: np.ndarray, axis: str, min_fill: float = 0.08, min_len_px: int = 4,
                       merge_gap_px: int = 3) -> list[tuple[int, int]]:
    """把前景遮罩投影到運動方向，回傳被工件佔據的區段 [(start, end)]（end 不含）。"""
    prof = (mask > 0).mean(axis=0 if axis == "x" else 1)
    occ = prof > min_fill
    segs, start = [], None
    for i, on in enumerate(np.append(occ, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segs.append([start, i])
            start = None
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap_px:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_len_px]


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple  # (x1, y1, x2, y2)，檢測區座標


@dataclass
class Track:
    id: int
    start: float
    end: float
    first_t: float
    last_t: float
    centers: list = field(default_factory=list)  # [(t, center, full)]
    frames_seen: int = 0
    frames_full: int = 0
    ng_frames: int = 0
    max_len: float = 0.0
    max_conf: dict = field(default_factory=dict)
    dark_frames: int = 0
    bright_frames: int = 0
    missed: int = 0
    best_score: float = -1.0
    best_image: np.ndarray | None = None
    mid_image: np.ndarray | None = None

    def velocity(self) -> float | None:
        """px/s；優先用完整入鏡的幀（邊緣幀的中心會偏）。"""
        pts = [(t, c) for t, c, full in self.centers if full]
        if len(pts) < 2:
            pts = [(t, c) for t, c, _ in self.centers]
        if len(pts) < 2 or pts[-1][0] == pts[0][0]:
            return None
        t, c = np.array(pts).T
        return float(np.polyfit(t, c, 1)[0])


@dataclass
class PartResult:
    id: int
    t_first: float
    t_last: float
    frames_seen: int
    frames_full: int
    ng_frames: int
    length_mm: float
    speed_mm_s: float | None
    max_conf: dict
    verdict: str
    reason: str
    image: np.ndarray | None = None


class Inspector:
    """逐幀餵影像，輸出每個工件的判定。

    detector(roi_bgr) → (缺陷 list[Detection], 工件框 list[(x1, y1, x2, y2)])，座標相對於檢測區。
    """

    EDGE_MARGIN = 2  # 區段碰到檢測區邊緣幾 px 內，視為沒有完整入鏡
    MAX_MISSED = 2  # 追蹤中的工件連續幾幀沒出現就結案

    def __init__(self, cfg: Config, detector, thresholds: dict):
        self.cfg = cfg
        self.detector = detector
        self.thr = thresholds
        self.tracks: list[Track] = []
        self.next_id = 1
        self.bg_frames: list[np.ndarray] = []
        self.bg: np.ndarray | None = None
        self.diff_thr = None
        self.noise_tracks = 0
        self.stray_detections = 0  # 落在工件外的缺陷偵測（背景誤報），數量多代表模型或門檻有問題
        self.global_v = None  # 已結案工件的平均速度（px/s），新工件第一幀還沒有速度時用它預測位置
        self.direction = cfg.direction  # +1 / -1；設定為 auto 時由第一個從邊緣進來的工件決定
        self.last_segments: list = []
        self.last_detections: list = []
        self.status = "建立背景中，請保持畫面中沒有工件" if cfg.presence == "background" else "檢測中"

    # ---- 背景 ----
    def _gray(self, roi):
        return cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)

    def _build_background(self):
        frames = np.stack(self.bg_frames)
        half = len(frames) // 2
        bg = np.median(frames[:half], axis=0)
        noise = float(np.percentile(np.abs(frames[half:] - bg), 99.5))
        if noise > 40:
            raise ValueError(
                f"背景不穩定（雜訊 {noise:.0f}）：開頭的畫面裡可能有工件或手在動、輸送帶有花紋，或燈光在閃。"
                "請讓開頭至少 1 秒畫面中沒有任何東西在動；輸送帶有花紋時改用 presence: model")
        self.bg = bg
        self.diff_thr = max(18.0, noise * 1.5)
        self.bg_frames = []
        self.status = "檢測中"

    def reset_background(self):
        self.bg, self.bg_frames = None, []
        self.status = "建立背景中，請保持畫面中沒有工件"

    # ---- 主流程 ----
    def feed(self, frame: np.ndarray, t: float) -> list[PartResult]:
        cfg = self.cfg
        x1, y1, x2, y2 = cfg.roi_box
        roi = frame[y1:y2, x1:x2]
        if cfg.presence == "background" and self.bg is None:
            self.bg_frames.append(self._gray(roi))
            if len(self.bg_frames) >= cfg.background_frames:
                self._build_background()
            return []
        defects, part_boxes = self.detector(roi)
        self.last_detections = defects
        if cfg.presence == "background":
            gray = self._gray(roi)
            mask = (np.abs(gray - self.bg) > self.diff_thr).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            segs = occupancy_segments(mask, cfg.motion_axis)
            if not segs:  # 沒有工件時慢慢更新背景，適應光線變化
                self.bg = 0.95 * self.bg + 0.05 * gray
        else:
            a = 0 if cfg.motion_axis == "x" else 1
            segs = sorted((int(b[a]), int(b[a + 2])) for b in part_boxes)
        self.last_segments = segs
        return self._update(segs, defects, roi, t)

    def _update(self, segs, defects, roi, t) -> list[PartResult]:
        cfg, axis_len = self.cfg, self.cfg.roi_len_px
        a = 0 if cfg.motion_axis == "x" else 1
        match = self._match(segs, t, axis_len)
        for j, (s, e) in enumerate(segs):
            if j in match:
                tr = self.tracks[match[j]]
            else:
                tr = Track(self.next_id, s, e, t, t)
                self.next_id += 1
                self.tracks.append(tr)
                if self.direction is None:  # 第一個從邊緣進來的工件決定移動方向
                    at_start, at_end = s <= self.EDGE_MARGIN, e >= axis_len - self.EDGE_MARGIN
                    if at_start != at_end:
                        self.direction = 1 if at_start else -1
            self._observe(tr, s, e, axis_len, a, defects, roi, t)
        finished = []
        for tr in list(self.tracks):
            if tr.last_t != t:
                tr.missed += 1
                if tr.missed > self.MAX_MISSED:
                    self.tracks.remove(tr)
                    res = self._finalize(tr)
                    if res:
                        finished.append(res)
        self.stray_detections += sum(
            1 for d in defects if not any(s <= (d.box[a] + d.box[a + 2]) / 2 < e for s, e in segs))
        return finished

    def _match(self, segs, t, axis_len) -> dict:
        """區段 → 追蹤中工件的配對（回傳 {區段索引: 工件索引}）。

        工件只會往同一個方向走：絕不接受「往回走」的配對。高速時下一件會落在上一件剛才的位置
        （像車輪倒轉的錯覺），只靠位置重疊會配錯；所以知道方向後，改配「前方最近」的區段。
        """
        part_px = self.cfg.part_len_px
        cands = []
        for i, tr in enumerate(self.tracks):
            v = tr.velocity() or self.global_v
            dt = t - tr.last_t
            for j, (s, e) in enumerate(segs):
                if self.direction is None:  # 方向未知：只接受和上一個位置重疊的區段
                    if min(tr.end, e) - max(tr.start, s) > 0:
                        cands.append((abs((s + e) / 2 - (tr.start + tr.end) / 2), i, j))
                    continue
                disp = self._displacement((tr.start, tr.end), (s, e), axis_len)
                fwd = disp * self.direction
                if fwd < -0.25 * part_px or fwd > 0.6 * axis_len:
                    continue
                if v:
                    cost = abs(disp - v * dt)
                    if cost > max(part_px, 0.5 * abs(v * dt)):
                        continue
                else:
                    cost = abs(fwd)
                cands.append((cost, i, j))
        used_t, match = set(), {}
        for _, i, j in sorted(cands):
            if i not in used_t and j not in match:
                used_t.add(i)
                match[j] = i
        return match

    def _displacement(self, old, new, axis_len) -> float:
        """兩個區段之間的位移。用沒有被畫面邊緣切掉的那一側邊緣量：剛進場或快出場的工件
        只露出一部分，中心點的移動會比工件本身慢，用中心量會把下一件誤認成上一件。"""
        (s0, e0), (s1, e1) = old, new
        m = self.EDGE_MARGIN
        free = lambda x: m < x < axis_len - m  # noqa: E731
        front0, front1, rear0, rear1 = (e0, e1, s0, s1) if (self.direction or 1) > 0 else (s0, s1, e0, e1)
        if free(front0) and free(front1):
            return front1 - front0
        if free(rear0) and free(rear1):
            return rear1 - rear0
        return (s1 + e1) / 2 - (s0 + e0) / 2

    def _observe(self, tr: Track, s, e, axis_len, a, defects, roi, t):
        full = s > self.EDGE_MARGIN and e < axis_len - self.EDGE_MARGIN
        tr.start, tr.end, tr.last_t, tr.missed = s, e, t, 0
        tr.frames_seen += 1
        tr.frames_full += int(full)
        tr.centers.append((t, (s + e) / 2, full))
        tr.max_len = max(tr.max_len, e - s)
        region = roi[:, s:e] if a == 0 else roi[s:e, :]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        tr.dark_frames += int(gray.mean() < 25)
        tr.bright_frames += int((gray >= 250).mean() > 0.05)
        ng, score = False, 0.0
        for d in defects:
            c = (d.box[a] + d.box[a + 2]) / 2
            if not s <= c < e:
                continue
            tr.max_conf[d.name] = max(tr.max_conf.get(d.name, 0.0), d.conf)
            if d.conf >= self.thr.get(d.name, 0.25):
                ng = True
            score = max(score, d.conf)
        tr.ng_frames += int(ng)
        if score > tr.best_score:  # 保留最可疑的那一幀，給人工複判與重訓
            tr.best_score, tr.best_image = score, region.copy()
        if full and tr.mid_image is None:
            tr.mid_image = region.copy()

    def _finalize(self, tr: Track) -> PartResult | None:
        cfg = self.cfg
        if tr.max_len < 0.4 * cfg.part_len_px:  # 太小的區段是雜訊（灰塵、反光），不算工件
            self.noise_tracks += 1
            return None
        # 只用完整拍到至少 3 幀的工件量速度；幀數太少時中心點都在畫面邊緣，量出來會錯
        v = tr.velocity() if tr.frames_full >= 3 else None
        speed = abs(v) * cfg.mm_per_px if v is not None else None
        length_mm = tr.max_len * cfg.mm_per_px
        if speed is not None and speed < 1.0:
            verdict, reason = "CHECK", ("工件看起來沒在動：產線停了，或速度剛好每幀走一個間距（影片上的錯覺），"
                                        "請改變 FPS 或速度再測")
            speed = None
        else:
            verdict, reason = decide(tr, cfg)
            if v:
                self.global_v = v if self.global_v is None else 0.7 * self.global_v + 0.3 * v
        image = tr.best_image if verdict != "OK" or tr.mid_image is None else tr.mid_image
        return PartResult(tr.id, tr.first_t, tr.last_t, tr.frames_seen, tr.frames_full, tr.ng_frames, length_mm,
                          speed, dict(tr.max_conf), verdict, reason, image)

    def finish(self) -> list[PartResult]:
        out = [r for r in (self._finalize(tr) for tr in self.tracks) if r]
        self.tracks = []
        return out


def decide(tr: Track, cfg: Config) -> tuple[str, str]:
    """多幀投票＋防呆：拿不準的一律「需人工確認」，絕不放行成 OK。"""
    if tr.frames_full < cfg.min_valid_frames:
        return "CHECK", f"只完整拍到 {tr.frames_full} 幀（需要 {cfg.min_valid_frames}），速度太快或 FPS 太低"
    if tr.max_len > 1.6 * cfg.part_len_px:
        return "CHECK", f"長度 {tr.max_len * cfg.mm_per_px:.0f} mm 超過設定的 1.6 倍，疑似兩件黏在一起"
    if tr.dark_frames > tr.frames_seen / 2:
        return "CHECK", "影像太暗"
    if tr.bright_frames > tr.frames_seen / 2:
        return "CHECK", "影像過曝"
    if tr.ng_frames >= cfg.vote_min_frames:
        return "NG", f"{tr.ng_frames} 幀看到缺陷"
    if tr.ng_frames > 0:
        return "CHECK", f"只有 {tr.ng_frames} 幀看到缺陷（判 NG 需要 {cfg.vote_min_frames} 幀）"
    return "OK", ""


# ───────────────────────── 速度測試：答案與評分 ─────────────────────────


@dataclass
class ManifestRow:
    video: Path
    speed_mm_s: float | None
    n_parts: int
    ng_positions: set


def load_manifest(path) -> list[ManifestRow]:
    """CSV 欄位：video, speed_mm_s（可空白，會從影片量）, n_parts, ng_parts（NG 是第幾件，例如 "3,8"）。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"找不到 manifest {path}")
    rows, errors = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need = {"video", "n_parts", "ng_parts"}
        if not need <= set(reader.fieldnames or []):
            raise ConfigError(f"manifest 欄位要有 {sorted(need)}（speed_mm_s 可選），現在是 {reader.fieldnames}")
        for i, r in enumerate(reader, 2):
            video = Path(r["video"].strip())
            video = video if video.is_absolute() else path.parent / video
            if not video.is_file():
                errors.append(f"第 {i} 列：找不到影片 {video}")
            try:
                n = int(r["n_parts"])
                assert n > 0
            except (ValueError, AssertionError):
                errors.append(f"第 {i} 列：n_parts = {r['n_parts']!r} 要是正整數")
                continue
            ng_raw = (r.get("ng_parts") or "").replace("，", ",").strip()
            try:
                ng = [int(x) for x in ng_raw.split(",") if x.strip()]
            except ValueError:
                errors.append(f"第 {i} 列：ng_parts = {ng_raw!r} 要寫成以逗號分隔的件數，例如 3,8")
                continue
            if len(set(ng)) != len(ng) or any(not 1 <= x <= n for x in ng):
                errors.append(f"第 {i} 列：ng_parts = {ng} 有重複、或超出 1~{n}")
            speed_raw = (r.get("speed_mm_s") or "").strip()
            speed = None
            if speed_raw:
                try:
                    speed = float(speed_raw)
                except ValueError:
                    errors.append(f"第 {i} 列：speed_mm_s = {speed_raw!r} 不是數字")
            rows.append(ManifestRow(video, speed, n, set(ng)))
    if errors:
        raise ConfigError("manifest 有以下問題：\n  - " + "\n  - ".join(errors))
    if not rows:
        raise ConfigError("manifest 是空的")
    return rows


def score_run(parts: list[PartResult], row: ManifestRow) -> dict:
    """依通過順序和答案比對。計數不符時整段結果作廢（無法對齊就不能算漏檢率）。"""
    parts = sorted(parts, key=lambda p: p.t_first)
    speeds = [p.speed_mm_s for p in parts if p.speed_mm_s]
    res = {"counted": len(parts), "expected": row.n_parts, "valid": len(parts) == row.n_parts,
           "measured_speed": float(np.median(speeds)) if speeds else None,
           "verdicts": {v: sum(p.verdict == v for p in parts) for v in VERDICTS}}
    if not res["valid"]:
        res["problem"] = (f"計數不符：數到 {len(parts)} 件，答案 {row.n_parts} 件"
                          "（工件黏在一起、背景雜訊、或影片開頭/結尾切到工件）")
        return res
    n_ng = len(row.ng_positions)
    n_ok = row.n_parts - n_ng
    escapes = [i for i, p in enumerate(parts, 1) if i in row.ng_positions and p.verdict == "OK"]
    overkill = [i for i, p in enumerate(parts, 1) if i not in row.ng_positions and p.verdict != "OK"]
    checks = [i for i, p in enumerate(parts, 1) if p.verdict == "CHECK"]
    res.update(
        escape_rate=len(escapes) / n_ng if n_ng else None,
        overkill_rate=len(overkill) / n_ok if n_ok else None,
        check_rate=len(checks) / row.n_parts,
        escaped_positions=escapes, overkill_positions=overkill,
        check_reasons=sorted({parts[i - 1].reason for i in checks}),
    )
    return res


def speed_limit(results: list[dict], cfg: Config) -> float | None:
    """由慢到快，第一個不合格的速度之前，最快的合格速度。"""
    best = None
    for r in sorted(results, key=lambda r: r["speed"] or 0):
        ok = (r["valid"] and (r["escape_rate"] or 0) <= cfg.max_escape_rate
              and (r["overkill_rate"] or 0) <= cfg.max_overkill_rate)
        if not ok:
            break
        best = r["speed"]
    return best


def load_thresholds(path: str | None, model_names: list[str], defect_classes: list | None,
                    part_class: str, presence: str) -> tuple[dict, list[str]]:
    """讀 Kaggle 輸出的 thresholds.json，並檢查和模型是否對得上。回傳 (門檻, 警告)。"""
    warns = []
    data = {}
    if path:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"找不到門檻檔 {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if "normal_class" in data or "abnormal_threshold" in data:
            raise ConfigError("這是「分類模型」的門檻檔；相機實測需要偵測（detect）或分割（segment）模型")
        unknown = [k for k in data if k not in model_names]
        if unknown:
            raise ConfigError(f"門檻檔的類別 {unknown} 不在模型裡（模型類別：{model_names}）；"
                              "門檻檔和模型可能不是同一次訓練的輸出")
    if presence == "model" and part_class not in model_names:
        raise ConfigError(f"presence: model 需要模型裡有 {part_class!r} 類別；模型類別：{model_names}")
    classes = defect_classes or [n for n in model_names if n != part_class]
    missing = [c for c in classes if c not in model_names]
    if missing:
        raise ConfigError(f"inspect.defect_classes 的 {missing} 不在模型裡；模型類別：{model_names}")
    thr = {}
    for c in classes:
        if c in data:
            thr[c] = float(data[c])
        else:
            thr[c] = 0.25
            if path:
                warns.append(f"門檻檔沒有 {c!r}，用預設 0.25")
    return thr, warns
