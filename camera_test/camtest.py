#!/usr/bin/env python3
"""相機實測工具：先用 iPhone 測，之後換監視鏡頭或工業相機，流程完全相同（改 camera.type 即可）。

  plan   依設定算出各種速度下能不能檢（不需要模型、不需要相機）
  calib  畫面中放一把尺，量出每像素幾 mm
  test   用錄好的影片做速度測試，和答案比對，找出最高可用速度
  live   接相機即時檢測（Mac 上的 iPhone 接續互通相機、USB 攝影機、RTSP 串流）

所有參數都在設定檔（複製 config.example.yaml 來改），常用的也能在指令列覆蓋，例如 --speed 300。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camtest_core as core  # noqa: E402

PART_CONF = 0.4  # presence: model 時，工件框的信心門檻
COLORS = {"OK": (60, 180, 60), "NG": (40, 40, 220), "CHECK": (0, 170, 240)}


# ───────────────────────── 模型 ─────────────────────────


class YoloDetector:
    """把檢測區丟給 YOLO，回傳缺陷與工件框（座標相對於檢測區）。"""

    def __init__(self, cfg: core.Config):
        if not cfg.model:
            raise core.ConfigError("inspect.model 沒填（填 Kaggle 輸出的 best.pt 路徑）")
        if not Path(cfg.model).is_file():
            raise core.ConfigError(f"找不到模型 {cfg.model}（把 Kaggle 輸出的 best.pt 放到這個位置）")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise core.ConfigError("請先安裝 ultralytics：pip install -r camera_test/requirements.txt") from e
        self.model = YOLO(cfg.model)
        if self.model.task not in ("detect", "segment"):
            raise core.ConfigError(f"模型任務是 {self.model.task}；相機實測需要 detect 或 segment 模型")
        names = [self.model.names[i] for i in sorted(self.model.names)]
        self.thr, warns = core.load_thresholds(cfg.thresholds, names, cfg.defect_classes, cfg.part_class,
                                               cfg.presence)
        cfg.warnings += warns
        self.cfg = cfg
        self.kw = dict(imgsz=cfg.imgsz, conf=max(0.01, min(self.thr.values()) * 0.5), verbose=False)
        if cfg.device is not None:
            self.kw["device"] = cfg.device

    def __call__(self, roi: np.ndarray):
        r = self.model.predict(roi, **self.kw)[0]
        defects, parts = [], []
        if r.boxes is None:
            return defects, parts
        for box, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = self.model.names[int(c)]
            if name == self.cfg.part_class:
                if conf >= PART_CONF:
                    parts.append(tuple(box))
            elif name in self.thr:
                defects.append(core.Detection(name, float(conf), tuple(box)))
        return defects, parts


# ───────────────────────── 輸出 ─────────────────────────


class PartLogger:
    """每個工件寫一列 CSV；NG / 需人工確認一律存圖，OK 依比例抽樣存圖（給閉環重訓用）。"""

    FIELDS = ["run", "part", "verdict", "reason", "t_first", "t_last", "frames_seen", "frames_full",
              "ng_frames", "length_mm", "speed_mm_s", "max_conf", "image"]

    def __init__(self, out_dir: Path, save_ok_ratio: float):
        self.out_dir = out_dir
        self.ratio = save_ok_ratio
        out_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(out_dir / "parts.csv", "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()

    def save(self, run: str, p: core.PartResult) -> None:
        image = ""
        keep = p.verdict != "OK" or int(hashlib.md5(f"{run}/{p.id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < self.ratio
        if keep and p.image is not None:
            path = self.out_dir / "parts" / p.verdict / f"{run}_part{p.id:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), p.image)
            image = str(path.relative_to(self.out_dir))
        self.w.writerow({
            "run": run, "part": p.id, "verdict": p.verdict, "reason": p.reason, "t_first": f"{p.t_first:.3f}",
            "t_last": f"{p.t_last:.3f}", "frames_seen": p.frames_seen, "frames_full": p.frames_full,
            "ng_frames": p.ng_frames, "length_mm": f"{p.length_mm:.1f}",
            "speed_mm_s": "" if p.speed_mm_s is None else f"{p.speed_mm_s:.1f}",
            "max_conf": json.dumps({k: round(v, 3) for k, v in p.max_conf.items()}), "image": image,
        })
        self.f.flush()

    def close(self):
        self.f.close()


def _run_dir(cfg: core.Config, kind: str) -> Path:
    d = cfg.output_dir / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_frame_size(cfg: core.Config, w: int, h: int, what: str, force: bool) -> None:
    """解析度和設定不同時，每像素幾 mm 就錯了，所有尺寸與速度都會算錯。"""
    if (w, h) == cfg.resolution:
        return
    hint = "（直式 / 橫式相反：手機錄影方向要和設定一致）" if (h, w) == cfg.resolution else ""
    msg = f"{what} 是 {w}×{h}，設定檔寫 {cfg.resolution[0]}×{cfg.resolution[1]}{hint}"
    if not force:
        raise core.ConfigError(msg + "；請改設定檔並重新 calib，或確定沒問題時加 --force")
    print(f"⚠️ {msg}（--force 繼續）")


def _print_warnings(cfg: core.Config) -> None:
    for w in cfg.warnings:
        print(f"⚠️ {w}")
    cfg.warnings.clear()


# ───────────────────────── plan / calib ─────────────────────────


def cmd_plan(args) -> None:
    cfg = core.load_config(args.config, {"line.speed_mm_s": args.speed})
    speeds = [float(s) for s in args.speeds.split(",")] if args.speeds else None
    text = core.plan_report(cfg, speeds)
    print(text)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "plan.md").write_text(text, encoding="utf-8")
    print(f"已存到 {cfg.output_dir / 'plan.md'}")


def _first_frame(source) -> np.ndarray:
    if isinstance(source, str) and Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        img = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        raise core.ConfigError(f"讀不到影像：{source}")
    return img


def cmd_calib(args) -> None:
    cfg = core.load_config(args.config)
    img = _first_frame(args.source)
    h, w = img.shape[:2]
    _check_frame_size(cfg, w, h, "校正影像", args.force)
    if args.points:
        x1, y1, x2, y2 = (float(v) for v in args.points.split(","))
    else:
        pts = []

        def on_click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((x, y))

        cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("calib", on_click)
        print(f"在尺上點兩個點（相距 {args.distance_mm:g} mm），按任意鍵結束")
        while len(pts) < 2:
            view = img.copy()
            for p in pts:
                cv2.circle(view, p, 6, (0, 0, 255), -1)
            cv2.imshow("calib", view)
            if cv2.waitKey(30) == 27:
                raise SystemExit("已取消")
        cv2.destroyAllWindows()
        (x1, y1), (x2, y2) = pts
    dist_px = float(np.hypot(x2 - x1, y2 - y1))
    if dist_px < 50:
        raise core.ConfigError(f"兩點只相距 {dist_px:.0f} px，太近會不準；請在尺上點相距較遠的兩點")
    mm_per_px = args.distance_mm / dist_px
    fov = mm_per_px * (w if cfg.motion_axis == "x" else h)
    print(f"每像素 {mm_per_px:.5f} mm；畫面沿移動方向約 {fov:.1f} mm")
    if args.write:
        path = Path(args.config)
        text = path.read_text(encoding="utf-8")
        new, n = re.subn(r"^(\s*mm_per_px:\s*)[^\n#]*", rf"\g<1>{mm_per_px:.5f}  ", text, count=1, flags=re.M)
        if n:
            path.write_text(new, encoding="utf-8")
            print(f"已寫入 {path} 的 camera.mm_per_px")
        else:
            print(f"設定檔裡找不到 mm_per_px，請手動在 camera 底下加：mm_per_px: {mm_per_px:.5f}")


# ───────────────────────── test：錄影速度測試 ─────────────────────────


def run_video(cfg: core.Config, detector, row: core.ManifestRow, logger: PartLogger, force: bool) -> dict:
    cap = cv2.VideoCapture(str(row.video))
    ok, first = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        return {"problem": "影片打不開或沒有畫面（格式不支援時請轉成 mp4/H.264）"}
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    try:  # 用解出來的畫面判斷尺寸：iPhone 影片帶旋轉資訊，屬性裡的寬高不一定是實際方向
        _check_frame_size(cfg, first.shape[1], first.shape[0], f"影片 {row.video.name}", force)
    except core.ConfigError as e:
        return {"problem": str(e)}
    notes = []
    if abs(fps - cfg.fps) / cfg.fps > 0.05:
        notes.append(f"影片 {fps:.1f} FPS 和設定 {cfg.fps:g} FPS 不同（規劃表要用實際 FPS 重算）")
    insp = core.Inspector(cfg, detector, detector.thr)
    parts, idx, t_proc = [], 0, 0.0
    run = row.video.stem
    frame = first
    while frame is not None:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000 if t_ms > 0 or idx == 0 else idx / fps
        idx += 1
        t0 = time.perf_counter()
        try:
            done = insp.feed(frame, t)
        except ValueError as e:  # 背景不穩定
            return {"problem": str(e)}
        t_proc += time.perf_counter() - t0
        for p in done:
            logger.save(run, p)
        parts += done
        ok, frame = cap.read()
        frame = frame if ok else None
    cap.release()
    if cfg.presence == "background" and insp.bg is None:
        return {"problem": f"影片只有 {idx} 幀，連建立背景（{cfg.background_frames} 幀）都不夠"}
    tail = insp.finish()
    for p in tail:
        logger.save(run, p)
    parts += tail
    res = core.score_run(parts, row)
    res["proc_fps"] = idx / t_proc if t_proc else None
    res["noise_tracks"] = insp.noise_tracks
    res["stray_detections"] = insp.stray_detections
    res["notes"] = notes
    return res


def cmd_test(args) -> None:
    cfg = core.load_config(args.config)
    rows = core.load_manifest(args.manifest)
    detector = YoloDetector(cfg)
    _print_warnings(cfg)
    out = _run_dir(cfg, "test")
    logger = PartLogger(out, cfg.save_ok_ratio)
    results = []
    for row in rows:
        print(f"▶ {row.video.name}（宣告速度 {row.speed_mm_s or '未填'} mm/s，{row.n_parts} 件，NG 在第 {sorted(row.ng_positions) or '無'} 件）")
        res = run_video(cfg, detector, row, logger, args.force)
        res.update(video=row.video.name, declared_speed=row.speed_mm_s)
        res.setdefault("valid", False)
        res["speed"] = row.speed_mm_s or res.get("measured_speed")
        if row.speed_mm_s and res.get("measured_speed"):
            diff = abs(res["measured_speed"] - row.speed_mm_s) / row.speed_mm_s
            if diff > 0.25:
                res.setdefault("notes", []).append(
                    f"宣告 {row.speed_mm_s:g} mm/s，影片量到 {res['measured_speed']:.0f} mm/s，差 {diff:.0%}："
                    "請確認 mm_per_px 或宣告速度")
        results.append(res)
        print("   " + (res.get("problem") or
                       f"漏檢 {_pct(res.get('escape_rate'))}、過殺 {_pct(res.get('overkill_rate'))}、"
                       f"需確認 {_pct(res.get('check_rate'))}、量到速度 {_num(res.get('measured_speed'))} mm/s"))
    logger.close()
    limit = core.speed_limit([r for r in results if r.get("speed")], cfg)
    report = _test_report(cfg, results, limit, out)
    (out / "speed_test_report.md").write_text(report, encoding="utf-8")
    (out / "speed_test.json").write_text(json.dumps({"results": results, "max_speed": limit}, ensure_ascii=False,
                                                    indent=2, default=list), encoding="utf-8")
    print("\n" + report)
    print(f"所有輸出在：{out}")


def _pct(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def _num(v) -> str:
    return "—" if v is None else f"{v:.0f}"


def _test_report(cfg, results, limit, out: Path) -> str:
    lines = ["# 速度測試報告", "",
             f"通過標準：漏檢率 ≤ {cfg.max_escape_rate:.1%}、過殺率（含需人工確認）≤ {cfg.max_overkill_rate:.1%}", ""]
    if limit:
        lines.append(f"## ✅ 建議最高速度：{limit:g} mm/s（{limit * 60 / 1000:.1f} m/min）")
    else:
        lines.append("## ❌ 沒有任何速度通過：先看下表的問題欄，從最慢的速度開始排除")
    lines += ["", "| 影片 | 速度 mm/s | 量到速度 | 件數（數到/答案） | 漏檢率 | 過殺率 | 需確認 | 處理 FPS | 問題 / 備註 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r.get("speed") or 0):
        notes = [r["problem"]] if r.get("problem") else []
        notes += r.get("notes", [])
        if r.get("escaped_positions"):
            notes.append(f"漏掉第 {r['escaped_positions']} 件")
        if r.get("check_reasons"):
            notes.append("需確認原因：" + "；".join(r["check_reasons"]))
        if r.get("stray_detections"):
            notes.append(f"工件外誤報 {r['stray_detections']} 次")
        lines.append(f"| {r['video']} | {_num(r.get('speed'))} | {_num(r.get('measured_speed'))} | "
                     f"{r.get('counted', '—')}/{r.get('expected', '—')} | {_pct(r.get('escape_rate'))} | "
                     f"{_pct(r.get('overkill_rate'))} | {_pct(r.get('check_rate'))} | {_num(r.get('proc_fps'))} | "
                     f"{'；'.join(notes)} |")
    chart = _speed_chart(results, out)
    if chart:
        lines += ["", f"![速度 vs 漏檢率 / 過殺率]({chart.name})"]
    lines += ["", "- `parts.csv`：每個工件的判定、幀數、量到的速度", "- `parts/NG`、`parts/CHECK`：不良與需確認工件的影像；"
              "`parts/OK`：抽樣的良品影像。人工確認後可加進 Kaggle 資料集重訓（閉環）"]
    return "\n".join(lines) + "\n"


def _speed_chart(results, out: Path) -> Path | None:
    pts = [(r["speed"], r.get("escape_rate"), r.get("overkill_rate")) for r in results if r.get("valid") and r.get("speed")]
    if not pts:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    pts.sort()
    sp = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sp, [100 * (p[1] or 0) for p in pts], "o-", label="escape rate %")
    ax.plot(sp, [100 * (p[2] or 0) for p in pts], "s-", label="overkill rate %")
    ax.set_xlabel("speed (mm/s)")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend()
    path = out / "speed_vs_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ───────────────────────── live：即時檢測 ─────────────────────────


def _open_source(src):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    return cap if cap.isOpened() else None


def cmd_live(args) -> None:
    cfg = core.load_config(args.config)
    src = args.source if args.source is not None else cfg.source
    detector = YoloDetector(cfg)
    _print_warnings(cfg)
    cap = _open_source(src)
    if cap is None:
        raise core.ConfigError(
            f"打不開相機 {src!r}。iPhone 當 Mac 的攝影機（接續互通相機）時通常是 0 或 1；RTSP 要確認網址和帳密")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.resolution[1])
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    ok, frame = cap.read()
    if not ok:
        raise core.ConfigError(f"相機 {src!r} 讀不到畫面")
    _check_frame_size(cfg, frame.shape[1], frame.shape[0], "相機畫面", args.force)
    out = _run_dir(cfg, "live")
    logger = PartLogger(out, cfg.save_ok_ratio)
    insp = core.Inspector(cfg, detector, detector.thr)
    counts = {v: 0 for v in core.VERDICTS}
    t0, fps_ema, fails, last = time.monotonic(), None, 0, None
    window = not args.no_window
    is_file = isinstance(src, str) and Path(src).is_file()  # 用影片檔當來源時，播完就結束（不要當成斷線重連）
    print("即時檢測中：q 結束、b 重建背景（先清空畫面）、p 暫停")

    def record(results):
        nonlocal last
        for p in results:
            counts[p.verdict] += 1
            logger.save("live", p)
            last = p
            print(("\a" if p.verdict != "OK" else "") + f"#{p.id} {core.VERDICT_ZH[p.verdict]} {p.reason}")

    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
            if not ok:
                if is_file:
                    break
                fails += 1
                if fails % 30 == 0:  # 斷線自動重連
                    print("⚠️ 相機沒有畫面，嘗試重新連線 ...")
                    cap.release()
                    cap = _open_source(src) or cap
                if fails > 300:
                    raise core.ConfigError("相機持續沒有畫面，已停止")
                frame = None
                continue
            fails = 0
            tick = time.monotonic()
            record(insp.feed(frame, tick - t0))
            dt = time.monotonic() - tick
            fps_ema = 1 / max(dt, 1e-6) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
            if window:
                cv2.imshow("camtest live", _overlay(frame, cfg, insp, counts, fps_ema, last))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    insp.reset_background()
                if key == ord("p"):
                    cv2.waitKey(0)
            frame = None
    except KeyboardInterrupt:
        pass
    finally:  # 不論怎麼結束（按 q、播完、斷線停止）都把還在畫面中的工件結案並寫出摘要
        record(insp.finish())
        cap.release()
        logger.close()
        if window:
            cv2.destroyAllWindows()
        total = sum(counts.values())
        summary = {"total": total, **counts, "noise_tracks": insp.noise_tracks,
                   "stray_detections": insp.stray_detections}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"共 {total} 件：良品 {counts['OK']}、不良 {counts['NG']}、需人工確認 {counts['CHECK']}；輸出在 {out}")


def _overlay(frame, cfg, insp, counts, fps, last):
    """OpenCV 只能畫英文，畫面上用英文縮寫；中文訊息印在終端機。"""
    view = frame.copy()
    x1, y1, x2, y2 = cfg.roi_box
    cv2.rectangle(view, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for s, e in insp.last_segments:
        if cfg.motion_axis == "x":
            cv2.rectangle(view, (x1 + s, y1), (x1 + e, y2), (200, 200, 200), 1)
        else:
            cv2.rectangle(view, (x1, y1 + s), (x2, y1 + e), (200, 200, 200), 1)
    for d in insp.last_detections:
        bx1, by1, bx2, by2 = (int(v) for v in d.box)
        color = (40, 40, 220) if d.conf >= insp.thr.get(d.name, 0.25) else (0, 170, 240)
        cv2.rectangle(view, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), color, 2)
        cv2.putText(view, f"{d.name} {d.conf:.2f}", (x1 + bx1, y1 + by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    lines = [f"OK {counts['OK']}  NG {counts['NG']}  CHECK {counts['CHECK']}   proc {fps or 0:.0f} fps"]
    if insp.bg is None and cfg.presence == "background":
        lines.append("BUILDING BACKGROUND - keep the view EMPTY")
    if fps and fps < 0.8 * cfg.fps:
        lines.append(f"SLOW: {fps:.0f} fps < camera {cfg.fps:g} fps -> fewer frames per part")
    if insp.stray_detections > 20:
        lines.append(f"{insp.stray_detections} detections outside parts - check model/thresholds")
    for i, text in enumerate(lines):
        color = (255, 255, 255) if i == 0 else (0, 0, 255)
        cv2.putText(view, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if last is not None:
        banner = f"LAST #{last.id}: {last.verdict}"
        cv2.rectangle(view, (0, view.shape[0] - 50), (view.shape[1], view.shape[0]), COLORS[last.verdict], -1)
        cv2.putText(view, banner, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return view


# ───────────────────────── main ─────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="依設定算出各種速度下能不能檢")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("--speed", type=float, help="產線速度 mm/s（覆蓋設定檔）")
    sp.add_argument("--speeds", help="要列出的速度，逗號分隔，例如 50,100,200,500")
    sp = sub.add_parser("calib", help="放尺量出每像素幾 mm")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("--source", required=True, help="照片、影片，或相機編號")
    sp.add_argument("--distance-mm", type=float, required=True, help="尺上兩點的實際距離")
    sp.add_argument("--points", help="直接給兩點座標 x1,y1,x2,y2（不開視窗點選）")
    sp.add_argument("--write", action="store_true", help="把結果寫回設定檔")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("test", help="錄影速度測試")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("--manifest", required=True, help="CSV：video, speed_mm_s, n_parts, ng_parts")
    sp.add_argument("--force", action="store_true", help="影片解析度和設定不同時仍繼續（不建議）")
    sp = sub.add_parser("live", help="接相機即時檢測")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("--source", help="覆蓋設定檔的 camera.source")
    sp.add_argument("--no-window", action="store_true", help="不開畫面視窗（只在終端機輸出）")
    sp.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    try:
        {"plan": cmd_plan, "calib": cmd_calib, "test": cmd_test, "live": cmd_live}[args.cmd](args)
    except core.ConfigError as e:
        print(f"❌ {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
