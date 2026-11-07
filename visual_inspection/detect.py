"""Detect vehicles, people and animals in a camera stream, a video or images.

Examples:
    python -m visual_inspection.detect --source 0                                # webcam
    python -m visual_inspection.detect --source http://192.168.1.20:8080/video   # phone camera app
    python -m visual_inspection.detect --source "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102" --track
    python -m visual_inspection.detect --source clip.mp4 --save --log detections.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from visual_inspection.categories import CATEGORIES, COLORS, category_of, class_ids_for

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
LOG_FIELDS = ["time", "source", "frame", "category", "class", "confidence", "x1", "y1", "x2", "y2", "track_id"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "visual-inspection (q to quit)"


@dataclass
class Detection:
    category: str
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass
class Frame:
    image: np.ndarray
    index: int
    name: str = ""  # file name when the source is images; empty for video and cameras


def parse_source(source: str) -> int | str:
    """Camera indices are given as digits ("0"); anything else is a path or URL."""
    return int(source) if source.isdigit() else source


def is_live(source: int | str) -> bool:
    return isinstance(source, int) or source.lower().startswith(LIVE_PREFIXES)


def image_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []


def video_fps(path: str | Path, default: float = 25.0) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


class LatestFrameReader:
    """Reads a live camera in the background and keeps only the newest frame.

    Without this, a model slower than the camera falls further and further behind
    and the window shows the scene from minutes ago. A dropped connection (common
    with CCTV over Wi-Fi) is reopened instead of ending the run.
    """

    def __init__(self, source: int | str, reconnect_delay: float = 2.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {source}")
        self._frame = None
        self._seq = 0
        self._stopped = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok:
                print(f"Lost camera {self.source}, reconnecting...", file=sys.stderr)
                self._cap.release()
                time.sleep(self.reconnect_delay)
                self._cap = cv2.VideoCapture(self.source)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()
        self._cap.release()

    def read(self, last_seq: int, timeout: float = 10.0) -> tuple[int, np.ndarray | None]:
        """Wait for a frame newer than last_seq. Returns (seq, frame), or (last_seq, None) on timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or self._stopped, timeout)
            if self._seq > last_seq:
                return self._seq, self._frame
            return last_seq, None

    def close(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


def live_frames(source: int | str) -> Iterator[Frame]:
    reader = LatestFrameReader(source)
    seq = 0
    try:
        while True:
            seq, image = reader.read(seq)
            if image is None:
                print(f"No frames from {source} yet, still waiting...", file=sys.stderr)
                continue
            yield Frame(image, seq)
    finally:
        reader.close()


def iter_frames(source: int | str, stride: int = 1) -> Iterator[Frame]:
    """Yield frames from an image, a folder of images, a video file or a live camera."""
    if is_live(source):
        yield from live_frames(source)
        return

    path = Path(source)
    if not path.exists():
        raise RuntimeError(f"Not found: {source}")
    images = image_files(path)
    if images or path.is_dir():
        for i, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image {image_path}", file=sys.stderr)
                continue
            yield Frame(image, i, image_path.name)
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    try:
        i = 0
        while cap.grab():
            if i % stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                yield Frame(image, i)
            i += 1
    finally:
        cap.release()


def extract_detections(result) -> list[Detection]:
    """Turn an Ultralytics result into Detections, dropping classes outside our categories."""
    boxes = result.boxes
    track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes.cls)
    detections = []
    for xyxy, cls, conf, track_id in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), track_ids):
        name = result.names[int(cls)]
        category = category_of(name)
        if category is None:
            continue
        detections.append(
            Detection(
                category=category,
                class_name=name,
                confidence=float(conf),
                box=tuple(int(v) for v in xyxy),
                track_id=None if track_id is None else int(track_id),
            )
        )
    return detections


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes and per-category counts onto the image in place, and return it."""
    for d in detections:
        color = COLORS[d.category]
        x1, y1, x2, y2 = d.box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        (w, h), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y1, h + 6)  # keep the label on screen for boxes touching the top edge
        cv2.rectangle(image, (x1, top - h - 6), (x1 + w + 4, top), color, -1)
        cv2.putText(image, label, (x1 + 2, top - 4), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    counts = Counter(d.category for d in detections)
    summary = "  ".join(f"{c}: {counts[c]}" for c in CATEGORIES)
    cv2.putText(image, summary, (10, 28), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, summary, (10, 28), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def output_path(source: int | str, frame: Frame, save_dir: Path) -> Path:
    if frame.name:
        return save_dir / frame.name
    if is_live(source):
        return save_dir / f"camera_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    return save_dir / f"{Path(source).stem}_detected.mp4"


def can_show() -> bool:
    # cv2.imshow needs a display; Linux servers and containers usually have none.
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect vehicles, people and animals in camera streams, videos or images.")
    p.add_argument("--source", required=True, help="camera index (0), RTSP/HTTP URL, video file, image, or image folder")
    p.add_argument("--model", default="yolo26n.pt", help="YOLO weights; pretrained ones download automatically (default: %(default)s)")
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES), help="categories to detect (default: all)")
    p.add_argument("--conf", type=float, default=0.4, help="minimum confidence (default: %(default)s)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size (default: %(default)s)")
    p.add_argument("--device", default=None, help="'cpu', '0' for the first NVIDIA GPU, 'mps' for Apple silicon (default: auto)")
    p.add_argument("--track", action="store_true", help="follow objects across frames so each one is counted once")
    p.add_argument("--stride", type=int, default=1, help="for video files, process every Nth frame (default: %(default)s)")
    p.add_argument("--save", action="store_true", help="save annotated images or video to --save-dir")
    p.add_argument("--save-dir", default="outputs", help="where --save writes (default: %(default)s)")
    p.add_argument("--save-fps", type=float, default=10.0, help="frame rate for videos recorded from live cameras (default: %(default)s)")
    p.add_argument("--log", metavar="CSV", help="append every detection to this CSV file")
    p.add_argument("--no-show", dest="show", action="store_false", help="don't open a preview window")
    args = p.parse_args(argv)
    args.stride = max(1, args.stride)
    return args


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO  # imported here so the helpers above work without PyTorch installed

    source = parse_source(args.source)
    model = YOLO(args.model)
    classes = class_ids_for(model.names, args.categories)
    if not classes:
        sys.exit(f"{args.model} has no classes in categories {args.categories}; see visual_inspection/categories.py")

    show = args.show and can_show()
    if args.show and not show:
        print("No display found, running without a preview window.", file=sys.stderr)
    single_images = not is_live(source) and bool(image_files(Path(source)) or Path(source).is_dir())
    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_writer = None
    if args.log:
        log_path = Path(args.log)
        new_log = not log_path.exists() or log_path.stat().st_size == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if new_log:
            log_writer.writerow(LOG_FIELDS)

    infer = model.track if args.track else model.predict
    options = dict(classes=classes, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
    if args.track:
        options["persist"] = True

    video_writer = None
    tracked = {c: set() for c in CATEGORIES}
    most_at_once = Counter()
    frames = 0
    started = time.time()
    try:
        for frame in iter_frames(source, args.stride):
            detections = extract_detections(infer(frame.image, **options)[0])
            frames += 1
            for category, n in Counter(d.category for d in detections).items():
                most_at_once[category] = max(most_at_once[category], n)
            for d in detections:
                if d.track_id is not None:
                    tracked[d.category].add(d.track_id)

            if log_writer:
                now = datetime.now().isoformat(timespec="seconds")
                for d in detections:
                    log_writer.writerow(
                        [now, args.source, frame.name or frame.index, d.category, d.class_name,
                         f"{d.confidence:.3f}", *d.box, "" if d.track_id is None else d.track_id]
                    )

            image = draw(frame.image, detections)
            if args.save:
                if single_images:
                    cv2.imwrite(str(output_path(source, frame, save_dir)), image)
                else:
                    if video_writer is None:
                        fps = args.save_fps if is_live(source) else video_fps(source) / args.stride
                        path = output_path(source, frame, save_dir)
                        h, w = image.shape[:2]
                        video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        print(f"Recording to {path}")
                    video_writer.write(image)

            if show:
                cv2.imshow(WINDOW, image)
                # Images wait for a key press before moving on; video keeps playing.
                if cv2.waitKey(0 if single_images else 1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if video_writer is not None:
            video_writer.release()
        if log_file is not None:
            log_file.close()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    print(f"Processed {frames} frames in {elapsed:.1f}s ({frames / elapsed if elapsed else 0:.1f} FPS)")
    for category in args.categories:
        line = f"  {category}: up to {most_at_once[category]} at once"
        if args.track:
            line += f", {len(tracked[category])} distinct"
        print(line)


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except RuntimeError as e:  # bad source: show the message, not a traceback
        sys.exit(str(e))


if __name__ == "__main__":
    main()
