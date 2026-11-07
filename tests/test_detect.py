from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from visual_inspection.categories import ANIMAL, PERSON, VEHICLE
from visual_inspection.detect import (
    Detection,
    Frame,
    LatestFrameReader,
    draw,
    extract_detections,
    is_live,
    iter_frames,
    output_path,
    parse_args,
    parse_source,
)


def fake_result(rows, track_ids=None):
    """Mimic the parts of an Ultralytics result that extract_detections reads."""
    rows = np.array(rows, dtype=float).reshape(-1, 6)  # x1, y1, x2, y2, conf, cls
    boxes = SimpleNamespace(
        xyxy=rows[:, :4],
        conf=rows[:, 4],
        cls=rows[:, 5],
        id=None if track_ids is None else np.array(track_ids, dtype=float),
    )
    return SimpleNamespace(boxes=boxes, names={0: "person", 2: "car", 16: "dog", 56: "chair"})


def write_video(path: Path, n_frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("clip.mp4") == "clip.mp4"
    assert parse_source("rtsp://10.0.0.2/stream") == "rtsp://10.0.0.2/stream"


def test_is_live():
    assert is_live(0)
    assert is_live("rtsp://10.0.0.2/stream")
    assert is_live("HTTP://192.168.1.20:8080/video")
    assert not is_live("clip.mp4")


def test_extract_detections_maps_categories_and_drops_other_classes():
    result = fake_result([[10, 20, 30, 40, 0.9, 0], [1, 2, 3, 4, 0.8, 2], [5, 5, 9, 9, 0.7, 56], [0, 0, 8, 8, 0.6, 16]])
    detections = extract_detections(result)
    assert [(d.category, d.class_name) for d in detections] == [(PERSON, "person"), (VEHICLE, "car"), (ANIMAL, "dog")]
    assert detections[0].box == (10, 20, 30, 40)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].track_id is None


def test_extract_detections_keeps_track_ids():
    detections = extract_detections(fake_result([[0, 0, 5, 5, 0.9, 2], [1, 1, 6, 6, 0.8, 0]], track_ids=[3, 7]))
    assert [d.track_id for d in detections] == [3, 7]


def test_extract_detections_handles_empty_result():
    assert extract_detections(fake_result([])) == []


def test_draw_marks_boxes_including_ones_at_the_top_edge():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = [Detection(PERSON, "person", 0.9, (50, 0, 120, 90), track_id=4), Detection(ANIMAL, "dog", 0.5, (10, 60, 40, 95))]
    out = draw(image, detections)
    assert out is image
    assert image.any()
    assert tuple(image[70, 120]) == (0, 200, 0)  # right edge of the person box


def test_iter_frames_reads_image_folder_in_name_order(tmp_path):
    for name in ["b.jpg", "a.png"]:
        cv2.imwrite(str(tmp_path / name), np.zeros((8, 8, 3), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    frames = list(iter_frames(str(tmp_path)))
    assert [f.name for f in frames] == ["a.png", "b.jpg"]


def test_iter_frames_reads_single_image(tmp_path):
    path = tmp_path / "cam.jpg"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    frames = list(iter_frames(str(path)))
    assert len(frames) == 1 and frames[0].name == "cam.jpg"


def test_iter_frames_reads_video_with_stride(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path, 10)
    assert [f.index for f in iter_frames(str(path))] == list(range(10))
    assert [f.index for f in iter_frames(str(path), stride=3)] == [0, 3, 6, 9]


@pytest.mark.parametrize("name", ["missing.mp4", "missing.jpg"])
def test_iter_frames_rejects_missing_file(tmp_path, name):
    with pytest.raises(RuntimeError, match="Not found"):
        list(iter_frames(str(tmp_path / name)))


def test_iter_frames_rejects_unreadable_video(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="Cannot open video"):
        list(iter_frames(str(path)))


def test_latest_frame_reader_keeps_delivering_after_the_stream_drops(tmp_path):
    # A short video file stands in for a camera: when it ends the reader treats it as a
    # dropped connection, reopens it, and frames keep arriving.
    path = tmp_path / "clip.mp4"
    write_video(path, 3)
    reader = LatestFrameReader(str(path), reconnect_delay=0.01)
    try:
        seq = 0
        for _ in range(6):
            new_seq, image = reader.read(seq, timeout=5)
            assert image is not None and new_seq > seq
            seq = new_seq
    finally:
        reader.close()


def test_latest_frame_reader_rejects_unopenable_source(tmp_path):
    with pytest.raises(RuntimeError):
        LatestFrameReader(str(tmp_path / "missing.mp4"))


def test_output_path(tmp_path):
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    assert output_path("photos", Frame(image, 0, "a.jpg"), tmp_path) == tmp_path / "a.jpg"
    assert output_path("videos/gate.mp4", Frame(image, 5), tmp_path) == tmp_path / "gate_detected.mp4"
    assert output_path(0, Frame(image, 5), tmp_path).name.startswith("camera_")


def test_parse_args_defaults_and_stride_floor():
    args = parse_args(["--source", "0", "--stride", "0"])
    assert args.categories == [VEHICLE, PERSON, ANIMAL]
    assert args.stride == 1
    assert args.show is True
    assert parse_args(["--source", "0", "--no-show", "--categories", "animal"]).categories == [ANIMAL]
