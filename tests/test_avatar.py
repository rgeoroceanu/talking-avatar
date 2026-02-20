"""
Tests for engine/avatar.py — run without a GPU or MuseTalk installation.

Tests cover load_idle_frames and scale_frames (pure cv2/numpy), which are
importable without musetalk because all musetalk imports are lazy in avatar.py.
"""

import numpy as np
import pytest


def _make_frames(count: int, h: int, w: int) -> list:
    """Create a list of solid black BGR frames."""
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(count)]


# ── scale_frames ─────────────────────────────────────────────────────────────

def test_scale_frames_noop_when_within_max():
    """Frames smaller than max_dim are returned unchanged (same list object)."""
    from engine.avatar import scale_frames

    frames = _make_frames(3, 100, 200)
    result = scale_frames(frames, max_dim=300)

    assert result is frames
    assert result[0].shape == (100, 200, 3)


def test_scale_frames_by_longest_edge():
    """200×400 frame (h=200, w=400) with max_dim=200 → shape (100, 200, 3)."""
    from engine.avatar import scale_frames

    frames = _make_frames(1, h=200, w=400)
    result = scale_frames(frames, max_dim=200)

    assert result[0].shape == (100, 200, 3)


def test_scale_frames_preserves_count():
    """All 5 input frames are returned after scaling."""
    from engine.avatar import scale_frames

    frames = _make_frames(5, h=400, w=400)
    result = scale_frames(frames, max_dim=200)

    assert len(result) == 5


# ── load_idle_frames ──────────────────────────────────────────────────────────

def test_load_idle_frames_raises_missing():
    """RuntimeError is raised when the video file does not exist."""
    from engine.avatar import load_idle_frames

    with pytest.raises(RuntimeError, match="not found"):
        load_idle_frames("/nonexistent/path/idle.mp4")


def test_load_idle_frames_raises_empty_video(tmp_path):
    """RuntimeError is raised when the file exists but yields zero frames."""
    import cv2
    from engine.avatar import load_idle_frames

    dummy_path = str(tmp_path / "dummy.mp4")
    with open(dummy_path, "wb") as f:
        f.write(b"\x00" * 128)  # invalid video data

    with pytest.raises(RuntimeError, match="No frames loaded"):
        load_idle_frames(dummy_path)


def test_load_idle_frames_reads_real_video(tmp_path):
    """3-frame MJPEG AVI written with cv2.VideoWriter is read back as 3 frames."""
    import cv2
    from engine.avatar import load_idle_frames

    video_path = str(tmp_path / "test.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, 25, (64, 64))
    for _ in range(3):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = load_idle_frames(video_path)
    assert len(frames) == 3
