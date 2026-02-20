"""
engine/avatar.py — Idle video I/O, frame scaling, face detection, VAE encoding.

All musetalk imports are lazy (inside preprocess_avatar) so this module can be
imported in test environments without MuseTalk installed.
"""

from __future__ import annotations

import dataclasses
import gc
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .models import ModelBundle


@dataclasses.dataclass
class AvatarData:
    frame_list: list          # np.ndarray per idle frame (BGR uint8, scaled)
    coord_list: list          # (x1, y1, x2, y2) face bbox per frame
    input_latent_list: list   # torch tensors, one per idle frame
    mask_list: list           # np.ndarray blending mask per frame
    mask_coords_list: list    # crop_box per frame
    frame_w: int              # width of scaled idle frames
    frame_h: int              # height of scaled idle frames


def load_idle_frames(idle_path: str) -> list:
    """Load all frames from an idle video file.

    Raises RuntimeError if file is missing or yields zero frames.
    """
    if not os.path.exists(idle_path):
        raise RuntimeError(f"idle.mp4 not found at {idle_path}")
    cap = cv2.VideoCapture(idle_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {idle_path}")
    return frames


def scale_frames(frames: list, max_dim: int) -> list:
    """Downscale so longest edge <= max_dim. Returns same list if already small."""
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frames
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return [
        cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        for f in frames
    ]


def preprocess_avatar(idle_path: str, models: ModelBundle, max_dim: int) -> AvatarData:
    """Preprocess idle video: face detection + VAE encoding.

    Lazy-imports musetalk.utils.preprocessing so this module can be imported
    without MuseTalk on the path (e.g. in unit tests).
    """
    # Lazy imports — musetalk loads mmpose/xtcocotools at import time
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder  # noqa: PLC0415
    from musetalk.utils.blending import get_image_prepare_material  # noqa: PLC0415
    from musetalk.utils.face_parsing import FaceParsing  # noqa: PLC0415

    print("Preprocessing avatar...")

    raw_frames = load_idle_frames(idle_path)
    print(f"  Loaded {len(raw_frames)} idle frames")

    h0, w0 = raw_frames[0].shape[:2]
    scaled_frames = scale_frames(raw_frames, max_dim)
    h1, w1 = scaled_frames[0].shape[:2]
    if (h1, w1) != (h0, w0):
        print(f"  Scaled frames: {w0}x{h0} -> {w1}x{h1}")
    else:
        print(f"  Frame resolution: {w0}x{h0} (within {max_dim} limit)")

    frame_h, frame_w = scaled_frames[0].shape[:2]

    # Face detection on first frame only — face position is stable across idle frames
    first_frame = scaled_frames[0]
    detect_path = "/tmp/idle_frame0.jpeg"
    cv2.imwrite(detect_path, first_frame)

    coord_list, _ = get_landmark_and_bbox([detect_path], upperbondrange=0)

    if not coord_list or coord_list[0] == coord_placeholder:
        raise RuntimeError("No face detected in idle video first frame")

    # Extend bottom margin for V1.5
    x1, y1, x2, y2 = coord_list[0]
    y2 = min(y2 + 10, first_frame.shape[0])
    bbox = (x1, y1, x2, y2)
    print(f"  Face bbox (first frame): {bbox}")

    # Encode all idle frames to VAE latents
    print(f"  Encoding {len(scaled_frames)} frames to VAE latents ...")
    input_latent_list = []
    for i, frame in enumerate(scaled_frames):
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = models.vae.get_latents_for_unet(resized)
        input_latent_list.append(latents)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(scaled_frames)} encoded")

    # Prepare blending mask once from first frame (BiSeNet is expensive)
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    mask, crop_box = get_image_prepare_material(
        first_frame,
        [x1, y1, x2, y2],
        upper_boundary_ratio=0.0,
        expand=1.5,
        fp=fp,
        mode="raw",
    )
    del fp
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Avatar cached: {len(input_latent_list)} idle frames")

    return AvatarData(
        frame_list=scaled_frames,
        coord_list=[bbox] * len(scaled_frames),
        input_latent_list=input_latent_list,
        mask_list=[mask] * len(scaled_frames),
        mask_coords_list=[crop_box] * len(scaled_frames),
        frame_w=frame_w,
        frame_h=frame_h,
    )
