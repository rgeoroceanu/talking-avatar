"""
engine/inference.py — UNet forward pass, blending loop, warmup, and MuseTalkEngine.

All musetalk imports are lazy (inside function bodies) so this module can be
imported in test environments without MuseTalk installed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .avatar import AvatarData
    from .models import ModelBundle


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    frame_skip: int = 2
    max_output_dim: int = 1280
    jpeg_quality: int = 70
    batch_size: int = 8
    fps: int = 25


def compute_feathered_mask(h: int, w: int, feather: int) -> np.ndarray:
    """Pure numpy. Linear-ramp alpha mask of shape (h, w) float32, values in [0, 1]."""
    mask = np.ones((h, w), dtype=np.float32)
    for fi in range(feather):
        alpha = (fi + 1) / feather
        mask[fi, :] = np.minimum(mask[fi, :], alpha)
        mask[h - 1 - fi, :] = np.minimum(mask[h - 1 - fi, :], alpha)
        mask[:, fi] = np.minimum(mask[:, fi], alpha)
        mask[:, w - 1 - fi] = np.minimum(mask[:, w - 1 - fi], alpha)
    return mask


@torch.inference_mode()
def run_inference(
    wav_path: str,
    models: ModelBundle,
    avatar: AvatarData,
    config: EngineConfig,
) -> tuple:
    """Synchronous inference: audio → JPEG frames. Must be called from a thread."""
    from musetalk.utils.utils import datagen  # noqa: PLC0415

    t0 = time.time()

    whisper_input_features, librosa_length = models.audio_processor.get_audio_feature(
        wav_path, weight_dtype=models.weight_dtype
    )
    t1 = time.time()

    whisper_chunks = models.audio_processor.get_whisper_chunk(
        whisper_input_features,
        models.device,
        models.weight_dtype,
        models.whisper,
        librosa_length,
        fps=config.fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )
    t2 = time.time()

    orig_count = len(whisper_chunks)
    if config.frame_skip > 1:
        whisper_chunks = whisper_chunks[::config.frame_skip]
    n_frames = len(whisper_chunks)

    print(f"  frames: {orig_count} -> {n_frames} | "
          f"audio_feat {t1-t0:.2f}s, whisper_chunk {t2-t1:.2f}s")

    gen = datagen(
        whisper_chunks,
        avatar.input_latent_list,
        batch_size=config.batch_size,
        delay_frame=0,
        device=str(models.device),
    )

    jpeg_frames = []
    cycle_len = len(avatar.coord_list)
    frame_idx = 0

    unet_total = 0.0
    vae_total = 0.0
    blend_total = 0.0

    for whisper_batch, latent_batch in gen:
        audio_feature_batch = models.pe(whisper_batch.to(models.device))
        latent_batch = latent_batch.to(device=models.device, dtype=models.weight_dtype)

        torch.cuda.synchronize()
        t_unet = time.time()
        pred_latents = models.unet(
            latent_batch,
            models.timesteps,
            encoder_hidden_states=audio_feature_batch,
        ).sample
        torch.cuda.synchronize()
        unet_total += time.time() - t_unet

        t_vae = time.time()
        pred_latents = pred_latents.to(device=models.device, dtype=models.vae.vae.dtype)
        recon_frames = models.vae.decode_latents(pred_latents)
        torch.cuda.synchronize()
        vae_total += time.time() - t_vae

        t_blend = time.time()
        for res_frame in recon_frames:
            idx = frame_idx % cycle_len
            bbox = avatar.coord_list[idx]
            ori_frame = avatar.frame_list[idx].copy()
            x1, y1, x2, y2 = bbox

            h, w = y2 - y1, x2 - x1
            res_frame = cv2.resize(res_frame.astype(np.uint8), (w, h))

            feather = max(1, min(h, w) // 6)
            mask = compute_feathered_mask(h, w, feather)
            mask_3d = mask[:, :, np.newaxis]
            ori_crop = ori_frame[y1:y2, x1:x2].astype(np.float32)
            blended = res_frame.astype(np.float32) * mask_3d + ori_crop * (1 - mask_3d)
            ori_frame[y1:y2, x1:x2] = blended.astype(np.uint8)

            _, buf = cv2.imencode(
                ".jpg", ori_frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            )
            jpeg_frames.append(buf.tobytes())
            frame_idx += 1
        blend_total += time.time() - t_blend

    t3 = time.time()
    print(f"  UNet {unet_total:.2f}s | VAE {vae_total:.2f}s | blend {blend_total:.2f}s | "
          f"TOTAL {t3-t0:.2f}s ({len(jpeg_frames)} frames, "
          f"{(t3-t2)/max(1,len(jpeg_frames)):.2f}s/frame)")

    return jpeg_frames, avatar.frame_w, avatar.frame_h


@torch.inference_mode()
def warmup(models: ModelBundle) -> None:
    """3 dummy forward passes to compile CUDA kernels at startup."""
    print("CUDA warmup (compiling kernels)...")
    t0 = time.time()
    dummy_latent = torch.randn(1, 8, 32, 32, device=models.device, dtype=models.weight_dtype)
    dummy_audio = torch.randn(1, 50, 384, device=models.device, dtype=models.weight_dtype)
    audio_feat = models.pe(dummy_audio)
    for _ in range(3):
        pred = models.unet(
            dummy_latent, models.timesteps, encoder_hidden_states=audio_feat
        ).sample
        pred = pred.to(dtype=models.vae.vae.dtype)
        models.vae.decode_latents(pred)
    torch.cuda.synchronize()
    print(f"CUDA warmup done in {time.time()-t0:.1f}s")


class MuseTalkEngine:
    """Thin coordinator: owns asyncio.Lock + thread-executor wrapper."""

    def __init__(self, model_dir: str, idle_path: str, config: EngineConfig) -> None:
        self._model_dir = model_dir
        self._idle_path = idle_path
        self._config = config
        self._models = None
        self._avatar = None
        self._lock = asyncio.Lock()

    def initialize(self) -> None:
        """Blocking. Call once from startup event via run_in_executor."""
        from .models import load_models  # noqa: PLC0415
        from .avatar import preprocess_avatar  # noqa: PLC0415

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Initializing MuseTalk engine on {device}")
        self._models = load_models(self._model_dir, device)
        self._avatar = preprocess_avatar(self._idle_path, self._models, self._config.max_output_dim)
        warmup(self._models)
        print("Engine ready")

    @property
    def ready(self) -> bool:
        return self._avatar is not None

    @property
    def avatar(self) -> AvatarData | None:
        return self._avatar

    async def process_audio_chunk(self, wav_path: str) -> tuple:
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, run_inference, wav_path, self._models, self._avatar, self._config
            )
