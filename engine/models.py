"""
engine/models.py — Load all MuseTalk model weights to GPU in fp16.
"""

import dataclasses
import gc
import json
import os

import torch
from diffusers import UNet2DConditionModel
from transformers import WhisperModel

from musetalk.models.unet import PositionalEncoding
from musetalk.models.vae import VAE
from musetalk.utils.audio_processor import AudioProcessor


@dataclasses.dataclass(frozen=True)
class ModelBundle:
    vae: VAE
    unet: UNet2DConditionModel
    pe: PositionalEncoding
    whisper: WhisperModel
    audio_processor: AudioProcessor
    timesteps: torch.Tensor
    device: torch.device
    weight_dtype: torch.dtype


def load_models(model_dir: str, device: torch.device) -> ModelBundle:
    """Blocking. Loads VAE, UNet, Whisper, PE, AudioProcessor to device in fp16."""
    weight_dtype = torch.float16

    # VAE
    print("Loading VAE...")
    vae = VAE(model_path=os.path.join(model_dir, "sd-vae"))
    vae.vae = vae.vae.half()
    vae._use_float16 = True

    # UNet
    print("Loading UNet...")
    unet_config_path = os.path.join(model_dir, "musetalkV15", "musetalk.json")
    unet_weights_path = os.path.join(model_dir, "musetalkV15", "unet.pth")

    with open(unet_config_path, "r") as f:
        unet_config = json.load(f)
    unet = UNet2DConditionModel(**unet_config)
    weights = torch.load(unet_weights_path, map_location=device)
    unet.load_state_dict(weights)
    del weights
    gc.collect()
    unet = unet.half().to(device).eval()
    unet.requires_grad_(False)

    # Positional encoding
    pe = PositionalEncoding(d_model=384).half().to(device)

    # Fixed timestep
    timesteps = torch.tensor([0], device=device)

    # AudioProcessor (CPU-based feature extraction)
    audio_processor = AudioProcessor(
        feature_extractor_path=os.path.join(model_dir, "whisper")
    )

    # Whisper
    print("Loading Whisper...")
    whisper = WhisperModel.from_pretrained(os.path.join(model_dir, "whisper"))
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    torch.cuda.empty_cache()

    return ModelBundle(
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        audio_processor=audio_processor,
        timesteps=timesteps,
        device=device,
        weight_dtype=weight_dtype,
    )
