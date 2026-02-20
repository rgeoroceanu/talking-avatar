# MuseTalk Live

Real-time talking head powered by [MuseTalk](https://github.com/TMElyralab/MuseTalk), running on a single GPU.

## Demo

<!-- Add a screenshot or GIF here -->

## Features

- Real-time WebSocket streaming of JPEG frames at 25 fps
- Text-to-speech via [edge-tts](https://github.com/rany2/edge-tts) (no API key required)
- fp16 VRAM management — runs on a single GTX 1650 (4 GB) or L4 (24 GB)
- One-command Cloud Run deployment via `./deployment/deploy.sh`

## Architecture

```
Text input
  → edge-tts (MP3)
  → ffmpeg (WAV 16 kHz mono)
  → Whisper-tiny (audio features)
  → MuseTalk UNet fp16 (per-frame latents)
  → SD-VAE decode (256×256 face crops)
  → feathered face blend into idle frame
  → JPEG encode
  → WebSocket stream to browser
```

## Requirements

- NVIDIA GPU with ≥ 4 GB VRAM (tested on GTX 1650 and L4)
- Docker
- `gcloud` CLI (for Cloud Run deployment)

## Quick Start

### Local (Docker Compose)

1. Provide your loopable avatar video:
   ```bash
   cp /path/to/your/video.mp4 examples/idle.mp4
   ```

2. Build and start:
   ```bash
   docker compose -f deployment/docker-compose.yml up --build
   ```

3. Open http://localhost:8080

### Cloud Run

```bash
./deployment/deploy.sh
```

### Local dev (no GPU — for frontend iteration)

```bash
cp examples/idle.mp4 /tmp/idle_test.mp4
FFMPEG_PATH=/path/to/your/ffmpeg uvicorn test_server:app --port 8080 --reload
```

Then open http://localhost:8080. No GPU is needed; the mock server returns still frames with real TTS audio.

## Providing your avatar

`idle.mp4` is user-supplied and is **not tracked in git**. Any loopable talking-head video works — a short clip of a person speaking or at rest, exported as H.264 MP4.

The video is baked into the Docker image at build time, so no volume mount is needed at runtime.

## Project structure

```
musetalk-live/
├── engine/
│   ├── __init__.py       # Public API: MuseTalkEngine, EngineConfig
│   ├── models.py         # Model loading (VAE, UNet, Whisper, PE)
│   ├── avatar.py         # Idle video I/O, face detection, VAE encoding
│   ├── inference.py      # Forward pass, blending, MuseTalkEngine coordinator
│   └── tts.py            # Text → MP3 (edge-tts) → WAV (ffmpeg)
├── tests/
│   ├── test_tts.py
│   ├── test_avatar.py
│   └── test_inference.py
├── static/
│   └── index.html
├── server.py             # FastAPI app: HTTP endpoints + WebSocket handler
├── test_server.py        # Mock server for frontend testing (no GPU)
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── deploy.sh
    └── start.sh
```

## Credits

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — TMElyralab
- [edge-tts](https://github.com/rany2/edge-tts) — TTS without an API key
- HuggingFace models: [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse), [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny), [`yzd-v/DWPose`](https://huggingface.co/yzd-v/DWPose)

## License

MIT — see [LICENSE](LICENSE) for details.

> **Note:** MuseTalk has its own license. Review the [MuseTalk repository](https://github.com/TMElyralab/MuseTalk) before any commercial use.
