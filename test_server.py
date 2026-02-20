#!/usr/bin/env python3
"""
test_server.py — Local mock server for frontend testing (no GPU required).

Exercises all three frontend features:
  1. Audio playback (real edge-tts TTS)
  2. Idle video loop (/idle.mp4 served from /tmp/idle_test.mp4)
  3. Large responsive avatar container (reports width=480, height=640)

Usage:
  # 1. Place your idle video at /tmp/idle_test.mp4:
  cp examples/idle.mp4 /tmp/idle_test.mp4

  # 2. Start mock server (needs: edge-tts, opencv, fastapi, uvicorn):
  FFMPEG_PATH=/path/to/your/ffmpeg \\
    uvicorn test_server:app --port 8080 --reload

  # 3. Open http://localhost:8080 and verify:
  #    - idle video loops with blinking
  #    - on send: avatar stills play + audio heard
  #    - box is large (480px wide in mock)
"""

import asyncio
import base64
import json
import os
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine.tts import tts_to_wav as _tts_to_wav

IDLE_PATH = "/tmp/idle_test.mp4"
AVATAR_PATH = os.path.join(os.path.dirname(__file__), "avatar.jpeg")
STATIC_DIR  = os.path.join(os.path.dirname(__file__), "static")
VOICE = "en-US-GuyNeural"
MOCK_W, MOCK_H = 480, 640
MOCK_FPS = 12
MOCK_FRAMES = 30  # ~2.5 s of still frames


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG_PATH", "ffmpeg")


async def tts_to_wav(text: str) -> tuple:
    return await _tts_to_wav(text, voice=VOICE, ffmpeg_bin=_ffmpeg())


def _build_mock_frames() -> list:
    """Load avatar.jpeg, resize to MOCK_W x MOCK_H, return MOCK_FRAMES JPEG bytes."""
    img = cv2.imread(AVATAR_PATH)
    if img is None:
        # Fallback: solid grey frame
        img = np.full((MOCK_H, MOCK_W, 3), 128, dtype=np.uint8)
    img = cv2.resize(img, (MOCK_W, MOCK_H), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_bytes = buf.tobytes()
    return [frame_bytes] * MOCK_FRAMES


# ── FastAPI ──────────────────────────────────────────────────

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/idle.mp4")
async def idle_video():
    if not os.path.exists(IDLE_PATH):
        raise HTTPException(
            status_code=404,
            detail=(
                f"idle.mp4 not found at {IDLE_PATH}. "
                "Copy your idle video: cp examples/idle.mp4 /tmp/idle_test.mp4"
            ),
        )
    return FileResponse(IDLE_PATH, media_type="video/mp4")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WS connected")
    closed = False

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            text = msg.get("text", "").strip()
            if not text:
                continue

            print(f"Received: {text!r}")
            await ws.send_json({"type": "status", "message": "processing"})

            try:
                wav_path, mp3_bytes = await tts_to_wav(text)
                try:
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None, _build_mock_frames
                    )
                finally:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

                # Send audio
                audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
                await ws.send_json({"type": "audio", "data": audio_b64})

                # Send chunk_start with mock dimensions
                await ws.send_json({
                    "type": "chunk_start",
                    "frame_count": len(frames),
                    "fps": MOCK_FPS,
                    "width": MOCK_W,
                    "height": MOCK_H,
                })

                # Send frames
                for i, fb in enumerate(frames):
                    b64 = base64.b64encode(fb).decode("ascii")
                    await ws.send_json(
                        {"type": "frame", "data": b64, "index": i, "total": len(frames)}
                    )

                await ws.send_json({"type": "chunk_end"})

            except Exception:
                traceback.print_exc()
                try:
                    await ws.send_json({"type": "error", "message": "mock inference failed"})
                except Exception:
                    break

    except WebSocketDisconnect:
        closed = True

    print("WS disconnected")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
