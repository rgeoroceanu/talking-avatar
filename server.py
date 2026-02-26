import base64
import json
import os
import sys
import asyncio
import traceback
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add MuseTalk to Python path; add /app so `import engine` works from /app/MuseTalk
sys.path.insert(0, "/app/MuseTalk")
sys.path.insert(0, "/app")

from engine import MuseTalkEngine, EngineConfig
from engine.tts import tts_to_wav

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOICE = "en-US-GuyNeural"
DATA_PATH: str = os.environ.get("DATA_PATH", "./data")
MAX_CACHED_AVATARS: int = int(os.environ.get("MAX_CACHED_AVATARS", "3"))

config = EngineConfig(frame_skip=2, max_output_dim=1280, jpeg_quality=70, batch_size=8, fps=25)

# ---------------------------------------------------------------------------
# Shared model bundle — loaded once, reused across all avatar engines.
# The models (VAE, UNet, Whisper, etc.) are heavy; we never want to reload
# them.  Each avatar engine only re-runs the fast avatar-preprocessing step.
# ---------------------------------------------------------------------------

_shared_models = None  # type: ignore[assignment]  # populated in startup()

# ---------------------------------------------------------------------------
# LRU cache for MuseTalkEngine instances (one per avatar_id).
# OrderedDict keeps insertion/access order; evict the LRU entry when full.
# ---------------------------------------------------------------------------

_engine_cache: OrderedDict[str, MuseTalkEngine] = OrderedDict()
_cache_lock = asyncio.Lock()


async def _get_or_create_engine(avatar_id: str) -> MuseTalkEngine:
    """Return a ready MuseTalkEngine for avatar_id, using the LRU cache.

    If the engine is not cached yet it is created, initialized (in a thread
    executor so the event loop stays free), and inserted into the cache.
    The least-recently-used entry is evicted when the cache is full.
    """
    async with _cache_lock:
        if avatar_id in _engine_cache:
            # Move to end (most recently used)
            _engine_cache.move_to_end(avatar_id)
            return _engine_cache[avatar_id]

        # Evict the LRU entry if the cache is full
        while len(_engine_cache) >= MAX_CACHED_AVATARS:
            evicted_id, _ = _engine_cache.popitem(last=False)
            print(f"[cache] Evicted avatar_id={evicted_id!r} from engine cache")

        # Build a new engine that shares the already-loaded model weights
        new_engine = MuseTalkEngine(
            model_dir="/app/models",
            idle_path=None,       # resolved dynamically from avatar_id
            config=config,
            avatar_id=avatar_id,
            models=_shared_models,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, new_engine.initialize)
        print(f"[cache] Initialized engine for avatar_id={avatar_id!r} "
              f"(cache size: {len(_engine_cache) + 1}/{MAX_CACHED_AVATARS})")

        _engine_cache[avatar_id] = new_engine
        return new_engine


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


@app.on_event("startup")
async def startup():
    """Load model weights + initialize the default avatar engine.

    The blocking work runs in a thread executor so the event loop stays free
    to answer Cloud Run health checks (GET /) during startup.
    """
    global _shared_models  # noqa: PLW0603

    loop = asyncio.get_event_loop()

    # Step 1: load model weights once (shared across all avatar engines).
    def _load_models():
        import torch
        from engine.models import load_models
        from engine.inference import warmup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading shared MuseTalk models on {device} ...")
        m = load_models("/app/models", device)
        warmup(m)
        print("Shared models ready")
        return m

    _shared_models = await loop.run_in_executor(None, _load_models)

    # Step 2: pre-warm the "default" avatar engine so existing behaviour is
    # preserved and the /healthz and /dims endpoints work immediately.
    await _get_or_create_engine("default")


@app.get("/")
async def root():
    return FileResponse("/app/static/index.html")


@app.get("/healthz")
async def healthz():
    async with _cache_lock:
        default_engine = _engine_cache.get("default")
    if default_engine is None or not default_engine.ready:
        raise HTTPException(status_code=503, detail="engine initialising")
    return {"status": "ok"}


@app.get("/dims")
async def dims():
    async with _cache_lock:
        default_engine = _engine_cache.get("default")
    if default_engine is None or not default_engine.ready:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {"width": default_engine.avatar.frame_w, "height": default_engine.avatar.frame_h}


@app.get("/idle.mp4")
async def idle_video():
    path = "/app/idle.mp4"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="idle.mp4 not found")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# WebSocket endpoint — supports multi-avatar via LRU cache
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
    avatar_id: Optional[str] = None,
    voice_ref_path: Optional[str] = None,
):
    """WebSocket endpoint for talking-avatar inference.

    Query parameters
    ----------------
    avatar_id : str, optional
        Identifies which avatar to use.  Defaults to ``"default"``.
        The idle video for the avatar is loaded from
        ``{DATA_PATH}/avatars/{avatar_id}/idle.mp4`` (falls back to the global
        default if not found).
    voice_ref_path : str, optional
        Path to a reference audio file used for Chatterbox voice cloning.
        When provided, Chatterbox TTS is attempted first; edge-tts is the
        fallback.

    Protocol (unchanged)
    --------------------
    Client sends: JSON ``{"text": "..."}``
    Server sends: status / audio / chunk_start / frame / chunk_end / error
    """
    effective_avatar_id = avatar_id if avatar_id else "default"

    await ws.accept()
    print(f"WS connected (avatar_id={effective_avatar_id!r}, "
          f"voice_ref_path={voice_ref_path!r})")

    # Retrieve (or create) the engine for this avatar
    try:
        engine = await _get_or_create_engine(effective_avatar_id)
    except Exception:
        traceback.print_exc()
        await ws.send_json({"type": "error", "message": "engine initialisation failed"})
        await ws.close()
        return

    text_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    closed = False

    async def producer():
        nonlocal closed
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                text = msg.get("text", "").strip()
                if not text:
                    continue
                if text_queue.full():
                    try:
                        text_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await text_queue.put(text)
        except WebSocketDisconnect:
            closed = True
            await text_queue.put(None)

    async def consumer():
        nonlocal closed
        while True:
            text = await text_queue.get()
            if text is None:
                break
            if closed:
                break

            try:
                await ws.send_json({"type": "status", "message": "processing"})
            except Exception:
                break

            try:
                wav_path, audio_bytes = await tts_to_wav(
                    text,
                    VOICE,
                    voice_ref_path=voice_ref_path,
                )
                try:
                    frames, frame_w, frame_h = await engine.process_audio_chunk(wav_path)
                finally:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

                if closed:
                    break

                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                await ws.send_json({"type": "audio", "data": audio_b64})

                await ws.send_json({
                    "type": "chunk_start",
                    "frame_count": len(frames),
                    "fps": config.fps // config.frame_skip,
                    "width": frame_w,
                    "height": frame_h,
                })
                for i, fb in enumerate(frames):
                    if closed:
                        break
                    b64 = base64.b64encode(fb).decode("ascii")
                    await ws.send_json(
                        {"type": "frame", "data": b64, "index": i, "total": len(frames)}
                    )
                if not closed:
                    await ws.send_json({"type": "chunk_end"})

            except WebSocketDisconnect:
                closed = True
                break
            except Exception:
                traceback.print_exc()
                if not closed:
                    try:
                        await ws.send_json({"type": "error", "message": "inference failed"})
                    except Exception:
                        break

    try:
        await asyncio.gather(producer(), consumer())
    except Exception:
        pass
    print(f"WS disconnected (avatar_id={effective_avatar_id!r})")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
