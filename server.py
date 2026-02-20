import base64
import json
import os
import sys
import asyncio
import traceback

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add MuseTalk to Python path; add /app so `import engine` works from /app/MuseTalk
sys.path.insert(0, "/app/MuseTalk")
sys.path.insert(0, "/app")

from engine import MuseTalkEngine, EngineConfig
from engine.tts import tts_to_wav

VOICE = "en-US-GuyNeural"
config = EngineConfig(frame_skip=2, max_output_dim=1280, jpeg_quality=70, batch_size=8, fps=25)
engine = MuseTalkEngine(model_dir="/app/models", idle_path="/app/idle.mp4", config=config)

app = FastAPI()
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


@app.on_event("startup")
async def startup():
    # Run the blocking model-load + preprocessing in a thread so the event loop
    # stays free to answer Cloud Run health checks (GET /) during startup.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.initialize)


@app.get("/")
async def root():
    return FileResponse("/app/static/index.html")


@app.get("/healthz")
async def healthz():
    if not engine.ready:
        raise HTTPException(status_code=503, detail="engine initialising")
    return {"status": "ok"}


@app.get("/dims")
async def dims():
    if not engine.ready:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {"width": engine.avatar.frame_w, "height": engine.avatar.frame_h}


@app.get("/idle.mp4")
async def idle_video():
    path = "/app/idle.mp4"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="idle.mp4 not found")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WS connected")
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
                wav_path, mp3_bytes = await tts_to_wav(text, VOICE)
                try:
                    frames, frame_w, frame_h = await engine.process_audio_chunk(wav_path)
                finally:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

                if closed:
                    break

                audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
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
    print("WS disconnected")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
