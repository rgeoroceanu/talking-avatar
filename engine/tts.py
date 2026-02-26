"""
engine/tts.py — Text → MP3/WAV.

Primary: Chatterbox TTS (self-hosted, configurable via CHATTERBOX_URL env var).
Fallback: edge-tts (Microsoft Neural TTS via the edge-tts library).

Chatterbox API contract:
  POST {CHATTERBOX_URL}/tts
  Body JSON: {"text": "...", "voice_ref_path": "..."}
  Response JSON: {"audio_base64": "..."}   (WAV bytes, base64-encoded)
"""

import base64
import os
import subprocess
import tempfile

import edge_tts
import httpx

CHATTERBOX_URL: str = os.environ.get("CHATTERBOX_URL", "http://localhost:8082")


async def synthesize_chatterbox(text: str, voice_ref_path: str | None) -> bytes:
    """Call the Chatterbox TTS API and return raw audio bytes (WAV).

    Raises httpx.HTTPError or ValueError on failure so the caller can fall back
    to edge-tts.
    """
    payload: dict = {"text": text}
    if voice_ref_path is not None:
        payload["voice_ref_path"] = voice_ref_path

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{CHATTERBOX_URL}/tts", json=payload)
        response.raise_for_status()

    data = response.json()
    audio_b64 = data.get("audio_base64")
    if not audio_b64:
        raise ValueError("Chatterbox response missing 'audio_base64' field")

    return base64.b64decode(audio_b64)


async def _edge_tts_to_wav(text: str, voice: str, ffmpeg_bin: str = "ffmpeg") -> tuple:
    """Convert text to speech via edge-tts, then transcode to 16 kHz mono WAV.

    Returns (wav_path, mp3_bytes). Caller must unlink wav_path.
    """
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(mp3_fd)
    os.close(wav_fd)
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(mp3_path)
        mp3_bytes = open(mp3_path, "rb").read()
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True,
            check=True,
        )
    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)
    return wav_path, mp3_bytes


async def tts_to_wav(
    text: str,
    voice: str,
    ffmpeg_bin: str = "ffmpeg",
    voice_ref_path: str | None = None,
) -> tuple:
    """Convert text to speech and return (wav_path, audio_bytes).

    Strategy:
      1. If voice_ref_path is provided, attempt Chatterbox TTS first.
      2. On any Chatterbox failure (or if voice_ref_path is None), fall back to edge-tts.

    Returns (wav_path, audio_bytes). Caller is responsible for unlinking wav_path.
    audio_bytes are the raw bytes of the returned audio (MP3 from edge-tts, or
    WAV from Chatterbox — the browser/client should handle both).
    """
    if voice_ref_path is not None:
        try:
            wav_bytes = await synthesize_chatterbox(text, voice_ref_path)

            # Write Chatterbox WAV bytes to a temp file for MuseTalk to consume.
            wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(wav_fd)
            with open(wav_path, "wb") as fh:
                fh.write(wav_bytes)

            # Ensure the WAV is 16 kHz mono (Chatterbox may return different sample rate).
            resampled_fd, resampled_path = tempfile.mkstemp(suffix=".wav")
            os.close(resampled_fd)
            try:
                subprocess.run(
                    [
                        ffmpeg_bin, "-y", "-i", wav_path,
                        "-ar", "16000", "-ac", "1", "-f", "wav", resampled_path,
                    ],
                    capture_output=True,
                    check=True,
                )
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

            return resampled_path, wav_bytes

        except Exception as exc:
            print(f"[tts] Chatterbox failed ({exc}); falling back to edge-tts")

    # Fallback: edge-tts
    return await _edge_tts_to_wav(text, voice, ffmpeg_bin=ffmpeg_bin)
