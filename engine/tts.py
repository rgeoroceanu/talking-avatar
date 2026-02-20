"""
engine/tts.py — Text → MP3 (edge-tts) → WAV (ffmpeg).
"""

import os
import subprocess
import tempfile

import edge_tts


async def tts_to_wav(text: str, voice: str, ffmpeg_bin: str = "ffmpeg") -> tuple:
    """Convert text to speech via edge-tts, then transcode to 16kHz mono WAV.

    Returns (wav_path, mp3_bytes). Caller must unlink wav_path.
    MP3 temp file is cleaned up internally.
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
