"""
Tests for engine/tts.py — run without a GPU or real edge-tts calls.

Mock targets:
  edge_tts.Communicate  — prevents network calls
  engine.tts.subprocess.run — prevents ffmpeg calls
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_returns_wav_path_and_mp3_bytes():
    """tts_to_wav returns a (wav_path, mp3_bytes) tuple; mp3_bytes is non-empty."""
    from engine.tts import tts_to_wav

    mp3_content = b"fake mp3 data"

    async def fake_save(path):
        with open(path, "wb") as f:
            f.write(mp3_content)

    mock_communicate = MagicMock()
    mock_communicate.save = fake_save

    def fake_run(cmd, **kwargs):
        wav_path = cmd[-1]
        with open(wav_path, "wb") as f:
            f.write(b"fake wav data")
        return MagicMock(returncode=0)

    with patch("edge_tts.Communicate", return_value=mock_communicate), \
         patch("engine.tts.subprocess.run", side_effect=fake_run):
        wav_path, mp3_bytes = await tts_to_wav("hello", voice="en-US-GuyNeural")

    assert wav_path.endswith(".wav")
    assert mp3_bytes == mp3_content
    os.unlink(wav_path)


@pytest.mark.asyncio
async def test_cleans_up_mp3_on_success():
    """MP3 temp file is unlinked after tts_to_wav returns successfully."""
    from engine.tts import tts_to_wav

    mp3_path_seen = []

    async def fake_save(path):
        mp3_path_seen.append(path)
        with open(path, "wb") as f:
            f.write(b"mp3 data")

    mock_communicate = MagicMock()
    mock_communicate.save = fake_save

    def fake_run(cmd, **kwargs):
        wav_path = cmd[-1]
        with open(wav_path, "wb") as f:
            f.write(b"wav data")
        return MagicMock(returncode=0)

    with patch("edge_tts.Communicate", return_value=mock_communicate), \
         patch("engine.tts.subprocess.run", side_effect=fake_run):
        wav_path, _ = await tts_to_wav("hello", voice="en-US-GuyNeural")

    assert mp3_path_seen, "communicate.save() was never called"
    assert not os.path.exists(mp3_path_seen[0]), "MP3 was not cleaned up after success"
    os.unlink(wav_path)


@pytest.mark.asyncio
async def test_cleans_up_mp3_on_communicate_error():
    """MP3 temp file is unlinked even when communicate.save() raises."""
    from engine.tts import tts_to_wav

    mp3_path_seen = []

    async def fake_save_error(path):
        mp3_path_seen.append(path)
        with open(path, "wb") as f:
            f.write(b"mp3 data")
        raise RuntimeError("edge-tts network failure")

    mock_communicate = MagicMock()
    mock_communicate.save = fake_save_error

    with patch("edge_tts.Communicate", return_value=mock_communicate):
        with pytest.raises(RuntimeError):
            await tts_to_wav("hello", voice="en-US-GuyNeural")

    assert mp3_path_seen, "communicate.save() was never called"
    assert not os.path.exists(mp3_path_seen[0]), "MP3 was not cleaned up after error"


@pytest.mark.asyncio
async def test_raises_on_ffmpeg_failure():
    """CalledProcessError propagates when ffmpeg exits with a non-zero return code."""
    from engine.tts import tts_to_wav

    async def fake_save(path):
        with open(path, "wb") as f:
            f.write(b"mp3 data")

    mock_communicate = MagicMock()
    mock_communicate.save = fake_save

    with patch("edge_tts.Communicate", return_value=mock_communicate), \
         patch("engine.tts.subprocess.run",
               side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        with pytest.raises(subprocess.CalledProcessError):
            await tts_to_wav("hello", voice="en-US-GuyNeural")
