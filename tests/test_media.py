from __future__ import annotations

import subprocess
import wave

import numpy as np

from minimax_h3_mlx import media


def test_save_wav_creates_output_parent(tmp_path):
    output = tmp_path / "nested" / "audio.wav"

    media.save_wav(output, np.zeros((2, 16), dtype=np.float32), 32000)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getnframes() == 16


def test_save_mp4_creates_output_parent_without_audio(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "video.mp4"
    monkeypatch.setattr(media, "_resolve_executable", lambda *args, **kwargs: "ffmpeg")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b"", b""),
    )

    media.save_mp4(output, np.zeros((1, 2, 2, 3), dtype=np.uint8), 24)

    assert output.parent.is_dir()