#!/usr/bin/env python3
"""
utils.py
────────
Standalone helpers shared by the transcription-service code-base.

Functions
---------
seconds_to_srt_time(seconds)      → "HH:MM:SS,mmm"
_parse_srt_time(timestr)          → float seconds
_waveform_from_upload(upload, sr) → (np.ndarray mono-wav, sr)
_load_disk_audio(path)            → (np.ndarray mono-wav, sr)
_read_rttm(rttm_paths, limit)     → list[dict] with ≤ limit-second segments
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List

import numpy as np
import torchaudio
from fastapi import UploadFile, HTTPException


# ─────────────────────── Time helpers ────────────────────────
def seconds_to_srt_time(seconds: float) -> str:
    """Convert *seconds* → SRT time-code string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1_000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _parse_srt_time(timestr: str) -> float:
    """Convert 'HH:MM:SS,mmm' → float seconds."""
    hh, mm, rest = timestr.split(":")
    ss, ms = rest.split(",")
    return (
        int(hh) * 3600
        + int(mm) * 60
        + int(ss)
        + int(ms) / 1_000
    )


# ─────────────────────── Audio helpers ───────────────────────
def _waveform_from_upload(
    upload: UploadFile,
    target_sr: int = 16_000,
) -> tuple[np.ndarray, int]:
    """
    Decode a FastAPI `UploadFile` into a **mono** float32 NumPy vector,
    resampled to *target_sr* and RMS-normalised.
    """
    raw = upload.file.read()

    try:
        wav, sr = torchaudio.load(io.BytesIO(raw), format=None)
    except Exception as exc:
        raise HTTPException(400, f"Unable to decode audio: {exc}") from exc

    if wav.size(0) > 1:                     # stereo → mono
        wav = wav.mean(0, keepdim=True)

    if sr != target_sr:                     # resample if needed
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.squeeze().numpy().astype("float32")
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 0:
        wav /= max(rms, 1e-4)              # simple RMS-norm
    return wav, sr


def _load_disk_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load *path* from disk and return a **mono** float32 NumPy vector
    with its native sample-rate (no resampling).
    """
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze().numpy().astype("float32"), sr


# ─────────────────────── RTTM helpers ────────────────────────
def _read_rttm(
    rttm_paths: List[Path],
    max_seg_sec: float = 30.0,
) -> list[dict]:
    """
    Parse one or more RTTM files and return a list of segments:
    {start, end, speaker}. Segments longer than *max_seg_sec* are split
    into equal-length chunks.
    """
    raw: list[dict] = []
    for p in rttm_paths:
        with open(p) as fh:
            for ln in fh:
                bits = ln.strip().split()
                if bits[0] != "SPEAKER":
                    continue
                start, dur, spk = float(bits[3]), float(bits[4]), bits[7]
                raw.append({"start": start, "end": start + dur, "speaker": spk})

    segs: list[dict] = []
    for seg in raw:
        span = seg["end"] - seg["start"]
        if span <= max_seg_sec:
            segs.append(seg)
        else:                               # split overly long segments
            parts = math.ceil(span / max_seg_sec)
            for i in range(parts):
                segs.append(
                    {
                        "start": seg["start"] + i * max_seg_sec,
                        "end": min(
                            seg["start"] + (i + 1) * max_seg_sec, seg["end"]
                        ),
                        "speaker": seg["speaker"],
                    }
                )
    segs.sort(key=lambda x: x["start"])
    return segs
