#!/usr/bin/env python3
"""
unified_whisper_service.py
──────────────────────────
• POST /transcribe            – Upload audio → plain transcript
• POST /transcribe-diarized   – Audio path + RTTM(s) → diarized JSON

Only /transcribe-diarized writes to the rotating file log (see logger.py).
Both routes share a single global vLLM Whisper checkpoint, loaded once
at import-time.
"""
from __future__ import annotations
import io, math, os, asyncio, uuid
from pathlib import Path
from typing import List
import requests
import numpy as np
import json
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi import Form
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from helpers.logger import get_logger
from typing import List, Union
from pydantic import BaseModel
from fastapi import Body
from fastapi.responses import JSONResponse
from helpers.utils import (                       # ← all helpers now imported
    _load_disk_audio,
    _parse_srt_time,
    _read_rttm,
    _waveform_from_upload,
    seconds_to_srt_time,
)

class AudioData(BaseModel):
    data: List[float]
    sampling_rate: int

class MultiModalData(BaseModel):
    audio: AudioData

class PromptPayload(BaseModel):
    prompt: str
    multi_modal_data: MultiModalData
    request_id: str                   # structured logger (used only below)

# ─────────────────────── Configuration ────────────────────────
load_dotenv()

WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR")
# WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR",
#                               "/home/gcp-admin/whisper_finetune_v4")
TARGET_SR   = 16_000
MAX_SEG_SEC = 30.0

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────── Global vLLM instance ──────────────────
log = get_logger("Transcription Service")
log.info("📦 Loading vLLM Whisper")

llm = LLM(
    model=WHISPER_MODEL_DIR,
    max_num_seqs=32,
    dtype = "bfloat16",
    limit_mm_per_prompt={"audio": 1},
    tensor_parallel_size=4,
    kv_cache_dtype="fp8",
    enforce_eager=True,  # for better performance
    gpu_memory_utilization=0.95
)
SAMPLING = SamplingParams(temperature=0.1, top_p=0.1, max_tokens=1024,frequency_penalty=0.6)


# ─────────────────────── Pydantic (diarized) ───────────────────
class TransReq(BaseModel):
    audio_path: str
    rttm_paths: List[str]
    file_id: str | None = None
    callback_url: str | None = None

class Seg(BaseModel):
    start_time: str
    end_time: str
    speaker: str
    message: str

class TransResp(BaseModel):
    call_id: str
    duration: float
    diarized_data: List[Seg]


# ─────────────────────── FastAPI initialise ────────────────────
app = FastAPI(title="Unified Whisper (vLLM) Service", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────── /transcribe (plain) ───────────────────
@app.post("/voiceapi/transcribe", summary="Upload an audio file → plain transcript")
async def transcribe(audio_file: UploadFile = File(...)):
    # 1. Validate MIME
    if not audio_file.content_type.startswith("audio/"):
        log.error("❌ Invalid file type: %s", audio_file.content_type)
        raise HTTPException(415, "File must be audio (WAV/MP3/M4A/…)")

    log.info("📥 Received file: %s (%s)", audio_file.filename, audio_file.content_type)

    # 2. Decode + normalise
    try:
        wav, sr = _waveform_from_upload(audio_file)  # 1-D NumPy float32
        log.info("🔊 Audio decoded and resampled to %d Hz", sr)
    except Exception as e:
        log.error("❌ Audio decoding failed: %s", e)
        raise

    # 3. Chunk into ≤ MAX_SEG_SEC pieces
    samples_per_chunk = int(MAX_SEG_SEC * sr)
    total_samples     = len(wav)
    segments: list[np.ndarray] = [
        wav[i: i + samples_per_chunk]
        for i in range(0, total_samples, samples_per_chunk)
    ]
    log.info("🔪 Audio split into %d chunk(s)", len(segments))

    # 4. Transcribe each chunk sequentially
    partial_texts: list[str] = []
    for idx, segment in enumerate(segments):
        try:
            prompts = [{
                "prompt": "<|startoftranscript|>",
                "multi_modal_data": {"audio": (segment, sr)},
                "request_id": f"{uuid.uuid4().hex}_{idx}",
            }]
            out = llm.generate(prompts, SAMPLING)
            text_chunk = out[0].outputs[0].text.strip()
            partial_texts.append(text_chunk)
            log.info("✅ Transcribed chunk %d/%d", idx + 1, len(segments))
        except Exception as e:
            log.error("❌ Transcription failed for chunk %d: %s", idx + 1, e)
            raise

    full_transcript: str = "\n".join(partial_texts)
    log.info("📄 Full transcription completed for: %s", audio_file.filename)

    return PlainTextResponse(full_transcript)

#───────────────────── /transcribe-diarized ────────────────────

@app.post("/voiceapi/transcribe-llm", summary="Use external prompts and sampling_params")
async def transcribe_with_custom_params(payload: List[PromptPayload] = Body(...)):
    try:
        processed_prompts = []
        log.info("📄 Received %d prompts", len(payload))

        for p in payload:
            audio_np = np.array(p.multi_modal_data.audio.data, dtype=np.float32)
            sr = p.multi_modal_data.audio.sampling_rate

            assert audio_np.ndim == 1, "Audio is not 1D"
            assert audio_np.dtype == np.float32, "Audio dtype is not float32"
            assert isinstance(sr, int), "Sampling rate must be int"

            processed_prompts.append({
                "prompt": p.prompt,
                "multi_modal_data": {
                    "audio": (audio_np, sr)
                },
                "request_id": p.request_id
            })

        log.info("🧪 Transcribing %d prompts...", len(processed_prompts))
        log.debug("🧾 First prompt: %s", processed_prompts[0])

        result = llm.generate(processed_prompts, SAMPLING)
        log.info("✅ Transcription successful")
        output_data = []
        for r in result:
            output_data.append({
                "request_id": r.request_id,
                "text": r.outputs[0].text if r.outputs else "",
            })
        return JSONResponse(content=output_data)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error("❌ LLM transcription failed: %s\n%s", e, tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": tb}
        )
