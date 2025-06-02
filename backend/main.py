from __future__ import annotations
import os, io, uuid, json, tempfile, asyncio, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torchaudio
from pydub import AudioSegment, silence
import torch
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ───────────────────────────── configuration ──────────────────────────────
AUDIO_DIR   = Path("data/audio")
RECORD_DIR  = Path("data/records")
ADAPTER_DIR = Path("data/lora_output")
WER_FILE    = Path("data/wer.txt")

MIN_SILENCE_MS     = 2000                   # chunk params
KEEP_SILENCE_MS    = 300
MANUAL_RETRAIN_N   = 20                      # reviewed chunks before retrain
COUNTER_FILE       = Path("data/manual_review_count.txt")

# dirs exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")
BASE_MODEL  = "NbAiLab/nb-whisper-medium"



# ───────────────────────────── helper functions ───────────────────────────

def _attach_lora(base) -> WhisperForConditionalGeneration:
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    adapter_files = [ (ADAPTER_DIR / f).exists() for f in required_files ]
    if all(adapter_files):
        print(f"Loading LoRA adapter from {ADAPTER_DIR}")
        return PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    else:
        print(f"No complete LoRA adapter found in {ADAPTER_DIR}, using base model.")
    return base

def _transcribe_wav(path: Path) -> str:
    """Blocking whisper inference (must be run under MODEL_LOCK)."""
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:                    # stereo to mono
        wav = wav.mean(dim=0, keepdim=True)

    feats = processor(
        wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    ids = model.generate(
        feats,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            task="transcribe", language="no"
        ),
    )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _atomic_json_write(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f_tmp:
        json.dump(data, f_tmp, ensure_ascii=False, indent=2)
    tmp.replace(path)                  


def _new_record_dict(audio_file: str, asr: str,
                     manual: str = "") -> Dict[str, Any]:
    return {
        "audio_file":         audio_file,
        "asr_transcript":     asr,
        "manual_transcript":  manual,
        "timestamp":          datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _increment_review_counter() -> int:
    cnt = 0
    if COUNTER_FILE.exists():
        try:
            cnt = int(COUNTER_FILE.read_text())
        except ValueError:
            pass
    cnt += 1
    COUNTER_FILE.write_text(str(cnt))
    return cnt


def _reset_review_counter() -> None:
    COUNTER_FILE.write_text("0")


def _split_long_chunks(chunks: List[AudioSegment], max_length_ms: int = 30000) -> List[AudioSegment]:
    """Further split chunks longer than max_length_ms into smaller chunks."""
    new_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length_ms:
            new_chunks.append(chunk)
        else:
            # split into smaller chunks of max_length_ms with no overlap
            start = 0
            while start < len(chunk):
                end = min(start + max_length_ms, len(chunk))
                new_chunks.append(chunk[start:end])
                start = end
    return new_chunks


# ──────────────────────────────  FastAPI   ─────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
print("Loading Whisper processor + base model ...")
processor: WhisperProcessor = WhisperProcessor.from_pretrained(BASE_MODEL)
_base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)

model = _attach_lora(_base)
MODEL_LOCK = asyncio.Lock()                 # serialise GPU access

# ───────── 1. receive stream/blob -> chunk -> transcribe ──────────
@app.post("/asr/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    raw = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        AudioSegment.from_file(io.BytesIO(raw)).export(tmp.name, format="wav")
        seg = AudioSegment.from_file(tmp.name)

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_MS,
        keep_silence=KEEP_SILENCE_MS,
        silence_thresh=seg.dBFS - 14,
    )

    # split chunks > 30s
    chunks = _split_long_chunks(chunks, max_length_ms=30000)

    parent = uuid.uuid4().hex[:8]
    created: List[str] = []

    for idx, chunk in enumerate(chunks):
        chunk_id  = f"{parent}_{idx}"
        wav_path  = AUDIO_DIR / f"{chunk_id}.wav"
        chunk.export(wav_path, format="wav")

        async with MODEL_LOCK:
            asr_text = _transcribe_wav(wav_path)

        rec = _new_record_dict(wav_path.name, asr_text)
        _atomic_json_write(RECORD_DIR / f"{chunk_id}.json", rec)
        created.append(chunk_id)

    os.remove(tmp.name)
    return {"created": len(created), "chunks": created}


# ───────── 2. operator annotates / approves a record ───────────
@app.post("/asr/save_record")
async def save_record(req: Request):
    body = await req.json()
    file_stem = Path(body["audio_id"]).stem    # strip .wav if present
    json_path = RECORD_DIR / f"{file_stem}.json"

    # load existing (to keep timestamp)
    record: Dict[str, Any]
    if json_path.exists():
        record = json.loads(json_path.read_text())
    else:
        record = _new_record_dict(f"{file_stem}.wav", body.get("asr_transcript", ""))

    # update fields
    record["asr_transcript"]    = body.get("asr_transcript",    record["asr_transcript"])
    record["manual_transcript"] = body.get("manual_transcript", record["manual_transcript"])
    # record["flagged"]           = body.get("flagged",           record.get("flagged", False))

    _atomic_json_write(json_path, record)

    # if reviewed -> bump counter
    # if record["manual_transcript"] and record["manual_transcript"] != record["asr_transcript"]:
    #     reviews = _increment_review_counter()
    #     print(f"Manual reviews since last retrain: {reviews}")
    #     if reviews >= MANUAL_RETRAIN_N:
    #         print("Threshold reached – launching LoRA fine-tune ...")
    #         subprocess.Popen(["python", "fl_train_lora.py"])
    #         _reset_review_counter()

    return {"ok": True}

@app.get("/train/retrain_lora")
async def retrain_lora():
    """Trigger LoRA fine-tuning (non-blocking)."""
    print("Launching manual LoRA fine-tune ...")
    subprocess.Popen(["python", "train_lora.py"])
    return {"status": "retraining started"}

# ───────── 3. list records, WER, serve audio ───────────
@app.get("/asr/records")
def list_records():
    recs = []
    for js in RECORD_DIR.glob("*.json"):
        try:
            recs.append(json.loads(js.read_text()))
        except json.JSONDecodeError:
            print("Skipped bad JSON:", js.name)
    recs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return recs


@app.get("/asr/wer")
def get_wer():
    if WER_FILE.is_file():
        line = WER_FILE.read_text().strip()
        if line.startswith("WER:"):
            return {"wer": line.split(":", 1)[1].strip()}
    return {"wer": None}


@app.get("/audio/{fname}")
def serve_audio(fname: str):
    path = AUDIO_DIR / fname
    if not path.exists():
        return JSONResponse({"error": "audio not found"}, status_code=404)
    return FileResponse(path)


# ───────── 4. hot-reload LoRA adapter (non-blocking) ───────────
@app.post("/asr/reload_adapter")
async def reload_adapter():
    print("Reloading LoRA adapter ...")
    global model
    async with MODEL_LOCK:
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
        model = _attach_lora(base)
    return {"status": "reloaded"}