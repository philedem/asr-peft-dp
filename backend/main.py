from __future__ import annotations
import os, io, uuid, json, tempfile, asyncio, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import soundfile as sf
import numpy as np
from pydub import AudioSegment, silence
import torch
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from peft import PeftModel

# ───────────────────────────── configuration ──────────────────────────────
AUDIO_DIR   = Path("data/audio")
RECORD_DIR  = Path("data/records")
ADAPTER_DIR = Path("data/lora_output")
WER_FILE    = Path("data/wer.txt")
BASELINE_WER_FILE = Path("data/baseline_wer.txt")
TRAINING_ITERATION_FILE = Path("data/training_iteration.txt")
TRAINING_STATUS_FILE = Path("data/training_status.json")

MIN_SILENCE_MS     = 2000                   # chunk params
KEEP_SILENCE_MS    = 300
MANUAL_RETRAIN_N   = 20                      # reviewed chunks before retrain
AUTO_RETRAIN_ENABLED = False                 # set to False to disable auto-retraining
COUNTER_FILE       = Path("data/manual_review_count.txt")

# dirs exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize WER file if it doesn't exist
if not WER_FILE.exists():
    WER_FILE.write_text("WER: N/A\n")

# Device detection: prioritize CUDA > MPS > CPU (can override with DEVICE env var)
DEVICE = os.getenv("DEVICE")
if DEVICE:
    print(f"Using device from environment: {DEVICE}")
else:
    # Debug CUDA detection
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA built version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Auto-detected device: {DEVICE}")

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
    """Whisper inference using transformers with LoRA support."""
    try:
        # Use the ASR pipeline with the model that has LoRA loaded
        result = asr_pipeline(str(path), generate_kwargs={"language": "norwegian", "task": "transcribe"})
        return result["text"].strip()
        
    except Exception as e:
        print(f"Error transcribing {path}: {e}")
        import traceback
        traceback.print_exc()
        return "[transcription failed]"


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


def _set_training_status(status: str, progress: int = 0, message: str = ""):
    """Update training status file."""
    status_data = {
        "status": status,  # "idle", "running", "completed", "failed"
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    TRAINING_STATUS_FILE.write_text(json.dumps(status_data))


def _get_training_status() -> dict:
    """Get current training status."""
    if TRAINING_STATUS_FILE.exists():
        try:
            return json.loads(TRAINING_STATUS_FILE.read_text())
        except:
            pass
    return {"status": "idle", "progress": 0, "message": "", "timestamp": None}


def _split_long_chunks(chunks: List[AudioSegment], max_length_ms: int = 40000, overlap_ms: int = 2000) -> List[AudioSegment]:
    """Further split chunks longer than max_length_ms into smaller chunks with overlap.
    
    Args:
        chunks: List of audio segments to potentially split
        max_length_ms: Maximum length of each chunk in milliseconds (default 40s)
        overlap_ms: Overlap between consecutive chunks to avoid cutting words (default 2s)
    """
    new_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length_ms:
            new_chunks.append(chunk)
        else:
            # Split into smaller chunks with overlap to avoid cutting words at boundaries
            start = 0
            while start < len(chunk):
                end = min(start + max_length_ms, len(chunk))
                new_chunks.append(chunk[start:end])
                # Move start forward by (max_length - overlap) to create overlap
                # This ensures the last overlap_ms of this chunk is also in the next chunk
                start = end - overlap_ms
                # If we're very close to the end, just include the rest
                if len(chunk) - start < overlap_ms:
                    break
    return new_chunks


# ──────────────────────────────  FastAPI   ─────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Reset training status to idle on startup (in case of crash/restart with stale status)
_set_training_status("idle", 0, "")

# Check if model is cached, download if needed
cache_dir = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
model_cache_path = cache_dir / "hub"
if model_cache_path.exists() and any(model_cache_path.iterdir()):
    print(f"Model cache found at {model_cache_path}")
else:
    print(f"Model not cached. Downloading {BASE_MODEL} to {cache_dir}...")
    print("This is a one-time download (~1.5GB) and will be cached for future use.")

# Load transformers model for both inference and training
print(f"Loading Whisper model ({BASE_MODEL}) for inference and training...")
processor: WhisperProcessor = WhisperProcessor.from_pretrained(BASE_MODEL)
_base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)

# Apply LoRA adapter if available
model = _attach_lora(_base)

# Create ASR pipeline for inference
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=DEVICE,
)

MODEL_LOCK  = asyncio.Lock()  # serialise GPU access
RECORD_LOCK = asyncio.Lock()  # serialise record read-modify-write
print(f"✓ Model loaded on {DEVICE} with LoRA adapter (if available)")

# ───────── 1. receive stream/blob -> chunk -> transcribe ──────────
@app.post("/asr/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        raw = await audio.read()
        if not raw:
            return JSONResponse({"error": "Empty audio file"}, status_code=400)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                AudioSegment.from_file(io.BytesIO(raw)).export(tmp_path, format="wav")
                seg = AudioSegment.from_file(tmp_path)
            except Exception as e:
                os.remove(tmp_path)
                return JSONResponse({"error": f"Invalid audio format: {str(e)}"}, status_code=400)

        chunks = silence.split_on_silence(
            seg,
            min_silence_len=MIN_SILENCE_MS,
            keep_silence=KEEP_SILENCE_MS,
            silence_thresh=seg.dBFS - 14,
        )
        
        if not chunks:
            # If no silence detected, treat entire audio as one chunk
            chunks = [seg]

        # split chunks > 40s with 2s overlap
        chunks = _split_long_chunks(chunks, max_length_ms=40000, overlap_ms=2000)

        parent = uuid.uuid4().hex[:8]
        created: List[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_id  = f"{parent}_{idx}"
            wav_path  = AUDIO_DIR / f"{chunk_id}.wav"
            chunk.export(wav_path, format="wav")

            async with MODEL_LOCK:
                asr_text = _transcribe_wav(wav_path)

            rec = _new_record_dict(wav_path.name, asr_text)
            async with RECORD_LOCK:
                _atomic_json_write(RECORD_DIR / f"{chunk_id}.json", rec)
            created.append(chunk_id)

        os.remove(tmp_path)
        return {"created": len(created), "chunks": created}
    
    except Exception as e:
        print(f"Error in transcribe_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=500)


# ───────── 2. operator annotates / approves a record ───────────
@app.post("/asr/save_record")
async def save_record(req: Request):
    try:
        body = await req.json()
        
        if "audio_id" not in body:
            return JSONResponse({"error": "Missing audio_id"}, status_code=400)
        
        file_stem = Path(body["audio_id"]).stem    # strip .wav if present
        json_path = RECORD_DIR / f"{file_stem}.json"

        async with RECORD_LOCK:
            # load existing (to keep timestamp)
            record: Dict[str, Any]
            if json_path.exists():
                try:
                    record = json.loads(json_path.read_text())
                except json.JSONDecodeError:
                    return JSONResponse({"error": "Corrupt record file"}, status_code=500)
            else:
                record = _new_record_dict(f"{file_stem}.wav", body.get("asr_transcript", ""))

            # update fields
            record["asr_transcript"]    = body.get("asr_transcript",    record["asr_transcript"])
            record["manual_transcript"] = body.get("manual_transcript", record["manual_transcript"])

            _atomic_json_write(json_path, record)

            # if reviewed -> bump counter
            if record["manual_transcript"] and record["manual_transcript"] != record["asr_transcript"]:
                reviews = _increment_review_counter()
                print(f"Manual reviews since last retrain: {reviews}")
                if AUTO_RETRAIN_ENABLED and reviews >= MANUAL_RETRAIN_N:
                    print("Threshold reached – launching LoRA fine-tune ...")
                    _set_training_status("running", 0, "Auto-training started (20 corrections reached)")
                    subprocess.Popen(["/usr/bin/python3.12", "train_lora.py"])
                    _reset_review_counter()
                elif not AUTO_RETRAIN_ENABLED:
                    print("Auto-retraining is disabled. Use manual retrain button to start training.")

        return {"ok": True}
    
    except Exception as e:
        print(f"Error in save_record: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Save failed: {str(e)}"}, status_code=500)

@app.get("/train/retrain_lora")
async def retrain_lora():
    """Trigger LoRA fine-tuning (non-blocking)."""
    print("Launching manual LoRA fine-tune ...")
    _set_training_status("running", 0, "Training started")
    subprocess.Popen(["/usr/bin/python3.12", "train_lora.py"])
    return {"status": "retraining started"}

@app.get("/train/calculate_wer")
async def calculate_wer_only():
    """Calculate WER without retraining - for benchmarking."""
    print("Calculating WER without training...")
    _set_training_status("running", 0, "Calculating WER...")
    try:
        # Run a Python script that calculates WER only
        subprocess.Popen(["/usr/bin/python3.12", "calculate_wer_only.py"])
        return {"status": "wer_calculation_started"}
    except Exception as e:
        _set_training_status("failed", 0, f"WER calculation failed: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

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


@app.delete("/asr/records/{record_id}")
def delete_record(record_id: str):
    """Delete a record and its associated audio file."""
    try:
        # Find and delete the JSON file
        json_file = RECORD_DIR / f"{record_id}.json"
        if not json_file.exists():
            return JSONResponse({"error": "Record not found"}, status_code=404)
        
        # Load the record to get audio filename
        record = json.loads(json_file.read_text())
        audio_filename = record.get("audio_file")
        
        # Delete the JSON file
        json_file.unlink()
        print(f"Deleted record: {record_id}.json")
        
        # Delete the audio file if it exists
        if audio_filename:
            audio_file = AUDIO_DIR / audio_filename
            if audio_file.exists():
                audio_file.unlink()
                print(f"Deleted audio: {audio_filename}")
        
        return {"ok": True, "deleted": record_id}
    
    except Exception as e:
        print(f"Error deleting record {record_id}: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Delete failed: {str(e)}"}, status_code=500)


@app.get("/asr/wer")
def get_wer():
    if WER_FILE.is_file():
        content = WER_FILE.read_text().strip()
        # Parse multi-line format - look for line starting with "WER:"
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("WER:"):
                wer_value = line.split(":", 1)[1].strip()
                if wer_value and wer_value != "N/A":
                    return {"wer": wer_value}
    return {"wer": None}


@app.get("/asr/device_info")
def get_device_info():
    """Get information about the compute device being used."""
    device_info = {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_device_count"] = torch.cuda.device_count()
    
    return device_info


@app.get("/asr/model_info")
def get_model_info():
    """Get information about the model being used."""
    # Check if LoRA adapter is loaded
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    adapter_files = [(ADAPTER_DIR / f).exists() for f in required_files]
    has_lora = all(adapter_files)
    
    # Get training iteration if available
    training_iteration = None
    if TRAINING_ITERATION_FILE.exists():
        try:
            training_iteration = int(TRAINING_ITERATION_FILE.read_text().strip())
        except:
            pass
    
    return {
        "base_model": BASE_MODEL,
        "has_lora_adapter": has_lora,
        "training_iteration": training_iteration,
        "device": DEVICE,
    }


@app.get("/train/status")
def get_training_status():
    """Get current training status."""
    return _get_training_status()


@app.get("/audio/{fname}")
def serve_audio(fname: str):
    # Guard against path traversal
    if ".." in fname or "/" in fname or "\\" in fname:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = (AUDIO_DIR / fname).resolve()
    if not path.parent == AUDIO_DIR.resolve():
        return JSONResponse({"error": "invalid filename"}, status_code=400)
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