#!/usr/bin/env python3
"""
Quick script to evaluate baseline WER before any fine-tuning.
This helps students see the improvement from the initial model.

Usage:
    python get_baseline_wer.py

This will:
1. Load all manually corrected samples
2. Run base model (no LoRA) on them  
3. Calculate WER
4. Save to data/baseline_wer.txt
5. Log to MLflow for comparison
"""

import os, json
import torch
import soundfile as sf
import torchaudio
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
import mlflow
from datetime import datetime

MODEL_NAME = "NbAiLab/nb-whisper-medium"
RECORD_DIR = "data/records"
AUDIO_DIR = "data/audio"
OUTPUT_FILE = "data/baseline_wer.txt"

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading base model (no fine-tuning): {MODEL_NAME}")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def transcribe(audio_path):
    """Transcribe a single audio file."""
    wav, sr = sf.read(audio_path)
    wav = torch.from_numpy(wav).float()
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T
    
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    features = processor(
        wav.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(DEVICE)
    
    ids = model.generate(
        features,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            task="transcribe", 
            language="no"
        )
    )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# Load all manually corrected records
print(f"\nLoading corrected records from {RECORD_DIR}/...")
records = []
for f in os.listdir(RECORD_DIR):
    if f.endswith(".json"):
        with open(os.path.join(RECORD_DIR, f)) as json_file:
            d = json.load(json_file)
            if d.get("manual_transcript", "").strip():
                records.append(d)

if not records:
    print("❌ No corrected records found!")
    print("   Please annotate some audio first, then run this script.")
    exit(1)

print(f"Found {len(records)} corrected samples")
print("\nEvaluating baseline model...")

# Transcribe all samples
preds, refs = [], []
for i, r in enumerate(records, 1):
    audio_path = os.path.join(AUDIO_DIR, r["audio_file"])
    pred = transcribe(audio_path)
    ref = r["manual_transcript"]
    
    preds.append(pred)
    refs.append(ref)
    
    if i % 10 == 0:
        print(f"  Processed {i}/{len(records)}...")

# Calculate WER
wer = jiwer.wer(refs, preds)

# Save results
print(f"\n{'='*60}")
print(f"BASELINE WER (Before Fine-Tuning)")
print(f"{'='*60}")
print(f"Samples: {len(records)}")
print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
print(f"{'='*60}\n")

# Write detailed report
with open(OUTPUT_FILE, "w") as f:
    f.write(f"=== Baseline WER Report ===\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Number of samples: {len(records)}\n")
    f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
    f.write(f"\n--- Sample Predictions (first 5) ---\n")
    for i in range(min(5, len(records))):
        f.write(f"\nSample {i+1}:\n")
        f.write(f"  Reference:  {refs[i]}\n")
        f.write(f"  Prediction: {preds[i]}\n")

print(f"✓ Report saved to: {OUTPUT_FILE}")

# Log to MLflow for comparison
mlflow.set_experiment("CISK-PEFT-FineTuning")
with mlflow.start_run(run_name="baseline_no_finetuning"):
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "num_samples": len(records),
        "training_iteration": 0,
        "is_baseline": True
    })
    mlflow.set_tag("experiment_type", "Baseline")
    mlflow.set_tag("training_date", datetime.now().isoformat())
    mlflow.log_metric("WER", wer)
    mlflow.log_metric("num_train_samples", len(records))
    mlflow.log_artifact(OUTPUT_FILE)

print("✓ Results logged to MLflow")
print("\nℹ️  You can now compare this baseline with fine-tuned models in MLflow UI:")
print("   mlflow ui --port 5000")
print("   Open: http://localhost:5000")
