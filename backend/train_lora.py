"""
Trains / updates LoRA from reviewed chunks,
writes WER, pings backend to reload.
"""

import os, json, torch, requests, warnings
from datasets import Dataset
import torchaudio, jiwer, torch
import soundfile as sf
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from functools import partial
from opacus import PrivacyEngine
import mlflow
import random
import numpy as np
from itertools import product
from pathlib import Path

MODEL_NAME      = "NbAiLab/nb-whisper-medium"

# Device detection: prioritize CUDA > MPS > CPU (can override with DEVICE env var)
DEVICE = os.getenv("DEVICE")
if DEVICE:
    print(f"Using device from environment: {DEVICE}")
else:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Auto-detected device: {DEVICE}")

RECORD_DIR      = "data/records"
AUDIO_DIR       = "data/audio"
ADAPTER_OUT_DIR = "data/lora_output"
ITERATION_FILE  = "data/training_iteration.txt"

# Feature flag: Enable/disable differential privacy
# Set ENABLE_DIFFERENTIAL_PRIVACY=false to disable DP (for CISK experiment)
ENABLE_DP = os.getenv("ENABLE_DIFFERENTIAL_PRIVACY", "false").lower() == "true"
print(f"Differential Privacy: {'ENABLED' if ENABLE_DP else 'DISABLED'}")

BATCH_SIZE      = 4
EPOCHS          = 3
MAX_GRAD_NORM   = 1.0

# Best-practice hyperparameters for production (grid search moved to separate script)
LEARNING_RATE   = 5e-5
TARGET_EPSILON  = 5.0
TARGET_DELTA    = 1e-5
LORA_R          = 8
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# Set to True to run hyperparameter search (for research/tuning only)
RUN_GRID_SEARCH = os.getenv("RUN_GRID_SEARCH", "false").lower() == "true"

# Grid search parameters (only used if RUN_GRID_SEARCH=True)
if RUN_GRID_SEARCH:
    lrs            = [5e-5, 7.5e-5, 1e-4]
    target_epsilons= [3.0, 6.0]
    lora_rs        = [4, 8]
    lora_alphas    = [16, 32]
    lora_dropouts  = [0.05, 0.1]
else:
    lrs            = [LEARNING_RATE]
    target_epsilons= [TARGET_EPSILON]
    lora_rs        = [LORA_R]
    lora_alphas    = [LORA_ALPHA]
    lora_dropouts  = [LORA_DROPOUT]

def get_training_iteration():
    """Get current training iteration number."""
    if Path(ITERATION_FILE).exists():
        try:
            return int(Path(ITERATION_FILE).read_text().strip())
        except:
            return 0
    return 0

def increment_training_iteration():
    """Increment and return the training iteration number."""
    iteration = get_training_iteration() + 1
    Path(ITERATION_FILE).write_text(str(iteration))
    return iteration

def collate_fn(batch, processor, bos_id):
    input_features = [{"input_features": b["input_features"]} for b in batch]
    batch_inp = processor.feature_extractor.pad(
        input_features, return_tensors="pt"
    )

    label_feats = [{"input_ids": b["labels"]} for b in batch]
    labels_batch = processor.tokenizer.pad(
        label_feats, return_tensors="pt"
    )
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )
    if (labels[:, 0] == bos_id).all().cpu().item():
        labels = labels[:, 1:]
    batch_inp["labels"] = labels
    return batch_inp["input_features"], batch_inp["labels"]


def map_sample(batch):
    # Use soundfile directly to avoid torchaudio 2.9+ breaking changes
    wav, sr = sf.read(os.path.join(AUDIO_DIR, batch["audio_file"]))
    wav = torch.from_numpy(wav).float()
    
    # Ensure we have the right shape [channels, samples]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T  # soundfile returns [samples, channels], we need [channels, samples]
    
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Whisper: input_features, labels
    batch["input_features"] = processor(
        wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["manual_transcript"], return_tensors="pt").input_ids[0]
    return batch


def evaluate_and_log(model, processor, records, out_dir):
    def transcribe(path):
        # Use soundfile directly to avoid torchaudio 2.9+ breaking changes
        wav, sr = sf.read(str(path))
        wav = torch.from_numpy(wav).float()
        
        # Ensure we have the right shape [channels, samples]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T  # soundfile returns [samples, channels], we need [channels, samples]
        
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr,16000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        feats = processor(wav.squeeze().numpy(), sampling_rate=16000,
                          return_tensors="pt").input_features.to(DEVICE)
        ids = model.generate(
            feats,
            forced_decoder_ids=processor.get_decoder_prompt_ids(task="transcribe",
                                                                language="no")
        )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    preds, refs = [], []
    for r in records:
        preds.append(transcribe(os.path.join(AUDIO_DIR, r["audio_file"])))
        refs.append(r["manual_transcript"])

    wer = jiwer.wer(refs, preds)
    
    # Write detailed WER report
    with open(Path(out_dir) / "wer.txt", "w") as f:
        from datetime import datetime
        f.write(f"=== WER Evaluation Report ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Number of samples: {len(records)}\n")
        f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
        f.write(f"\n--- Sample Predictions (first 3) ---\n")
        for i in range(min(3, len(records))):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"  Reference:  {refs[i]}\n")
            f.write(f"  Prediction: {preds[i]}\n")
    
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    return wer

def train_with_dp(
        model, loader, lr=5e-5, epochs=3,
        target_epsilon=5.0, target_delta=1e-5,
        max_grad_norm=1.0,
        device=DEVICE
):
    """
    Train model with optional differential privacy.
    If ENABLE_DP=False, uses standard training without privacy guarantees.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if ENABLE_DP:
        # Training WITH differential privacy
        print(f"Training with DP (ε={target_epsilon}, δ={target_delta})")
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )
    else:
        # Training WITHOUT differential privacy (standard PEFT fine-tuning)
        print("Training without DP (standard fine-tuning)")

    for ep in range(epochs):
        running_loss = 0.0
        for input_features, labels in loader:
            input_features = input_features.to(device)
            labels = labels.to(device)
            out = model(input_features=input_features, labels=labels)
            loss = out.loss
            loss.backward()
            
            # Apply gradient clipping for non-DP training
            if not ENABLE_DP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        
        if ENABLE_DP:
            eps = privacy_engine.get_epsilon(target_delta)
            print(f"Epoch {ep+1}/{epochs}  |  Loss {avg_loss:.4f}  |  ε={eps:.2f}")
            mlflow.log_metric("epsilon", eps, step=ep+1)
        else:
            print(f"Epoch {ep+1}/{epochs}  |  Loss {avg_loss:.4f}")
        
        # Log loss per epoch to MLflow
        mlflow.log_metric("loss", avg_loss, step=ep+1)

    # Return epsilon if DP is enabled, otherwise return None
    if ENABLE_DP:
        return privacy_engine.get_epsilon(target_delta)
    else:
        return None

grid = list(product(lrs, target_epsilons,
                    lora_rs, lora_alphas, lora_dropouts))
print(f"{len(grid)} configs to run")

# Set MLflow experiment name based on DP mode
experiment_name = "CISK-PEFT-FineTuning" if not ENABLE_DP else "Whisper-LoRA-DP"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

warnings.filterwarnings("ignore", category=UserWarning)

print("Loading Whisper processor and base model...")
processor  = WhisperProcessor.from_pretrained(MODEL_NAME)
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

bos_id = base_model.generation_config.decoder_start_token_id
torch_collate = partial(collate_fn, processor=processor, bos_id=bos_id)

# -------- gather data ----------
records = []
for f in os.listdir(RECORD_DIR):
    if f.endswith(".json"):
        d = json.load(open(os.path.join(RECORD_DIR,f)))
        if d.get("manual_transcript","").strip():
            records.append(d)

if not records:
    raise RuntimeError("No manually-reviewed records to train on.")


ds  = Dataset.from_list(records).map(map_sample)
ds = ds.remove_columns([c for c in ds.column_names if c not in ("input_features", "labels")])

train_loader = DataLoader(
    ds,     
    batch_size=BATCH_SIZE,         
    shuffle=True,
    collate_fn=torch_collate,
)

# Get training iteration for tracking
training_iteration = increment_training_iteration()
print(f"\n{'='*60}")
print(f"TRAINING ITERATION #{training_iteration}")
print(f"{'='*60}\n")

for i, (lr, eps, r, alpha, dropout) in enumerate(grid, 1):
    # Clean up previous model to free memory
    if i > 1:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    # Better run naming for single runs
    if not RUN_GRID_SEARCH:
        run_name = f"iter_{training_iteration:03d}_{len(records)}_samples"
    else:
        run_name = f"run_{i:02d}_lr{lr}_eps{eps}_r{r}_a{alpha}_d{dropout}"
    
    adapter_dir = Path(ADAPTER_OUT_DIR) / run_name if RUN_GRID_SEARCH else Path(ADAPTER_OUT_DIR)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg = LoraConfig(
        r           = r,
        lora_alpha  = alpha,
        lora_dropout= dropout,
        target_modules=["q_proj","v_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    with mlflow.start_run(run_name=run_name):
        # --- Log hyperparameters and system info
        params = {
            "lr"            : lr,
            "lora_r"        : r,
            "lora_alpha"    : alpha,
            "lora_dropout"  : dropout,
            "epochs"        : EPOCHS,
            "batch_size"    : BATCH_SIZE,
            "max_grad_norm" : MAX_GRAD_NORM,
            "differential_privacy": ENABLE_DP,
            "device"        : DEVICE,
            "num_train_samples": len(records),
            "model_name"    : MODEL_NAME,
            "training_iteration": training_iteration
        }
        if ENABLE_DP:
            params["target_eps"] = eps
            params["target_delta"] = TARGET_DELTA
        mlflow.log_params(params)

        # Log training metadata
        from datetime import datetime
        mlflow.set_tag("training_date", datetime.now().isoformat())
        mlflow.set_tag("experiment_type", "CISK" if not ENABLE_DP else "DP")
        mlflow.set_tag("model_type", "LoRA-Whisper")

        print(f"\n=== [{run_name}] training ===")
        print(f"Training samples: {len(records)}")
        print(f"Device: {DEVICE}")
        
        final_eps = train_with_dp(
            model=model,
            loader=train_loader,
            lr=lr,
            epochs=EPOCHS,
            target_epsilon=eps,
            max_grad_norm=MAX_GRAD_NORM,
        )

        if ENABLE_DP and final_eps is not None:
            print(f"Training finished.  (ε, δ)=({final_eps:.2f}, 1e-5)")
            mlflow.log_metric("final_epsilon", final_eps)
        else:
            print("Training finished (no DP).")
        
        print("=== saving adapter ===")
        model.save_pretrained(adapter_dir)
        wer = evaluate_and_log(model, processor, records, adapter_dir) 

        # Log final metrics
        mlflow.log_metric("WER", wer)
        mlflow.log_metric("num_train_samples", len(records))
        mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")

        # Ping backend to hot-reload the adapter
        try:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            requests.post(f"{backend_url}/asr/reload_adapter", timeout=5)
            print("✓ Successfully notified backend to reload adapter")
        except Exception as e:
            print(f"⚠ Warning: Could not reload adapter in backend: {e}")
        
        # Print summary for students
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model: {MODEL_NAME}")
        print(f"Training Mode: {'Standard PEFT (no DP)' if not ENABLE_DP else 'PEFT with Differential Privacy'}")
        print(f"Training Samples: {len(records)}")
        print(f"Final WER: {wer:.4f} ({wer*100:.2f}%)")
        if ENABLE_DP and final_eps:
            print(f"Privacy Budget: ε={final_eps:.2f}, δ={TARGET_DELTA}")
        print(f"Adapter saved to: {adapter_dir}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print("="*60)

print("\n✓ Training complete!")
if not RUN_GRID_SEARCH:
    print(f"\n📊 View results in MLflow UI:")
    print(f"   mlflow ui --port 5000")
    print(f"   Then open: http://localhost:5000")
    print(f"\n📁 Adapter location: {ADAPTER_OUT_DIR}")
    print("🔄 Backend has been notified to reload the model.")
