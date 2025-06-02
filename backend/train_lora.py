"""
Trains / updates LoRA from reviewed chunks,
writes WER, pings backend to reload.
"""

import os, json, torch, requests, warnings
from datasets import Dataset
import torchaudio, jiwer, torch
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

MODEL_NAME      = "NbAiLab/nb-whisper-tiny"
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"
RECORD_DIR      = "data/records"
AUDIO_DIR       = "data/audio"
ADAPTER_OUT_DIR = "data/lora_output"

BATCH_SIZE      = 4
EPOCHS          = 4 
MAX_GRAD_NORM  = 1.0

lrs            = [5e-5, 7.5e-5, 1e-4]
target_epsilons= [3.0, 6.0]
lora_rs        = [4, 8]
lora_alphas    = [16, 32]
lora_dropouts  = [0.05, 0.1]

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
    import torchaudio
    wav, sr = torchaudio.load(os.path.join(AUDIO_DIR, batch["audio_file"]))
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
        wav, sr = torchaudio.load(path)
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
    with open(Path(out_dir) / "wer.txt", "w") as f:
        f.write(f"WER: {wer:.4f}\n")
    return wer

def train_with_dp(
        model, loader, lr=5e-5, epochs=3,
        target_epsilon=5.0, target_delta=1e-5,
        max_grad_norm=1.0,
        device=DEVICE
):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

    for ep in range(epochs):
        running_loss = 0.0
        for input_features, labels in loader:
            input_features = input_features.to(device)
            labels = labels.to(device)
            out = model(input_features=input_features, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        eps = privacy_engine.get_epsilon(target_delta)
        avg_loss = running_loss / len(loader)
        print(f"Epoch {ep+1}/{epochs}  |  Loss {avg_loss:.4f}  |  ε={eps:.2f}")
        
        # Log loss and epsilon per epoch to MLflow
        mlflow.log_metric("loss", avg_loss, step=ep+1)
        mlflow.log_metric("epsilon", eps, step=ep+1)


    return privacy_engine.get_epsilon(target_delta)

grid = list(product(lrs, target_epsilons,
                    lora_rs, lora_alphas, lora_dropouts))
print(f"{len(grid)} configs to run")

mlflow.set_experiment("Whisper LoRA DP Tuning")

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

for i, (lr, eps, r, alpha, dropout) in enumerate(grid, 1):
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    run_name = f"gs_{i:02d}_lr{lr}_eps{eps}_r{r}_a{alpha}_d{dropout}"
    adapter_dir = Path(ADAPTER_OUT_DIR) / run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg = LoraConfig(
        r           = r,
        lora_alpha  = alpha,
        lora_dropout= dropout,
        target_modules=["q_proj","v_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    with mlflow.start_run():
        # --- Log hyperparameters
        mlflow.log_params({
            "lr"            : lr,
            "target_eps"    : eps,
            "lora_r"        : r,
            "lora_alpha"    : alpha,
            "lora_dropout"  : dropout,
            "epochs"        : EPOCHS,
            "batch_size"    : BATCH_SIZE,
            "max_grad_norm" : MAX_GRAD_NORM
        })

        print(f"\n=== [{run_name}] training ===")
        final_eps = train_with_dp(
            model=model,
            loader=train_loader,
            lr=lr,
            epochs=EPOCHS,
            target_epsilon=eps,
            max_grad_norm=MAX_GRAD_NORM,
        )

        print(f"Training finished.  (ε, δ)=({final_eps:.2f}, 1e-5)")
        print("=== saving adapter ===")
        model.save_pretrained(adapter_dir)
        wer = evaluate_and_log(model, processor, records, adapter_dir) 

        mlflow.log_metric("final_epsilon", final_eps)
        mlflow.log_metric("WER", wer)
        mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")

        # ---- ping backend to hot-reload ----
        # try:
        #     requests.post("http://localhost:8000/asr/reload_adapter", timeout=3)
        # except Exception as e:
        #     print("reload ping failed:", e)
