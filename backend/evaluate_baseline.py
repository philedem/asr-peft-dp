"""
Evaluation script for measuring baseline ASR performance.
Run this before fine-tuning to establish baseline metrics.
"""

import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
from tqdm import tqdm

MODEL_NAME = "NbAiLab/nb-whisper-medium"

# Device detection: prioritize CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Evaluating on device: {DEVICE}")

def load_test_set(test_file: str):
    """Load test dataset with ground truth transcriptions."""
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def transcribe_audio(audio_path: str, processor, model):
    """Transcribe a single audio file."""
    # Use soundfile directly to avoid torchaudio 2.9+ breaking changes
    wav, sr = sf.read(audio_path)
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
        ),
    )
    
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def evaluate_model(test_set, model, processor):
    """Evaluate model on test set and calculate WER."""
    predictions = []
    references = []
    
    print(f"Evaluating on {len(test_set)} samples...")
    
    for item in tqdm(test_set):
        audio_path = item['audio_file']
        reference = item['ground_truth']
        
        prediction = transcribe_audio(audio_path, processor, model)
        predictions.append(prediction)
        references.append(reference)
    
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    
    return {
        'wer': wer,
        'cer': cer,
        'predictions': predictions,
        'references': references
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate baseline ASR model')
    parser.add_argument('--test-set', required=True, help='Path to test set JSON')
    parser.add_argument('--output', default='baseline_results.json', help='Output file')
    args = parser.parse_args()
    
    print(f"Loading model: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    
    print(f"Loading test set: {args.test_set}")
    test_set = load_test_set(args.test_set)
    
    results = evaluate_model(test_set, model, processor)
    
    print(f"\n=== Baseline Results ===")
    print(f"Word Error Rate (WER): {results['wer']:.2%}")
    print(f"Character Error Rate (CER): {results['cer']:.2%}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
