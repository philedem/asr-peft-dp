#!/usr/bin/env python3
"""
Pre-download and cache Whisper models on container startup
"""
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODELS = [
    "NbAiLab/nb-whisper-tiny",
    "NbAiLab/nb-whisper-small",
    "NbAiLab/nb-whisper-medium"
]

def cache_models():
    """Download and cache all models"""
    print("=" * 60)
    print("🔄 Caching Whisper Models")
    print("=" * 60)
    
    for model_name in MODELS:
        print(f"\n📥 Downloading {model_name}...")
        try:
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            print(f"✅ {model_name} cached successfully")
        except Exception as e:
            print(f"❌ Failed to cache {model_name}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model caching complete!")
    print("=" * 60)

if __name__ == "__main__":
    cache_models()
