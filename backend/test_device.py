#!/usr/bin/env python3
"""
Device Detection Test Script
Tests PyTorch device availability and provides recommendations.
"""

import sys

try:
    import torch
except ImportError:
    print("❌ PyTorch not installed. Install it with: pip install torch")
    sys.exit(1)

print("🔍 PyTorch Device Detection")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print()

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
if cuda_available:
    print(f"  - Device Count: {torch.cuda.device_count()}")
    print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA Version: {torch.version.cuda}")
print()

# Check MPS (Apple Silicon)
mps_available = torch.backends.mps.is_available()
print(f"MPS Available: {'✅ Yes' if mps_available else '❌ No'}")
if mps_available:
    print(f"  - Apple Silicon detected (M1/M2/M3)")
    try:
        # Test MPS functionality
        test_tensor = torch.randn(10).to("mps")
        print(f"  - MPS Test: ✅ Working")
    except Exception as e:
        print(f"  - MPS Test: ⚠️ Available but not functional: {e}")
print()

# Determine best device
if cuda_available:
    recommended = "cuda"
    speed = "🚀 Fastest"
elif mps_available:
    recommended = "mps"
    speed = "⚡ Fast"
else:
    recommended = "cpu"
    speed = "🐌 Slowest"

print(f"Recommended Device: {recommended} ({speed})")
print()

# Auto-detection logic (same as application)
if cuda_available:
    auto_device = "cuda"
elif mps_available:
    auto_device = "mps"
else:
    auto_device = "cpu"

print(f"Application will auto-select: {auto_device}")
print()

# Performance estimates
print("⏱️  Estimated Performance (nb-whisper-medium):")
print("-" * 50)
if auto_device == "cuda":
    print("Transcription: ~2-5 seconds per minute of audio")
    print("Training: ~10-15 minutes for 100 samples")
elif auto_device == "mps":
    print("Transcription: ~5-10 seconds per minute of audio")
    print("Training: ~15-25 minutes for 100 samples")
else:
    print("Transcription: ~20-40 seconds per minute of audio")
    print("Training: ~30-60 minutes for 100 samples")
print()

# Recommendations
print("💡 Recommendations:")
print("-" * 50)
if not cuda_available and not mps_available:
    print("⚠️  No GPU detected. Training will be slow.")
    print("   Consider using a machine with NVIDIA GPU or Apple Silicon.")
elif auto_device == "mps":
    print("✅ Apple Silicon detected and will be used.")
    print("   Performance is good for development and small datasets.")
elif auto_device == "cuda":
    print("✅ NVIDIA GPU detected and will be used.")
    print("   Optimal performance for production workloads.")

print()
print("To override device selection, set DEVICE environment variable:")
print("  export DEVICE=cuda    # Force CUDA")
print("  export DEVICE=mps     # Force MPS")
print("  export DEVICE=cpu     # Force CPU")
