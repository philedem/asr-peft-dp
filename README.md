# ASR-PEFT-DP: Automatic Speech Recognition with Privacy-Preserving Fine-Tuning

A production-ready framework for transcribing and annotating audio (military communications, medical dictations, etc.) with **continuous model improvement** through **LoRA (Low-Rank Adaptation)** and **Differential Privacy**.

![System Architecture](https://img.shields.io/badge/Framework-ASR--PEFT--DP-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

This system enables operators to:
1. **Upload or record audio** (files or live streams)
2. **Automatically transcribe** using Whisper ASR
3. **Review and correct** transcriptions through an intuitive web interface
4. **Automatically retrain** the model using corrections (with differential privacy)
5. **Track model performance** via Word Error Rate (WER) metrics

### Key Features

- ✅ **Automatic Chunking**: Splits long audio into manageable segments
- ✅ **Real-time Transcription**: Norwegian Whisper model (NbAiLab/nb-whisper-medium)
- ✅ **Parameter-Efficient Fine-Tuning**: LoRA adapters for lightweight updates
- ✅ **Differential Privacy**: Opacus integration for privacy-preserving training
- ✅ **MLflow Tracking**: Experiment tracking and model versioning
- ✅ **Hot-Reload**: Model updates without system restart
- ✅ **Docker Deployment**: One-command setup

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- At least **8GB RAM** (16GB recommended for training)
- **GPU support** optional but recommended:
  - **NVIDIA GPU**: CUDA support (fastest)
  - **Apple Silicon**: MPS support (M1/M2/M3 Macs)
  - **CPU only**: Works but slower

**Note**: The system automatically detects and uses the best available device (CUDA > MPS > CPU).

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/asr-peft-dp
   ```

2. **Configure environment variables**
   ```bash
   # Backend configuration
   cp backend/.env.example backend/.env
   
   # Frontend configuration  
   cp frontend/.env.example frontend/.env
   ```

3. **Build and start services**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

5. **Verify device detection**
   ```bash
   # Check what device is being used
   curl http://localhost:8000/asr/device_info
   
   # Or run the test script
   docker exec asr-peft-dp-backend-1 python test_device.py
   ```
   
   You should see:
   - `"device": "cuda"` on systems with NVIDIA GPUs
   - `"device": "mps"` on Apple Silicon Macs (M1/M2/M3)
   - `"device": "cpu"` on systems without GPU support

---

## 📖 User Guide

### Recording & Transcription

#### Method 1: Live Recording
1. Click **"🎤 Start Recording"** in the web interface
2. Speak into your microphone (military comms, dictation, etc.)
3. Click **"⏹ Stop Recording"** when done
4. System automatically chunks and transcribes audio

#### Method 2: File Upload
1. Click **"📁 Upload Audio"**
2. Select audio file (WAV, MP3, OGG, etc.)
3. System processes and transcribes

### Annotation Workflow

1. **Review transcriptions** in the table
2. **Edit text** directly in the textarea (auto-saves on blur)
3. **Approve correct transcriptions** with the "✓ Approve" button
4. **Track progress**: Yellow rows = needs review, White rows = approved

### Automatic Model Training

- System automatically retrains after **20 corrections**
- Manual trigger available: **"🔄 Manual Retrain"** button
- Training runs in background (~5-15 minutes depending on data)
- Model hot-reloads automatically when training completes

### Performance Monitoring

- **WER (Word Error Rate)**: Displayed in top dashboard
- Lower WER = better model accuracy
- Updates after each training cycle

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend    │─────▶│  Whisper    │
│  (Svelte)   │      │  (FastAPI)   │      │    Model    │
└─────────────┘      └──────────────┘      └─────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │  Training    │
                     │  (LoRA+DP)   │
                     └──────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   MLflow     │
                     │  Tracking    │
                     └──────────────┘
```

### Technology Stack

**Backend:**
- FastAPI (REST API)
- PyTorch + Transformers (Whisper)
- PEFT (LoRA adapters)
- Opacus (Differential Privacy)
- MLflow (Experiment tracking)
- Pydub (Audio processing)

**Frontend:**
- SvelteKit (UI framework)
- Vite (Build tool)
- TypeScript

**Infrastructure:**
- Docker & Docker Compose
- Uvicorn (ASGI server)

---

## ⚙️ Configuration

### Backend Settings (`backend/.env`)

```bash
# Model Configuration
BASE_MODEL=NbAiLab/nb-whisper-medium  # Whisper model variant

# Device Configuration (auto-detects best available)
# Uncomment to manually override: DEVICE=cuda | mps | cpu
# DEVICE=cuda                          # Force NVIDIA GPU
# DEVICE=mps                           # Force Apple Silicon
# DEVICE=cpu                           # Force CPU (slowest)

# Audio Processing
MIN_SILENCE_MS=2000                    # Silence threshold for chunking
MAX_CHUNK_LENGTH_MS=30000              # Max chunk duration (30s)

# Training Configuration
MANUAL_RETRAIN_N=20                    # Corrections before auto-retrain
BATCH_SIZE=4                           # Training batch size
EPOCHS=3                               # Training epochs
LEARNING_RATE=5e-5                     # Learning rate

# Differential Privacy
TARGET_EPSILON=5.0                     # Privacy budget (lower = more private)
TARGET_DELTA=1e-5                      # Privacy parameter

# LoRA Configuration
LORA_R=8                               # LoRA rank
LORA_ALPHA=32                          # LoRA alpha
LORA_DROPOUT=0.05                      # LoRA dropout

# Grid Search (for hyperparameter tuning)
RUN_GRID_SEARCH=false                  # Set to 'true' for research mode
```

### Frontend Settings (`frontend/.env`)

```bash
VITE_BACKEND_URL=http://localhost:8000
VITE_WER_POLL_INTERVAL=70000           # WER refresh interval (ms)
```

---

## 🔬 Research & Evaluation

### For Bachelor Students: Evaluation Protocol

#### Metrics to Track

1. **Word Error Rate (WER)**
   - Baseline: Test with pre-trained model
   - After N corrections: Measure improvement
   - Plot WER over time

2. **Privacy Guarantees**
   - Epsilon (ε) values from training logs
   - Lower ε = stronger privacy

3. **User Experience**
   - Time per annotation
   - Correction rate
   - Interface usability

4. **Model Performance**
   - Transcription accuracy on military jargon
   - Domain adaptation effectiveness
   - Training convergence speed

#### Experiment Design

```python
# Example evaluation script
# 1. Collect baseline metrics
python evaluate_baseline.py --test-set military_comms.json

# 2. Run annotation session
# (Use web interface for 50+ samples)

# 3. Trigger training
# (Automatic after 20 corrections)

# 4. Measure improvement
python evaluate_finetuned.py --test-set military_comms.json

# 5. Compare results
python compare_results.py --baseline baseline.json --finetuned finetuned.json
```

### MLflow Experiment Tracking

Access MLflow UI:
```bash
docker exec -it asr-peft-dp-backend-1 mlflow ui --host 0.0.0.0 --port 5000
```

Then open: http://localhost:5000

Tracked metrics:
- Loss per epoch
- Final epsilon (ε)
- WER after training
- Hyperparameters

### Hyperparameter Tuning

Enable grid search mode:
```bash
# In backend/.env
RUN_GRID_SEARCH=true
```

Then run training:
```bash
docker exec -it asr-peft-dp-backend-1 python train_lora.py
```

This will test multiple combinations of:
- Learning rates: [5e-5, 7.5e-5, 1e-4]
- Target epsilons: [3.0, 6.0]
- LoRA ranks: [4, 8]
- LoRA alphas: [16, 32]
- Dropout rates: [0.05, 0.1]

Results saved to MLflow for comparison.

---

## 🗂️ Data Management

### Directory Structure

```
backend/data/
├── audio/              # Stored audio chunks (.wav)
├── records/            # Transcription records (.json)
├── lora_output/        # Trained LoRA adapters
├── manual_review_count.txt   # Correction counter
└── wer.txt             # Latest WER score
```

### Record Format

```json
{
  "audio_file": "abc123_0.wav",
  "asr_transcript": "original transcription",
  "manual_transcript": "corrected transcription",
  "timestamp": "2025-11-15T10:30:00Z"
}
```

### Exporting Data

```bash
# Backup all data
docker cp asr-peft-dp-backend-1:/workspace/data ./backup_data

# Export specific dataset
docker exec asr-peft-dp-backend-1 python export_dataset.py --output dataset.json
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "No module named 'torch'"
**Solution:** Dependencies not installed. Rebuild containers:
```bash
docker-compose down
docker-compose up --build
```

#### Issue: "Could not load records. Is the backend running?"
**Solution:** Check backend logs:
```bash
docker logs asr-peft-dp-backend-1
```

#### Issue: Training fails with OOM (Out of Memory)
**Solutions:**
- Reduce `BATCH_SIZE` in `.env` (try 2 or 1)
- Use smaller model: `NbAiLab/nb-whisper-tiny`
- Increase Docker memory limit

#### Issue: Model not reloading after training
**Solution:** Manually trigger reload:
```bash
curl -X POST http://localhost:8000/asr/reload_adapter
```

#### Issue: High WER (poor transcription quality)
**Causes:**
- Insufficient training data (need >50 corrections)
- Domain mismatch (military jargon vs general speech)
- Audio quality issues (noise, compression)

**Solutions:**
- Annotate more samples
- Increase `EPOCHS` for more training
- Preprocess audio (noise reduction)

#### Issue: MPS (Apple Silicon) not being used
**Solution:** Check MPS availability:
```bash
docker exec asr-peft-dp-backend-1 python test_device.py
```

If MPS is available but not being used, try:
```bash
# Force MPS in backend/.env
echo "DEVICE=mps" >> backend/.env
docker-compose restart backend
```

#### Issue: CUDA out of memory
**Solutions:**
- Reduce `BATCH_SIZE` in `.env` (try 2 or 1)
- Use smaller model: `NbAiLab/nb-whisper-tiny`
- Clear CUDA cache: `torch.cuda.empty_cache()`

#### Issue: Slow performance on CPU
**Cause:** No GPU detected

**Solutions:**
- Use a machine with NVIDIA GPU or Apple Silicon
- Reduce model size to `nb-whisper-tiny`
- Process smaller batches
- Be patient (CPU mode is 5-10x slower)

---

## 🔒 Security & Privacy

### Differential Privacy

- **Epsilon (ε)**: Privacy budget
  - Lower = more privacy, less accuracy
  - Typical: ε=3-10 for sensitive data
  - Current default: ε=5.0

- **Delta (δ)**: Probability of privacy breach
  - Set to 1e-5 (very low)

### Best Practices

1. **Data Isolation**: Keep audio/transcripts in secure storage
2. **Access Control**: Implement authentication (not included in demo)
3. **Audit Logging**: Track all corrections and retraining events
4. **Model Versioning**: MLflow maintains full model history

---

## 📊 Performance Benchmarks

### Expected Performance (nb-whisper-medium)

| Stage | WER | Notes |
|-------|-----|-------|
| Baseline | 15-25% | Pre-trained on Norwegian |
| After 50 corrections | 10-18% | Domain adaptation |
| After 200 corrections | 8-15% | Significant improvement |

### System Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 4 cores | 8+ cores | 16+ cores |
| RAM | 8 GB | 16 GB | 32 GB |
| GPU | None (CPU mode) | Apple Silicon (MPS) | NVIDIA RTX 3060+ (CUDA) |
| Storage | 10 GB | 50 GB+ | 100 GB+ |

### Device Performance Comparison

| Device Type | Transcription Speed | Training Speed | Notes |
|-------------|-------------------|----------------|-------|
| NVIDIA GPU (CUDA) | ~2-5 sec/min | ~10-15 min/100 samples | ⚡ Fastest, optimal for production |
| Apple Silicon (MPS) | ~5-10 sec/min | ~15-25 min/100 samples | ✅ Good for development, M1/M2/M3 |
| CPU Only | ~20-40 sec/min | ~30-60 min/100 samples | 🐌 Slowest, for testing only |

*Speeds based on nb-whisper-medium model*

---

## 🛠️ Development

### Running Tests

```bash
# Backend tests
docker exec asr-peft-dp-backend-1 pytest tests/

# Frontend tests  
docker exec asr-peft-dp-frontend-1 npm test
```

### API Endpoints

- `POST /asr/transcribe` - Upload audio for transcription
- `POST /asr/save_record` - Save corrected transcription
- `GET /asr/records` - List all records
- `GET /asr/wer` - Get current WER
- `GET /audio/{filename}` - Stream audio file
- `POST /asr/reload_adapter` - Hot-reload model
- `GET /train/retrain_lora` - Trigger manual training

Full API docs: http://localhost:8000/docs

---

## 📚 References

- [Whisper Paper](https://arxiv.org/abs/2212.04356) - OpenAI's ASR model
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning
- [Opacus](https://github.com/pytorch/opacus) - PyTorch differential privacy
- [NbAiLab Whisper](https://huggingface.co/NbAiLab) - Norwegian Whisper models

---

## 👥 Contributors

Developed for bachelor student evaluation in military communications ASR.

## 📄 License

MIT License - See LICENSE file for details

---

## 🆘 Support

For issues during evaluation:
1. Check logs: `docker-compose logs`
2. Review troubleshooting section above
3. Contact: [Your contact information]

**Good luck with your evaluation! 🎓**
