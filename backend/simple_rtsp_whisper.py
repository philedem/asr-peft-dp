#!/usr/bin/env python3
"""
Simple RTSP to Whisper PoC - reads RTSP stream, extracts audio, transcribes
Uses ffmpeg for RTSP handling (simpler than WhisperLive for initial PoC)
"""
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path

# Environment variables
RTSP_URL = os.getenv("RTSP_URL", "rtsp://rtsp-test-server:8554/test")
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "10"))  # seconds
MODEL_NAME = os.getenv("MODEL_NAME", "NbAiLab/nb-whisper-small")

print(f"""
{'='*60}
Simple RTSP Transcription PoC
{'='*60}
RTSP URL: {RTSP_URL}
Segment Duration: {SEGMENT_DURATION}s
Model: {MODEL_NAME}
{'='*60}
""")

def capture_audio_segment(rtsp_url, duration, output_file):
    """Capture audio segment from RTSP stream using ffmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', rtsp_url,
            '-t', str(duration),
            '-vn',  # no video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=duration + 10
        )
        
        if result.returncode != 0:
            print(f"❌ ffmpeg error: {result.stderr.decode()}")
            return False
            
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout capturing audio segment")
        return False
    except Exception as e:
        print(f"❌ Error capturing audio: {e}")
        return False

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper"""
    try:
        import torch
        import soundfile as sf
        import numpy as np
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Load model (only once)
        if not hasattr(transcribe_audio, 'model'):
            print("Loading Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            transcribe_audio.processor = WhisperProcessor.from_pretrained(MODEL_NAME)
            transcribe_audio.model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
            transcribe_audio.device = device
            print("Model loaded!")
        
        # Load audio with soundfile
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple resampling
            duration = len(audio_data) / sample_rate
            target_length = int(duration * 16000)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000
        
        # Transcribe
        feats = transcribe_audio.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(transcribe_audio.device)
        
        ids = transcribe_audio.model.generate(
            feats,
            forced_decoder_ids=transcribe_audio.processor.get_decoder_prompt_ids(
                task="transcribe",
                language="no"
            )
        )
        
        text = transcribe_audio.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main loop - capture and transcribe"""
    segment_count = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            segment_count += 1
            audio_file = os.path.join(tmpdir, f"segment_{segment_count}.wav")
            
            print(f"\n{'─'*60}")
            print(f"Segment #{segment_count} - {time.strftime('%H:%M:%S')}")
            print(f"{'─'*60}")
            print(f"📡 Capturing {SEGMENT_DURATION}s audio from RTSP...")
            
            if not capture_audio_segment(RTSP_URL, SEGMENT_DURATION, audio_file):
                print("⚠️  Failed to capture audio, waiting 5s before retry...")
                time.sleep(5)
                continue
            
            print(f"✅ Audio captured: {os.path.getsize(audio_file)} bytes")
            print(f"🎤 Transcribing...")
            
            transcript = transcribe_audio(audio_file)
            
            if transcript:
                print(f"\n{'='*60}")
                print(f"📝 TRANSCRIPT:")
                print(f"   {transcript}")
                print(f"{'='*60}\n")
            else:
                print(f"⚠️  No transcript generated")
            
            # Cleanup
            try:
                os.remove(audio_file)
            except:
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping transcription...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
