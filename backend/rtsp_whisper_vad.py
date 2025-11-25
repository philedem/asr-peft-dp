#!/usr/bin/env python3
"""
RTSP to Whisper with Voice Activity Detection (VAD)
Transcribes on silence (>2s) or max duration (10s)
"""
import subprocess
import tempfile
import os
import sys
import time
import numpy as np
from pathlib import Path

# Environment variables
RTSP_URL = os.getenv("RTSP_URL", "rtsp://rtsp-test-server:8554/test")
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "10"))  # Max duration
MODEL_NAME = os.getenv("MODEL_NAME", "NbAiLab/nb-whisper-small")
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.02"))  # RMS threshold for silence
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "2.0"))  # Seconds of silence to trigger
SAMPLE_RATE = 16000

print(f"""
{'='*60}
RTSP Transcription with VAD
{'='*60}
RTSP URL: {RTSP_URL}
Max Segment: {SEGMENT_DURATION}s
Min Silence: {SILENCE_DURATION}s
Silence Threshold: {SILENCE_THRESHOLD}
Model: {MODEL_NAME}
{'='*60}
""")

def calculate_rms(audio_data):
    """Calculate RMS (Root Mean Square) energy of audio"""
    return np.sqrt(np.mean(audio_data**2))

def detect_silence_end(audio_data, sample_rate, silence_threshold, silence_duration):
    """
    Detect if there's silence at the end of audio data
    Returns (has_silence, silence_start_idx)
    """
    # Calculate RMS in small windows (0.1s)
    window_size = int(sample_rate * 0.1)
    num_windows = len(audio_data) // window_size
    
    if num_windows == 0:
        return False, len(audio_data)
    
    # Find consecutive silent windows
    silence_windows = 0
    silence_start = len(audio_data)
    
    for i in range(num_windows - 1, -1, -1):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = audio_data[start_idx:end_idx]
        rms = calculate_rms(window_data)
        
        if rms < silence_threshold:
            silence_windows += 1
            silence_start = start_idx
        else:
            break
    
    silence_duration_actual = silence_windows * 0.1
    has_enough_silence = silence_duration_actual >= silence_duration
    
    return has_enough_silence, silence_start

def capture_audio_streaming(rtsp_url, max_duration):
    """
    Capture audio from RTSP stream, monitoring for silence
    Returns when: silence detected OR max duration reached
    """
    try:
        import soundfile as sf
        
        # Use ffmpeg to stream audio to stdout
        cmd = [
            'ffmpeg',
            '-i', rtsp_url,
            '-f', 's16le',  # Raw PCM
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-'  # Output to stdout
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        audio_chunks = []
        start_time = time.time()
        bytes_per_sample = 2  # 16-bit = 2 bytes
        chunk_size = SAMPLE_RATE * bytes_per_sample * 1  # 1 second chunks
        
        print(f"📡 Capturing audio (max {max_duration}s, listening for {SILENCE_DURATION}s silence)...")
        
        while True:
            # Read chunk
            chunk_bytes = process.stdout.read(chunk_size)
            if not chunk_bytes:
                break
            
            # Convert bytes to numpy array
            chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunks.append(chunk_array)
            
            elapsed = time.time() - start_time
            
            # Check for max duration
            if elapsed >= max_duration:
                print(f"⏱️  Max duration ({max_duration}s) reached")
                process.terminate()
                break
            
            # Check for silence (only after 2 seconds minimum)
            if elapsed >= 2.0:
                full_audio = np.concatenate(audio_chunks)
                has_silence, silence_idx = detect_silence_end(
                    full_audio, SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION
                )
                
                if has_silence:
                    silence_duration_actual = (len(full_audio) - silence_idx) / SAMPLE_RATE
                    print(f"🔇 Silence detected ({silence_duration_actual:.1f}s) - triggering transcription")
                    process.terminate()
                    break
        
        process.wait()
        
        if not audio_chunks:
            return None
        
        # Combine all chunks
        full_audio = np.concatenate(audio_chunks)
        actual_duration = len(full_audio) / SAMPLE_RATE
        
        print(f"✅ Audio captured: {actual_duration:.1f}s, {len(full_audio)} samples")
        return full_audio
        
    except Exception as e:
        print(f"❌ Error capturing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def transcribe_audio(audio_data):
    """Transcribe audio using Whisper"""
    try:
        import torch
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
        
        # Transcribe
        feats = transcribe_audio.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
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
    """Main loop - capture with VAD and transcribe"""
    segment_count = 0
    
    while True:
        segment_count += 1
        
        print(f"\n{'─'*60}")
        print(f"Segment #{segment_count} - {time.strftime('%H:%M:%S')}")
        print(f"{'─'*60}")
        
        audio_data = capture_audio_streaming(RTSP_URL, SEGMENT_DURATION)
        
        if audio_data is None or len(audio_data) == 0:
            print("⚠️  No audio captured, waiting 5s before retry...")
            time.sleep(5)
            continue
        
        print(f"🎤 Transcribing...")
        transcript = transcribe_audio(audio_data)
        
        if transcript:
            print(f"\n{'='*60}")
            print(f"📝 TRANSCRIPT:")
            print(f"   {transcript}")
            print(f"{'='*60}\n")
        else:
            print(f"⚠️  No transcript generated")

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
