#!/usr/bin/env python3
"""
FastAPI backend for RTSP Live Transcription Frontend
Runs simple_rtsp_whisper.py as a subprocess with configurable parameters
"""
import subprocess
import time
import threading
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RTSP Transcription API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
class TranscriptionState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.transcripts: List[Dict] = []
        self.is_running = False
        self.config = {}
        self.start_time: Optional[float] = None
        self.segment_times: List[float] = []
        self.monitor_thread: Optional[threading.Thread] = None

state = TranscriptionState()

# Models
class StartRequest(BaseModel):
    rtsp_url: str
    segment_duration: int = 10
    model_name: str = "NbAiLab/nb-whisper-small"
    use_vad: bool = True  # Use Voice Activity Detection
    silence_duration: float = 2.0  # Seconds of silence to trigger transcription
    silence_threshold: float = 0.02  # RMS threshold for silence detection

class TranscriptItem(BaseModel):
    id: int
    timestamp: str
    text: str
    duration: float

# Transcript Parser
class TranscriptMonitor:
    """Monitors subprocess output for transcripts"""
    
    def __init__(self, state: TranscriptionState):
        self.state = state
        self.transcript_buffer = []
        self.in_transcript = False
    
    def parse_output(self, line: str) -> Optional[Dict]:
        """Parse a line of output looking for transcripts"""
        line = line.strip()
        
        # Start of transcript
        if "📝 TRANSCRIPT:" in line:
            self.in_transcript = True
            self.transcript_buffer = []
            return None
        
        # End of transcript block
        if "====" in line and self.in_transcript:
            self.in_transcript = False
            if self.transcript_buffer:
                text = " ".join(self.transcript_buffer).strip()
                if text:
                    return {
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    }
            self.transcript_buffer = []
            return None
        
        # Inside transcript block - collect non-empty lines
        if self.in_transcript and line and not line.startswith("="):
            self.transcript_buffer.append(line)
        
        return None
    
    def monitor_process_output(self, process: subprocess.Popen):
        """Monitor subprocess stdout for transcripts"""
        print("Starting output monitor for transcription process")
        
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                
                if not self.state.is_running:
                    print("Stopping monitor (is_running=False)")
                    break
                
                # Print to our stdout for debugging
                print(f"[PoC] {line.rstrip()}")
                
                transcript = self.parse_output(line)
                if transcript:
                    # Add transcript to state
                    transcript_item = {
                        "id": len(self.state.transcripts) + 1,
                        "timestamp": transcript["timestamp"],
                        "text": transcript["text"],
                        "duration": self.state.config.get("segment_duration", 10.0)
                    }
                    self.state.transcripts.append(transcript_item)
                    self.state.segment_times.append(time.time())
                    print(f"✓ Added transcript #{transcript_item['id']}: {transcript['text'][:60]}...")
                    
        except Exception as e:
            print(f"Error monitoring output: {e}")
        finally:
            print("Output monitor stopped")

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "is_running": state.is_running,
        "total_transcripts": len(state.transcripts)
    }

@app.post("/rtsp/start")
async def start_transcription(request: StartRequest):
    """Start RTSP transcription by running simple_rtsp_whisper.py"""
    if state.is_running:
        raise HTTPException(status_code=400, detail="Transcription already running")
    
    try:
        # Store config
        state.config = request.dict()
        state.is_running = True
        state.start_time = time.time()
        # Don't clear transcripts - let them persist across sessions
        # Only clear via explicit user action (clear button)
        state.segment_times = []
        
        print(f"Starting transcription with config: {state.config}")
        
        # Choose script based on VAD setting
        if request.use_vad:
            script_name = "rtsp_whisper_vad.py"
        else:
            script_name = "simple_rtsp_whisper.py"
        
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            raise HTTPException(status_code=500, detail=f"{script_name} not found")
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["RTSP_URL"] = request.rtsp_url
        env["SEGMENT_DURATION"] = str(request.segment_duration)
        env["MODEL_NAME"] = request.model_name
        
        # VAD-specific parameters
        if request.use_vad:
            env["SILENCE_DURATION"] = str(request.silence_duration)
            env["SILENCE_THRESHOLD"] = str(request.silence_threshold)
        
        # Start the transcription process
        state.process = subprocess.Popen(
            ["python3", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        print(f"Started process PID: {state.process.pid}")
        
        # Start monitoring the output in a separate thread
        monitor = TranscriptMonitor(state)
        state.monitor_thread = threading.Thread(
            target=monitor.monitor_process_output,
            args=(state.process,),
            daemon=True
        )
        state.monitor_thread.start()
        
        print("Monitor thread started")
        
        return {
            "status": "started",
            "config": state.config,
            "pid": state.process.pid,
            "message": "Transcription process started"
        }
    except Exception as e:
        state.is_running = False
        print(f"Error starting transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rtsp/stop")
async def stop_transcription():
    """Stop RTSP transcription"""
    if not state.is_running:
        raise HTTPException(status_code=400, detail="Transcription not running")
    
    print("Stopping transcription...")
    state.is_running = False
    
    # Stop the process
    if state.process:
        print(f"Terminating process PID: {state.process.pid}")
        state.process.terminate()
        
        # Wait for process to end
        try:
            state.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process didn't terminate, killing it")
            state.process.kill()
            state.process.wait()
        
        state.process = None
        print("Process terminated")
    
    # Wait for monitor thread to finish
    if state.monitor_thread:
        state.monitor_thread.join(timeout=2.0)
        state.monitor_thread = None
        print("Monitor thread stopped")
    
    return {
        "status": "stopped",
        "total_transcripts": len(state.transcripts)
    }

@app.get("/rtsp/transcripts")
async def get_transcripts():
    """Get all transcripts with statistics"""
    
    # Calculate statistics
    elapsed_time = time.time() - state.start_time if state.start_time else 0
    avg_transcription_time = 0.0
    
    if len(state.segment_times) > 1:
        intervals = [state.segment_times[i] - state.segment_times[i-1] 
                    for i in range(1, len(state.segment_times))]
        avg_transcription_time = sum(intervals) / len(intervals) if intervals else 0.0
    
    return {
        "transcripts": state.transcripts,
        "total": len(state.transcripts),
        "is_running": state.is_running,
        "elapsed_time": elapsed_time,
        "avg_transcription_time": avg_transcription_time
    }

@app.post("/rtsp/clear")
async def clear_transcripts():
    """Clear all transcripts"""
    state.transcripts = []
    state.segment_times = []
    return {
        "status": "cleared",
        "total_transcripts": 0
    }

@app.post("/rtsp/transcript")
async def add_transcript(transcript: TranscriptItem):
    """Manually add a transcript (for testing)"""
    state.transcripts.append(transcript.dict())
    return {"status": "added", "id": transcript.id}

@app.get("/rtsp/status")
async def get_status():
    """Get current transcription status"""
    return {
        "is_running": state.is_running,
        "config": state.config,
        "total_transcripts": len(state.transcripts),
        "start_time": state.start_time,
        "pid": state.process.pid if state.process else None
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting RTSP Transcription API...")
    print("This API will run simple_rtsp_whisper.py as a subprocess")
    uvicorn.run(app, host="0.0.0.0", port=8000)
