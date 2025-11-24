#!/usr/bin/env python3
"""
Calculate WER only (without training) - for benchmarking purposes.
Compares stored ASR transcripts against manual corrections.
"""
import os
import sys
import json
from pathlib import Path
import jiwer

# Configuration
RECORD_DIR = Path("data/records")
WER_FILE = Path("data/wer.txt")
TRAINING_STATUS_FILE = Path("data/training_status.json")

def update_status(status, progress, message):
    """Update training status file."""
    status_data = {
        "status": status,
        "progress": progress,
        "message": message
    }
    with open(TRAINING_STATUS_FILE, "w") as f:
        json.dump(status_data, f)

def calculate_wer():
    """Calculate WER by comparing stored ASR transcripts against manual corrections."""
    try:
        update_status("running", 20, "Loading records...")
        print("Loading manually reviewed records...")
        
        # Load all records with manual transcripts
        records = []
        for json_path in RECORD_DIR.glob("*.json"):
            try:
                record = json.loads(json_path.read_text())
                if record.get("manual_transcript") and record.get("asr_transcript"):
                    records.append(record)
            except json.JSONDecodeError:
                print(f"Skipping corrupt file: {json_path}")
        
        if not records:
            print("❌ No manually reviewed records found!")
            update_status("failed", 0, "No manually reviewed records to evaluate")
            WER_FILE.write_text("WER: N/A (no reviewed records)\n")
            return
        
        print(f"Found {len(records)} manually reviewed records")
        update_status("running", 60, f"Calculating WER for {len(records)} records...")
        
        # Compare stored ASR transcripts vs manual corrections
        preds = [r["asr_transcript"] for r in records]
        refs = [r["manual_transcript"] for r in records]
        
        update_status("running", 90, "Computing WER metric...")
        wer = jiwer.wer(refs, preds)
        
        # Write WER to file
        with open(WER_FILE, "w") as f:
            from datetime import datetime
            f.write(f"=== WER Benchmark Report (Base Model) ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Number of samples: {len(records)}\n")
            f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
            f.write(f"\nNote: Comparing stored ASR transcripts vs manual corrections\n")
            f.write(f"\n--- Sample Comparisons (first 3) ---\n")
            for i in range(min(3, len(records))):
                f.write(f"\nSample {i+1}:\n")
                f.write(f"  ASR Output:    {preds[i]}\n")
                f.write(f"  Manual Correction: {refs[i]}\n")
        
        print(f"\n✅ WER Calculation Complete!")
        print(f"   WER: {wer:.4f} ({wer*100:.2f}%)")
        print(f"   Samples evaluated: {len(records)}")
        print(f"   Method: Comparing stored ASR transcripts vs manual corrections")
        
        update_status("completed", 100, f"WER calculated: {wer*100:.2f}%")
        
    except Exception as e:
        print(f"\n❌ WER calculation failed with error: {e}")
        import traceback
        traceback.print_exc()
        update_status("failed", 0, f"WER calculation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    calculate_wer()
