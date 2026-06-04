#!/usr/bin/env -S uv run --script
import argparse
import os
import sys
import torch
from silero_vad import (
    load_silero_vad, 
    read_audio, 
    get_speech_timestamps, 
    collect_chunks, 
    save_audio
)

def main():
    parser = argparse.ArgumentParser(
        description="Extract, splice sequentially, and save only the voice portions of an audio file."
    )
    parser.add_argument("input_path", type=str, help="Path to the input audio file.")
    parser.add_argument("output_path", type=str, help="Path to save the spliced voice-only WAV file.")
    
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Load Silero VAD Model
    print("Loading Silero VAD model...")
    model = load_silero_vad()

    # 3. Read and automatically resample the big file to 16,000Hz mono
    print(f"Opening and resampling: {args.input_path}")
    wav = read_audio(args.input_path, sampling_rate=16000)

    # 4. Detect the timestamps where speech occurs
    print("Detecting speech regions...")
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        threshold=0.5,              # 0.5 is standard. Lower it (e.g. 0.4) if quiet speech is cut off
        min_silence_duration_ms=500  # Pause threshold before separating a segment
    )

    if not speech_timestamps:
        print("No voice segments detected in the audio file.")
        sys.exit(0)

    print(f"Found {len(speech_timestamps)} voiced segments.")

    # 5. Slice and splice the segments sequentially using Silero's utility
    print("Splicing voiced segments together...")
    voiced_audio = collect_chunks(speech_timestamps, wav)

    # 6. Save the final processed audio to disk
    print(f"Saving voice-only audio to: {args.output_path}")
    save_audio(args.output_path, voiced_audio, sampling_rate=16000)

    # Optional: Calculate and display duration savings
    original_duration = len(wav) / 16000
    spliced_duration = len(voiced_audio) / 16000
    reduction = ((original_duration - spliced_duration) / original_duration) * 100
    
    print("\n" + "="*40)
    print("VAD METRICS:")
    print(f"  Original Duration: {original_duration:.2f}s")
    print(f"  Voice-only Duration: {spliced_duration:.2f}s")
    print(f"  Silence Removed: {reduction:.1f}%")
    print("="*40)

if __name__ == "__main__":
    main()

