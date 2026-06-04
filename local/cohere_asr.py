#!/usr/bin/env -S uv run --script
import argparse
import os
import sys
import torch
from transformers import (
    AutoProcessor, 
    CohereAsrForConditionalGeneration, 
    BitsAndBytesConfig
)
from transformers.audio_utils import load_audio

def main():
    parser = argparse.ArgumentParser(description="Definitive Cohere ASR Script (BnB & Non-BnB)")
    
    # Required
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    
    # Core Settings
    parser.add_argument("--quant", type=str, choices=["4bit", "8bit", "none"], default="none", 
                        help="Quantization level. 'none' uses pure FP16 (fastest). '4bit' uses lowest VRAM.")
    parser.add_argument("--lang", type=str, default="el", help="Language code (default: 'el' for Greek)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Compute device")
    
    # Optional Output
    parser.add_argument("--save", type=str, help="Optional: Path to save the transcription text file.")

    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file '{args.audio_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Device & Precision Logic
    device = "cuda" if torch.cuda.is_available() and args.device in ["auto", "cuda"] else "cpu"
    
    # If CPU is forced or fallback, disable BnB (BitsAndBytes requires GPU)
    if device == "cpu" and args.quant != "none":
        print("Warning: bitsandbytes requires a GPU. Falling back to non-quantized CPU (float32).")
        args.quant = "none"

    bnb_config = None
    model_dtype = torch.float32 # Default for CPU

    if device == "cuda":
        model_dtype = torch.float16  # FP16 is standard for GPU
        
        if args.quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif args.quant == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 3. Load Model and Processor
    model_id = "CohereLabs/cohere-transcribe-03-2026"
    print(f"\n--- Initializing Model ({model_id}) ---")
    print(f"Device: {device.upper()} | Quantization: {args.quant.upper()} | Compute Dtype: {model_dtype}")

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    
    try:
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, local_files_only=True,
            quantization_config=bnb_config,
            torch_dtype=model_dtype,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Ensure you ran 'huggingface-cli login' and accepted the model license.")
        sys.exit(1)

    # 4. Audio Processing (handles the 1-10 min chunking)
    print(f"\n--- Processing Audio: {args.audio_path} ---")
    audio = load_audio(args.audio_path, sampling_rate=16000)
    
    inputs = processor(
        audio=audio, 
        sampling_rate=16000, 
        return_tensors="pt", 
        language=args.lang
    ).to(device)

    # Match input precision to model precision on GPU to avoid dtype mismatch errors
    if device == "cuda":
        inputs = inputs.to(dtype=torch.float16)

    # 5. Generation
    print("--- Transcribing (This may take a moment) ---")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=448)

    # 6. Decode & Reassemble
    print("--- Finalizing Transcript ---")
    transcript = processor.decode(
        outputs, 
        skip_special_tokens=True, 
        audio_chunk_index=inputs.get("audio_chunk_index"), 
        language=args.lang
    )

    # 7. Output Results
    print("\n" + "="*50)
    print(" TRANSCRIPTION RESULT")
    print("="*50)
    print(transcript)
    print("="*50 + "\n")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Saved successfully to: {args.save}")

if __name__ == "__main__":
    main()

