import os
import io
import json
import base64
import soundfile as sf
import runpod
from cryptography.fernet import Fernet
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
cipher = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None

MODEL_PATH = os.environ["LOCAL_MODEL_PATH"]

print(f"Loading model from {MODEL_PATH}...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    MODEL_PATH, local_files_only=True, device_map="auto"
)
print(f"Model loaded on {model.device} with dtype {model.dtype}")


def handler(job):
    job_input = job["input"]
    is_encrypted = job_input.get("is_encrypted", False)
    debug_mode = job_input.get("debug", False)

    try:
        if is_encrypted:
            if not cipher:
                return {"status": "error", "message": "Server missing ENCRYPTION_KEY"}
            print("Decrypting payload...")
            decrypted = cipher.decrypt(job_input["encrypted_input"].encode()).decode()
            inner = json.loads(decrypted)
            audio = inner.get("audio")
            prompt = inner.get("prompt")
            fmt = inner.get("format", "parsed")
            tokenizer_chunk = inner.get("tokenizer_chunk_size")
        else:
            audio = job_input.get("audio")
            prompt = job_input.get("prompt")
            fmt = job_input.get("format", "parsed")
            tokenizer_chunk = job_input.get("tokenizer_chunk_size")
    except Exception as e:
        return {"status": "error", "message": f"Input error: {str(e)}"}

    if not audio:
        return {"status": "error", "message": "No audio provided"}

    print(f"Processing audio (format={fmt}, tokenizer_chunk={tokenizer_chunk})")

    try:
        if isinstance(audio, str):
            audio_bytes = base64.b64decode(audio)
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            audio = (audio, sr)

        inputs = processor.apply_transcription_request(
            audio=audio, prompt=prompt
        ).to(model.device, model.dtype)

        gen_kwargs = {}
        if tokenizer_chunk:
            gen_kwargs["tokenizer_chunk_size"] = tokenizer_chunk

        output_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        transcription = processor.decode(generated_ids, return_format=fmt)[0]
    except Exception as e:
        return {"status": "error", "message": f"ASR failed: {str(e)}"}

    result = {"transcription": transcription}
    if is_encrypted or debug_mode:
        return {"status": "success", "output": result}
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})