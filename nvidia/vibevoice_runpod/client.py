import requests
import time
import base64
import os
import json
import argparse
from cryptography.fernet import Fernet

ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "YOUR_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_API_KEY")
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def encode_audio(path):
    if not os.path.exists(path):
        print(f"Error: audio file '{path}' not found.")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encrypt_payload(data_dict):
    if not ENCRYPTION_KEY:
        print("Error: ENCRYPTION_KEY environment variable not set.")
        exit(1)
    f = Fernet(ENCRYPTION_KEY.encode())
    json_bytes = json.dumps(data_dict).encode()
    return f.encrypt(json_bytes).decode()


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR RunPod Client")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav)")
    parser.add_argument("--prompt", help="Optional context/hotword prompt")
    parser.add_argument("--format", default="parsed",
                        choices=["parsed", "raw", "transcription_only"])
    parser.add_argument("--tokenizer_chunk_size", type=int,
                        help="Override tokenizer chunk size (default 1440000)")
    parser.add_argument("--poll_interval", type=int, default=2,
                        help="Seconds between status checks")
    parser.add_argument("--debug", action="store_true",
                        help="Send plaintext (no encryption)")
    args = parser.parse_args()

    audio_b64 = encode_audio(args.audio)
    if not audio_b64:
        return

    inner = {
        "audio": audio_b64,
        "prompt": args.prompt,
        "format": args.format,
        "tokenizer_chunk_size": args.tokenizer_chunk_size,
    }

    if args.debug:
        print("DEBUG mode: sending plaintext")
        payload = {
            "input": {
                **inner,
                "is_encrypted": False,
                "debug": True,
            }
        }
    else:
        print("Encrypting payload...")
        token = encrypt_payload(inner)
        payload = {
            "input": {
                "encrypted_input": token,
                "is_encrypted": True,
                "debug": False,
            }
        }

    print(f"Submitting job to endpoint {ENDPOINT_ID}...")
    try:
        resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
        resp.raise_for_status()
        job_id = resp.json().get("id")
        print(f"Job ID: {job_id}")
    except Exception as e:
        print(f"Submission failed: {e}")
        return

    print("Polling for results...")
    last_status = None
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status != last_status:
                print(f"\nStatus: {status}", end="", flush=True)
                last_status = status

            if status == "COMPLETED":
                print()
                output = data.get("output", {})
                if isinstance(output, dict) and output.get("status") == "success":
                    print(json.dumps(output["output"], indent=2))
                elif isinstance(output, dict) and output.get("status") == "error":
                    print(f"Worker error: {output.get('message')}")
                else:
                    print(json.dumps(output, indent=2))
                break
            elif status in ("FAILED", "CANCELLED"):
                print(f"\nJob {status}: {data.get('error', 'unknown')}")
                break
            else:
                print(".", end="", flush=True)
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print(f"\nInterrupted. Job ID: {job_id}")
            break
        except Exception as e:
            print(f"\nPoll error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()