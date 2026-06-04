import os
from huggingface_hub import snapshot_download

TARGET_DIR = "/app/model_download"


def prepare_model():
    model_id = os.environ.get("MODEL_ID", "microsoft/VibeVoice-ASR-HF")

    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=TARGET_DIR,
    )
    print(f"Downloaded to: {TARGET_DIR}")
    with open("/app/.model_path", "w") as f:
        f.write(TARGET_DIR)


if __name__ == "__main__":
    prepare_model()