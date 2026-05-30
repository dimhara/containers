import os
import shutil
from huggingface_hub import hf_hub_download

RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def find_in_runpod_cache(repo_id):
    safe_repo = f"models--{repo_id.replace('/', '--')}"
    repo_path = os.path.join(RUNPOD_CACHE_DIR, safe_repo, "snapshots")

    if not os.path.exists(repo_path):
        return None

    for snapshot in sorted(os.listdir(repo_path)):
        full_path = os.path.join(repo_path, snapshot)
        if os.path.isdir(full_path):
            return os.path.realpath(full_path)
    return None


def prepare_model():
    model_id = os.environ.get("MODEL_ID", "microsoft/VibeVoice-ASR-HF")

    cached_path = find_in_runpod_cache(model_id)
    if cached_path:
        print(f"Found in RunPod cache: {cached_path}")
        with open("/app/.model_path", "w") as f:
            f.write(cached_path)
        return

    print(f"Not in cache. Downloading {model_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            local_dir="/app/model_download",
            local_dir_use_symlinks=False,
        )
        parent = os.path.dirname(downloaded_path)
        with open("/app/.model_path", "w") as f:
            f.write(parent)
        print(f"Downloaded to: {parent}")
    except Exception as e:
        print(f"Download failed (may be multi-file model): {e}")
        print("Attempting full repo download via snapshot_download...")
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=model_id,
            local_dir="/app/model_download",
            local_dir_use_symlinks=False,
        )
        with open("/app/.model_path", "w") as f:
            f.write(local_path)
        print(f"Downloaded snapshot to: {local_path}")


if __name__ == "__main__":
    prepare_model()