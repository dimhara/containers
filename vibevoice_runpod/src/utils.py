import os
from huggingface_hub import snapshot_download

RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
TARGET_DIR = "/app/model_download"


def find_in_runpod_cache(repo_id):
    safe_repo = f"models--{repo_id.replace('/', '--').lower()}"
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

    os.makedirs(TARGET_DIR, exist_ok=True)

    # Point HF_HOME at the RunPod cache so snapshot_download reuses it
    os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"

    cached_path = find_in_runpod_cache(model_id)
    if cached_path:
        print(f"Found in RunPod cache: {cached_path}")
        for fname in os.listdir(cached_path):
            src = os.path.join(cached_path, fname)
            dst = os.path.join(TARGET_DIR, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        print(f"Linked cached files to: {TARGET_DIR}")
        with open("/app/.model_path", "w") as f:
            f.write(TARGET_DIR)
        return

    print(f"Not in RunPod cache. Downloading {model_id}...")
    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=True,
    )
    print(f"Downloaded to: {local_path}")
    with open("/app/.model_path", "w") as f:
        f.write(TARGET_DIR)


if __name__ == "__main__":
    prepare_model()