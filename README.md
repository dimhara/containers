# containers

Two groups separated into subdirectories:

### `nvidia/` — Cloud (RunPod)
Demanding models that require NVIDIA CUDA GPUs (24GB+ VRAM).

- `Containerfile.vibevoice` — Self-contained RunPod serverless worker for VibeVoice-ASR 8B
- `vibevoice_runpod/` — Handler, client, tests, start script

### `local/` — Local (CPU / ROCm)
Lightweight models that run on CPU or ROCm, up to 8GB VRAM.

- `Containerfile.hf` — Generic base: `ARG TARGET_DEVICE=cpu` (override to `rocm7.2` for local ROCm builds)
- `Containerfile.vad` — Extends `hf-base:cpu` with Silero VAD + BnB + torchcodec
- `cohere_asr.py` — Cohere ASR (1B) with 4-bit/8-bit quantization support
- `silero_vad_split.py` — Voice activity detection splicer

### CI

Three independent build jobs — see `.github/workflows/build-hf.yml`:
- `hf-base` — builds `local/Containerfile.hf`, pushes `ghcr.io/dimhara/containers/hf-base:{latest,cpu}`
- `vibevoice` — builds `nvidia/Containerfile.vibevoice`, pushes `ghcr.io/dimhara/containers/vibevoice:{latest,cuda13}`
- `vad` — builds `local/Containerfile.vad`, pushes `ghcr.io/dimhara/containers/vad:{latest,cpu}`
