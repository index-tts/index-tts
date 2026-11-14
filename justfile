set shell := ["bash", "-uc"]
set dotenv-load

model_dir := env_var_or_default("MODEL_DIR", "./checkpoints")
host := env_var_or_default("HOST", "0.0.0.0")
port := env_var_or_default("PORT", "7860")

default:
    @just --list

# Windows recipes are intentionally omitted until they are validated.

[group("linux")]
install-webui:
    uv sync --extra webui

[group("linux")]
install-all:
    uv sync --all-extras

[group("linux")]
download-model model_dir=model_dir:
    uv run indextts2 download --model-dir '{{ model_dir }}'

[group("linux")]
download-aux model_dir=model_dir:
    MODEL_DIR='{{ model_dir }}' uv run --extra webui python -c 'import os; from indextts.utils.model_download import ensure_models_available; ensure_models_available(os.environ["MODEL_DIR"])'

[group("linux")]
check-model model_dir=model_dir:
    uv run indextts2 check --model-dir '{{ model_dir }}'

[group("linux")]
check-gpu:
    if [ -n "${CUDA_LIB:-}" ]; then export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"; fi; uv run --extra webui python -c 'import torch; print("cuda available:", torch.cuda.is_available()); print("device count:", torch.cuda.device_count()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

[group("linux")]
smi:
    nvidia-smi

[group("linux")]
webui-gpu model_dir=model_dir host=host port=port:
    if [ -n "${CUDA_LIB:-}" ]; then export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"; fi; uv run --extra webui webui.py --host '{{ host }}' --port '{{ port }}' --model_dir '{{ model_dir }}' --fp16

[group("linux")]
webui-cpu model_dir=model_dir host=host port=port:
    uv run --extra webui webui.py --host '{{ host }}' --port '{{ port }}' --model_dir '{{ model_dir }}'
