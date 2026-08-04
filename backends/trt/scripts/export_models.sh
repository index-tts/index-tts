#!/usr/bin/env bash
# Export ONNX models for all non-GPT components.
#
# Usage:
#   bash backends/trt/scripts/export_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

DEPLOY_DIR="backends/trt"
MODEL_DIR="checkpoints"
ONNX_DIR="${DEPLOY_DIR}/onnx_models"
EXPORT_DIR="${DEPLOY_DIR}/export"

echo "=== export_models.sh ==="
echo "  MODEL_DIR: ${MODEL_DIR}"
echo "  ONNX_DIR:  ${ONNX_DIR}"
echo ""

# Pre-download HuggingFace artifacts needed at runtime
echo ">>> Pre-caching HuggingFace artifacts..."
hf download facebook/w2v-bert-2.0 preprocessor_config.json --quiet && echo "  [OK] w2v-bert-2.0 preprocessor cached" || echo "  [SKIP] w2v-bert-2.0 preprocessor (already cached or no network)"
echo ""

mkdir -p "$ONNX_DIR"

# --- Export ONNX models ---
echo ">>> Exporting ONNX models..."

declare -a EXPORT_SCRIPTS=(
    "export_bigvgan_onnx.py"
    "export_dit_onnx.py"
    "export_speech_semantic_encoder_onnx.py"
    "export_semantic_codec_onnx.py"
    "export_speaker_perceiver_conditioner_onnx.py"
    "export_emotion_perceiver_conditioner_onnx.py"
    "export_latent_projector_onnx.py"
    "export_campplus_onnx.py"
    "export_length_regulator_onnx.py"
)

for script in "${EXPORT_SCRIPTS[@]}"; do
    script_path="${EXPORT_DIR}/${script}"
    onnx_name="${script%_onnx.py}.onnx"
    onnx_name="${onnx_name#export_}"
    onnx_path="${ONNX_DIR}/${onnx_name}"

    if [ ! -f "$script_path" ]; then
        echo "  [SKIP] ${script} (not found)"
        continue
    fi
    if [ -f "$onnx_path" ]; then
        echo "  [SKIP] ${script} (${onnx_path} already exists)"
        continue
    fi
    echo "  [RUN] ${script}"
    python "$script_path" --output "$onnx_path"
done

# --- Export speed_emb.pt ---
SPEED_EMB_SCRIPT="${EXPORT_DIR}/export_speed_emb.py"
SPEED_EMB_PATH="${ONNX_DIR}/speed_emb.pt"
if [ -f "$SPEED_EMB_SCRIPT" ] && [ ! -f "$SPEED_EMB_PATH" ]; then
    echo "  [RUN] export_speed_emb.py"
    python "$SPEED_EMB_SCRIPT" --model_dir "$MODEL_DIR" --output "$SPEED_EMB_PATH"
elif [ -f "$SPEED_EMB_PATH" ]; then
    echo "  [SKIP] export_speed_emb.py (${SPEED_EMB_PATH} already exists)"
else
    echo "  [SKIP] export_speed_emb.py (not found)"
fi

echo ""
echo "=== export_models.sh complete ==="
echo "  ONNX models: ${ONNX_DIR}/"
