#!/usr/bin/env bash
# Build TRT and TRT-LLM engines for all model components.
#
# Usage:
#   PRECISION=fp16 bash backends/trt/scripts/build_engines.sh
#   PRECISION=int8 MAX_BATCH_SIZE=8 bash backends/trt/scripts/build_engines.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PRECISION="${PRECISION:-fp16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
DEPLOY_DIR="backends/trt"
MODEL_DIR="checkpoints"
ONNX_DIR="${DEPLOY_DIR}/onnx_models"
BUILD_DIR="${DEPLOY_DIR}/build_engines"

# Derived values
CONDITIONING_TOKENS=34
MAX_PROMPT_EMBEDDING_TABLE_SIZE=$((MAX_BATCH_SIZE * CONDITIONING_TOKENS))
DIT_MAX_BATCH=$((MAX_BATCH_SIZE * 2))  # DiT uses CFG (conditional + unconditional)

echo "=== build_engines.sh ==="
echo "  PRECISION:      ${PRECISION}"
echo "  MAX_BATCH_SIZE: ${MAX_BATCH_SIZE}"
echo "  DIT_MAX_BATCH:  ${DIT_MAX_BATCH} (2x for CFG)"
echo "  ONNX_DIR:       ${ONNX_DIR}"
echo ""

# Validate PRECISION
case "$PRECISION" in
  fp32|fp16|int8|int4) ;;
  *) echo "ERROR: Invalid PRECISION '${PRECISION}'. Must be one of: fp32, fp16, int8, int4." >&2; exit 1 ;;
esac

# Determine precision for non-GPT TRT engines
case "$PRECISION" in
  fp32) TRT_FP16_FLAG="--no-fp16"; TRT_SUFFIX="fp32" ;;
  *)    TRT_FP16_FLAG="";          TRT_SUFFIX="fp16" ;;
esac

TRT_ENGINE_DIR="${DEPLOY_DIR}/trt_engines_${TRT_SUFFIX}"
TLLM_ENGINE_DIR="${DEPLOY_DIR}/tllm_engines_${PRECISION}"
TLLM_CKPT_DIR="${DEPLOY_DIR}/tllm_checkpoint_${PRECISION}"

echo "  TRT_ENGINE_DIR:  ${TRT_ENGINE_DIR}"
echo "  TLLM_ENGINE_DIR: ${TLLM_ENGINE_DIR}"
echo "  TLLM_CKPT_DIR:   ${TLLM_CKPT_DIR}"
echo ""

# Verify checkpoint exists
if [ ! -d "$TLLM_CKPT_DIR" ] || [ ! -f "${TLLM_CKPT_DIR}/config.json" ]; then
    echo "ERROR: TRT-LLM checkpoint not found at ${TLLM_CKPT_DIR}" >&2
    echo "Run convert_checkpoint.sh first: PRECISION=${PRECISION} bash backends/trt/scripts/convert_checkpoint.sh" >&2
    exit 1
fi

# --- Step 1: Build TRT engines for non-GPT components ---
echo ">>> Step 1/2: Building TRT engines (precision=${TRT_SUFFIX})..."

mkdir -p "$TRT_ENGINE_DIR"

declare -A BUILD_MODULES=(
    ["build_bigvgan_trt.py"]="bigvgan.onnx:bigvgan.engine"
    ["build_dit_trt.py"]="dit.onnx:dit.engine"
    ["build_speech_semantic_encoder_trt.py"]="speech_semantic_encoder.onnx:speech_semantic_encoder.engine"
    ["build_semantic_codec_trt.py"]="semantic_codec.onnx:semantic_codec.engine"
    ["build_latent_projector_trt.py"]="latent_projector.onnx:latent_projector.engine"
    ["build_campplus_trt.py"]="campplus.onnx:campplus.engine"
    ["build_length_regulator_trt.py"]="length_regulator.onnx:length_regulator.engine"
    ["build_speaker_perceiver_conditioner_trt.py"]="speaker_perceiver_conditioner.onnx:speaker_perceiver_conditioner.engine"
    ["build_emotion_perceiver_conditioner_trt.py"]="emotion_perceiver_conditioner.onnx:emotion_perceiver_conditioner.engine"
)

for script in "${!BUILD_MODULES[@]}"; do
    script_path="${BUILD_DIR}/${script}"
    if [ ! -f "$script_path" ]; then
        echo "  [SKIP] ${script} (not found)"
        continue
    fi
    IFS=':' read -r onnx_name engine_name <<< "${BUILD_MODULES[$script]}"
    onnx_path="${ONNX_DIR}/${onnx_name}"
    engine_path="${TRT_ENGINE_DIR}/${engine_name}"

    if [ -f "$engine_path" ]; then
        echo "  [SKIP] ${script} (${engine_path} already exists)"
        continue
    fi
    if [ ! -f "$onnx_path" ]; then
        echo "  [SKIP] ${script} (${onnx_path} not found)"
        continue
    fi
    echo "  [RUN] ${script} ${TRT_FP16_FLAG}"
    # DiT needs 2x batch for CFG (conditional + unconditional stacked)
    if [ "$script" = "build_dit_trt.py" ]; then
        python "$script_path" --onnx "$onnx_path" --output "$engine_path" --max-batch "$DIT_MAX_BATCH" ${TRT_FP16_FLAG}
    else
        python "$script_path" --onnx "$onnx_path" --output "$engine_path" --max-batch "$MAX_BATCH_SIZE" ${TRT_FP16_FLAG}
    fi
done

# --- Step 2: Build TRT-LLM engine for GPT ---
echo ""
echo ">>> Step 2/2: Building TRT-LLM engine (PRECISION=${PRECISION})..."

if [ -d "$TLLM_ENGINE_DIR" ] && ls "${TLLM_ENGINE_DIR}"/*.engine 1>/dev/null 2>&1; then
    echo "  [SKIP] ${TLLM_ENGINE_DIR} already contains engine(s)"
else
    BUILD_TRTLLM_SCRIPT="${DEPLOY_DIR}/trtllm/build_gpt_engine.py"
    if [ ! -f "$BUILD_TRTLLM_SCRIPT" ]; then
        echo "  [SKIP] build_gpt_engine.py (not found)"
    else
        echo "  [RUN] build_gpt_engine.py"
        python "$BUILD_TRTLLM_SCRIPT" \
            --checkpoint_dir "$TLLM_CKPT_DIR" \
            --max_batch_size "$MAX_BATCH_SIZE" \
            --max_beam_width 3 \
            --output_dir "$TLLM_ENGINE_DIR" \
            --gather_all_token_logits \
            --max_prompt_embedding_table_size "$MAX_PROMPT_EMBEDDING_TABLE_SIZE"
    fi
fi

echo ""
echo "=== build_engines.sh complete ==="
echo "  TRT engines:        ${TRT_ENGINE_DIR}/"
echo "  TRT-LLM checkpoint: ${TLLM_CKPT_DIR}/"
echo "  TRT-LLM engine:     ${TLLM_ENGINE_DIR}/"
