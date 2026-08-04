#!/usr/bin/env bash
# Convert IndexTTS2 GPT checkpoint to TRT-LLM format.
#
# Usage:
#   PRECISION=fp16 bash backends/trt/scripts/convert_checkpoint.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PRECISION="${PRECISION:-fp16}"
DEPLOY_DIR="backends/trt"
MODEL_DIR="checkpoints"
TLLM_CKPT_DIR="${DEPLOY_DIR}/tllm_checkpoint_${PRECISION}"

echo "=== convert_checkpoint.sh ==="
echo "  PRECISION:     ${PRECISION}"
echo "  MODEL_DIR:     ${MODEL_DIR}"
echo "  TLLM_CKPT_DIR: ${TLLM_CKPT_DIR}"
echo ""

# Validate PRECISION
case "$PRECISION" in
  fp32|fp16|int8|int4) ;;
  *) echo "ERROR: Invalid PRECISION '${PRECISION}'. Must be one of: fp32, fp16, int8, int4." >&2; exit 1 ;;
esac

if [ -d "$TLLM_CKPT_DIR" ] && [ -f "${TLLM_CKPT_DIR}/config.json" ]; then
    echo "  [SKIP] ${TLLM_CKPT_DIR} already exists"
else
    CONVERT_SCRIPT="${DEPLOY_DIR}/trtllm/convert_gpt_checkpoint.py"
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        echo "  [ERROR] convert_gpt_checkpoint.py not found at ${CONVERT_SCRIPT}" >&2
        exit 1
    fi

    CONVERT_ARGS=(
        --model_dir "$MODEL_DIR"
        --output_dir "$TLLM_CKPT_DIR"
    )
    case "$PRECISION" in
        fp32)  CONVERT_ARGS+=(--dtype float32) ;;
        fp16)  CONVERT_ARGS+=(--dtype float16) ;;
        int8)  CONVERT_ARGS+=(--dtype float16 --use_weight_only --weight_only_precision int8) ;;
        int4)  CONVERT_ARGS+=(--dtype float16 --use_weight_only --weight_only_precision int4) ;;
    esac

    echo "  [RUN] convert_gpt_checkpoint.py ${CONVERT_ARGS[*]}"
    python "$CONVERT_SCRIPT" "${CONVERT_ARGS[@]}"
fi

echo ""
echo "=== convert_checkpoint.sh complete ==="
echo "  TRT-LLM checkpoint: ${TLLM_CKPT_DIR}/"
