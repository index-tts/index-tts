#!/usr/bin/env bash
# Build the faster-indextts-2 Docker image (export models + build engines).
#
# Usage:
#   bash backends/trt/scripts/build_image.sh --triton --fast [PRECISION]          # fast build (native Triton, default)
#   bash backends/trt/scripts/build_image.sh --triton [PRECISION]                 # full build (native Triton)
#   bash backends/trt/scripts/build_image.sh --pytriton [PRECISION]               # full build (PyTriton)
#   bash backends/trt/scripts/build_image.sh --export-only                        # export only
#   bash backends/trt/scripts/build_image.sh --engines-only [PRECISION]           # skip export
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Parse args
ENGINES_ONLY=false
EXPORT_ONLY=false
FAST_MODE=false
SERVING_MODE="triton"  # pytriton or triton

while [ $# -gt 0 ]; do
    case "${1}" in
        --engines-only) ENGINES_ONLY=true; shift ;;
        --export-only)  EXPORT_ONLY=true; shift ;;
        --fast)         FAST_MODE=true; shift ;;
        --triton)       SERVING_MODE="triton"; shift ;;
        --pytriton)     SERVING_MODE="pytriton"; shift ;;
        *)              break ;;
    esac
done

PRECISION="${1:-fp16}"
BASE_IMAGE="faster-indextts-2-base"
EXPORTED_IMAGE="faster-indextts-2-exported"
REMOTE_BASE_IMAGE="myrond88302/faster-indextts-2-exported:latest"

if [ "$SERVING_MODE" = "triton" ]; then
    FINAL_IMAGE="faster-indextts-2-triton:${PRECISION}"
    DOCKERFILE="backends/trt/Dockerfile.triton"
else
    FINAL_IMAGE="faster-indextts-2-pytriton:${PRECISION}"
    DOCKERFILE="backends/trt/Dockerfile.pytriton"
fi

echo "=== faster-indextts-2 image build ==="
echo "  PRECISION:      ${PRECISION}"
echo "  SERVING_MODE:   ${SERVING_MODE}"
echo "  FAST_MODE:      ${FAST_MODE}"
echo "  EXPORT_ONLY:    ${EXPORT_ONLY}"
echo "  ENGINES_ONLY:   ${ENGINES_ONLY}"
echo "  DOCKERFILE:     ${DOCKERFILE}"
echo "  BASE_IMAGE:     ${BASE_IMAGE}"
echo "  EXPORTED_IMAGE: ${EXPORTED_IMAGE}"
echo "  FINAL_IMAGE:    ${FINAL_IMAGE}"
echo ""

# Validate PRECISION
case "$PRECISION" in
  fp32|fp16|int8|int4) ;;
  *) echo "ERROR: Invalid PRECISION '${PRECISION}'. Must be one of: fp32, fp16, int8, int4." >&2; exit 1 ;;
esac

if [ "$FAST_MODE" = true ]; then
    # --- Fast mode: pull pre-built exported image, skip Steps 1-2 ---
    echo ">>> Fast mode: pulling pre-built exported image..."
    if ! docker image inspect "$EXPORTED_IMAGE" >/dev/null 2>&1; then
        docker pull "$REMOTE_BASE_IMAGE"
        docker tag "$REMOTE_BASE_IMAGE" "$EXPORTED_IMAGE"
        echo "  Pulled and tagged: ${REMOTE_BASE_IMAGE} -> ${EXPORTED_IMAGE}"
    else
        echo "  [SKIP] ${EXPORTED_IMAGE} already exists locally"
    fi

elif [ "$ENGINES_ONLY" = false ]; then
    # --- Step 1: Build base image ---
    echo ">>> Step 1/3: Building base image..."

    # Validate checkpoints exist
    if [ ! -f "checkpoints/config.yaml" ]; then
        echo "ERROR: checkpoints/config.yaml not found." >&2
        echo "Download checkpoints first:" >&2
        echo "  hf download IndexTeam/IndexTTS-2 --local-dir checkpoints" >&2
        exit 1
    fi

    docker build --network=host \
        ${HTTP_PROXY:+--build-arg HTTP_PROXY="${HTTP_PROXY}"} \
        ${HTTPS_PROXY:+--build-arg HTTPS_PROXY="${HTTPS_PROXY}"} \
        ${NO_PROXY:+--build-arg NO_PROXY="${NO_PROXY}"} \
        -t "$BASE_IMAGE" -f "$DOCKERFILE" .

    # --- Step 2: Export ONNX models ---
    echo ""
    echo ">>> Step 2/3: Exporting models (precision-independent)..."
    docker rm -f "faster-indextts-2-exporter" 2>/dev/null || true
    docker run --gpus all --network=host --name "faster-indextts-2-exporter" \
        ${HTTP_PROXY:+--env HTTP_PROXY="${HTTP_PROXY}"} \
        ${HTTPS_PROXY:+--env HTTPS_PROXY="${HTTPS_PROXY}"} \
        ${NO_PROXY:+--env NO_PROXY="${NO_PROXY}"} \
        ${HF_ENDPOINT:+--env HF_ENDPOINT="${HF_ENDPOINT}"} \
        "$BASE_IMAGE" \
        bash backends/trt/scripts/export_models.sh

    echo "  Committing exported image: ${EXPORTED_IMAGE}"
    docker commit "faster-indextts-2-exporter" "$EXPORTED_IMAGE"
    docker rm "faster-indextts-2-exporter"

    if [ "$EXPORT_ONLY" = true ]; then
        echo ""
        echo "=== Export complete ==="
        echo "  Exported image: ${EXPORTED_IMAGE}"
        echo ""
        echo "To build engines:"
        echo "  bash backends/trt/scripts/build_image.sh --engines-only fp16"
        exit 0
    fi
else
    # Skip steps 1-2, verify exported image exists
    if ! docker image inspect "$EXPORTED_IMAGE" >/dev/null 2>&1; then
        echo "ERROR: Exported image '${EXPORTED_IMAGE}' not found." >&2
        echo "Run without --engines-only first to create it." >&2
        exit 1
    fi
    echo ">>> Skipping Steps 1-2 (using existing ${EXPORTED_IMAGE})"
fi

# --- Step 3: Convert checkpoint + build engines ---
echo ""
echo ">>> Step 3/3: Converting checkpoint + building engines (PRECISION=${PRECISION})..."
_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
docker rm -f "faster-indextts-2-builder" 2>/dev/null || true
docker run --gpus all --name "faster-indextts-2-builder" \
    -e PRECISION="$PRECISION" \
    -e MAX_BATCH_SIZE="$_BATCH_SIZE" \
    "$EXPORTED_IMAGE" \
    bash -c "PRECISION=${PRECISION} bash backends/trt/scripts/convert_checkpoint.sh && PRECISION=${PRECISION} MAX_BATCH_SIZE=${_BATCH_SIZE} bash backends/trt/scripts/build_engines.sh && sed -i 's/^max_batch_size:.*/max_batch_size: ${_BATCH_SIZE}/' backends/trt/serving/model_repository/indextts2/config.pbtxt backends/trt/serving/model_repository/indextts2_stream/config.pbtxt"

echo "  Committing final image: ${FINAL_IMAGE}"
docker commit "faster-indextts-2-builder" "$FINAL_IMAGE"
docker rm "faster-indextts-2-builder"

echo ""
echo "=== Build complete ==="
echo "  Exported image: ${EXPORTED_IMAGE}"
echo "  Final image:    ${FINAL_IMAGE}"
echo ""
if [ "$SERVING_MODE" = "triton" ]; then
    echo "Run with (native Triton):"
    echo "  docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 ${FINAL_IMAGE}"
else
    echo "Run with (PyTriton):"
    echo "  docker run --gpus all -p 8001:8001 ${FINAL_IMAGE} python backends/trt/serving/triton_server.py --mode streaming --precision ${PRECISION}"
fi
echo ""
echo "To rebuild engines with a different precision (reuses exported models):"
echo "  bash backends/trt/scripts/build_image.sh ${SERVING_MODE:+--${SERVING_MODE} }--engines-only int8"
