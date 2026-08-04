#!/usr/bin/env bash
# Check that this host can run the TensorRT backend, before anything heavy starts.
#
# Usage:
#   bash backends/trt/scripts/preflight.sh [PRECISION]

PRECISION="${1:-${PRECISION:-fp16}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backends/trt"
cd "$PROJECT_ROOT"

FAIL=0
WARN=0

ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$1"; WARN=$((WARN+1)); }
hint() { printf '         -> %s\n' "$1"; }

echo "=== preflight (precision=${PRECISION}) ==="

echo "-- environment --"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    warn "no virtualenv active"
    hint "source ${BACKEND_DIR#$PROJECT_ROOT/}/.venv/bin/activate"
elif [ "$(cd "$VIRTUAL_ENV" && pwd)" != "$BACKEND_DIR/.venv" ]; then
    warn "active venv is not the backend's: $VIRTUAL_ENV"
else
    ok "venv: $VIRTUAL_ENV"
fi

if ! command -v python >/dev/null 2>&1; then
    bad "no 'python' on PATH"
    hint "source ${BACKEND_DIR#$PROJECT_ROOT/}/.venv/bin/activate"
    echo ""
    echo "=== ${FAIL} failed, ${WARN} warnings ==="
    exit 1
fi

PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
if [ "$PY_VER" = "3.12" ]; then
    ok "python $PY_VER"
else
    bad "python $PY_VER (backend requires 3.12)"
    hint "uv sync --directory backends/trt"
fi

case ":${PYTHONPATH:-}:" in
    *":$PROJECT_ROOT:"*) ok "PYTHONPATH includes project root" ;;
    *) bad "project root not on PYTHONPATH"
       hint "source backends/trt/scripts/setup_env.sh" ;;
esac

echo "-- system libraries --"
if python -c "import ctypes; ctypes.CDLL('libmpi.so.40')" 2>/dev/null; then
    ok "libmpi.so.40 loadable"
else
    bad "libmpi.so.40 not loadable (tensorrt_llm links against it)"
    hint "apt-get install libopenmpi3 openmpi-bin"
    hint "no sudo? apt-get download libopenmpi3 openmpi-bin openmpi-common && dpkg -x each into ~/local-mpi/root,"
    hint "then export OPAL_PREFIX=~/local-mpi/root/usr PATH=~/local-mpi/root/usr/bin:\$PATH \\"
    hint "  LD_LIBRARY_PATH=~/local-mpi/root/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
fi

if command -v orted >/dev/null 2>&1; then
    ok "orted on PATH ($(command -v orted))"
else
    bad "orted not on PATH (OpenMPI singleton init needs it)"
    hint "it ships in openmpi-bin; add its directory to PATH"
fi

if python -c "import ctypes,sysconfig; ctypes.CDLL('libpython%s.so.1.0' % sysconfig.get_python_version())" 2>/dev/null; then
    ok "libpython loadable"
else
    warn "libpython not loadable"
    hint "source backends/trt/scripts/setup_env.sh"
fi

echo "-- python packages --"
for mod in tensorrt torch; do
    ver="$(python -c "import $mod; print($mod.__version__)" 2>/dev/null)"
    if [ -n "$ver" ]; then ok "$mod $ver"; else
        bad "cannot import $mod"
        hint "uv sync --directory backends/trt"
    fi
done

# Judge by exit status, not by stderr being empty: importing tensorrt_llm emits
# warnings (e.g. TORCH_CUDA_ARCH_LIST) on a perfectly healthy install.
# tail -1: tensorrt_llm prints a banner to stdout on import.
if TRTLLM_OUT="$(python -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/tmp/.trtllm_err | tail -1)"; then
    ok "tensorrt_llm $TRTLLM_OUT"
else
    TRTLLM_ERR="$(grep -E "Error|error:" /tmp/.trtllm_err | tail -1)"
    [ -z "$TRTLLM_ERR" ] && TRTLLM_ERR="$(tail -1 /tmp/.trtllm_err)"
    bad "cannot import tensorrt_llm: $TRTLLM_ERR"
    case "$TRTLLM_ERR" in
        *MPI*|*mpi*) hint "see the libmpi.so.40 hints above" ;;
        *libpython*) hint "source backends/trt/scripts/setup_env.sh" ;;
        *)           hint "uv sync --directory backends/trt" ;;
    esac
fi
rm -f /tmp/.trtllm_err

echo "-- gpu --"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    bad "nvidia-smi not found"
elif python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ok "cuda available: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [ -n "$FREE_MIB" ]; then
        if [ "$FREE_MIB" -lt 8000 ]; then
            warn "only ${FREE_MIB} MiB free on GPU 0"
            hint "engine build and inference may OOM; pick another GPU with CUDA_VISIBLE_DEVICES"
        else
            ok "${FREE_MIB} MiB free on GPU 0"
        fi
    fi
else
    bad "torch.cuda.is_available() is False"
fi

echo "-- build artifacts --"
TRT_SUFFIX="fp32"; [ "$PRECISION" != "fp32" ] && TRT_SUFFIX="fp16"
ONNX_DIR="$BACKEND_DIR/onnx_models"
TRT_DIR="$BACKEND_DIR/trt_engines_${TRT_SUFFIX}"
GPT_DIR="$BACKEND_DIR/tllm_engines_${PRECISION}"

N_ONNX=$(ls "$ONNX_DIR"/*.onnx 2>/dev/null | wc -l | tr -d ' ')
N_TRT=$(ls "$TRT_DIR"/*.engine 2>/dev/null | wc -l | tr -d ' ')

if [ "$N_ONNX" -ge 9 ]; then ok "onnx models: $N_ONNX"
else warn "onnx models: $N_ONNX/9"
     hint "bash backends/trt/scripts/export_models.sh" ; fi

[ -f "$ONNX_DIR/speed_emb.pt" ] && ok "speed_emb.pt" || warn "speed_emb.pt missing"

if [ "$N_TRT" -ge 9 ]; then ok "trt engines: $N_TRT"
else warn "trt engines: $N_TRT/9"
     hint "PRECISION=${PRECISION} MAX_BATCH_SIZE=1 bash backends/trt/scripts/build_engines.sh" ; fi

if ls "$GPT_DIR"/*.engine >/dev/null 2>&1; then ok "gpt engine present"
else warn "gpt engine missing"
     hint "PRECISION=${PRECISION} bash backends/trt/scripts/convert_checkpoint.sh, then build_engines.sh" ; fi

CKPT_MISSING=0
for f in config.yaml gpt.pth s2mel.pth bpe.model; do
    if [ ! -e "$PROJECT_ROOT/checkpoints/$f" ]; then
        bad "checkpoints/$f missing"
        CKPT_MISSING=$((CKPT_MISSING+1))
    fi
done
if [ "$CKPT_MISSING" -eq 0 ]; then
    ok "checkpoints present"
else
    hint "see the main README for model download"
fi

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "=== ${FAIL} blocking problem(s), ${WARN} warning(s) ==="
    exit 1
fi
if [ "$WARN" -gt 0 ]; then
    echo "=== ready to build (${WARN} warning(s): artifacts not built yet) ==="
    exit 0
fi
echo "=== all checks passed ==="
