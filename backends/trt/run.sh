#!/usr/bin/env bash
# IndexTTS2 TensorRT backend — single entry point for setup, build and inference.
#
# Usage:
#   bash backends/trt/run.sh <command> [options]
#
# Commands:
#   check             Verify the environment and print what is missing
#   build             Export ONNX -> convert GPT checkpoint -> build engines
#   infer <options>   Synthesize speech         (see `infer --help`)
#   serve <options>   Start the PyTriton server (see `serve --help`)
#
# Examples:
#   bash backends/trt/run.sh check
#   bash backends/trt/run.sh build
#   bash backends/trt/run.sh infer --text "hello" --speaker examples/voice_01.wav --output out.wav
#   bash backends/trt/run.sh serve --mode streaming
#
# Environment:
#   PRECISION        fp32|fp16|int8|int4       (default: fp16)
#   MAX_BATCH_SIZE   engine build batch size   (default: 1)
#   OPENMPI_PREFIX   OpenMPI prefix, if it is not on the default library path
#   SKIP_CHECK=1     skip the environment check before build/infer/serve
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PRECISION="${PRECISION:-fp16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
cd "$ROOT"

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
hint() { printf '        -> %s\n' "$1"; }

# --- environment -------------------------------------------------------------
setup() {
    if [ ! -x "$HERE/.venv/bin/python" ]; then
        echo "ERROR: no venv at $HERE/.venv" >&2
        echo "  -> uv sync --directory backends/trt" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$HERE/.venv/bin/activate"
    # shellcheck disable=SC1091
    source "$HERE/scripts/setup_env.sh" >/dev/null

    python -c "import ctypes; ctypes.CDLL('libmpi.so.40')" 2>/dev/null && return 0
    for p in "${OPENMPI_PREFIX:-}" "$HOME/local-mpi/root/usr" \
             /usr/lib/x86_64-linux-gnu/openmpi /usr /usr/local /opt/hpcx/ompi; do
        [ -n "$p" ] || continue
        local libdir="$p/lib"
        [ -d "$p/lib/x86_64-linux-gnu" ] && libdir="$p/lib/x86_64-linux-gnu"
        [ -e "$libdir/libmpi.so.40" ] || continue
        export OPAL_PREFIX="$p"
        export PATH="$p/bin:$PATH"
        export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        return 0
    done
    return 0  # let check/import report it
}

# --- check -------------------------------------------------------------------
check() {
    local fail=0
    echo "environment"

    local pv
    pv="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    [ "$pv" = "3.12" ] && ok "python $pv" || {
        bad "python $pv (needs 3.12)"; hint "uv sync --directory backends/trt"; fail=1; }

    local err
    if err="$(python -c 'import tensorrt_llm' 2>&1 >/dev/null)"; then
        ok "tensorrt_llm $(python -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null | tail -1)"
    else
        bad "cannot import tensorrt_llm"
        fail=1
        case "$err" in
            *libmpi*|*"MPI library"*)
                hint "OpenMPI 4.x missing: apt-get install libopenmpi3 openmpi-bin"
                hint "no sudo? apt-get download libopenmpi3 openmpi-bin openmpi-common,"
                hint "  dpkg -x each into ~/local-mpi/root, then re-run (it is auto-detected)"
                hint "note: Intel MPI (impi-rt) is NOT a substitute" ;;
            *OMPI_COMM_TYPE_HOST*|*split_type*)
                hint "Intel MPI is being used; uninstall impi-rt and install OpenMPI 4.x" ;;
            *orted*|*ess_singleton*)
                hint "OpenMPI found but 'orted' is missing: apt-get install openmpi-bin" ;;
            *libpython*)
                hint "libpython not on LD_LIBRARY_PATH; report this, run.sh should have set it" ;;
            *)  printf '        %s\n' "$(printf '%s' "$err" | tail -1)"
                hint "uv sync --directory backends/trt" ;;
        esac
    fi

    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        && ok "cuda: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)" \
        || { bad "no usable CUDA device"; fail=1; }

    for f in config.yaml gpt.pth s2mel.pth bpe.model; do
        [ -e "$ROOT/checkpoints/$f" ] || { bad "checkpoints/$f missing"; fail=1; }
    done

    echo "artifacts (precision=$PRECISION)"
    local trt_suffix="fp16"; [ "$PRECISION" = "fp32" ] && trt_suffix="fp32"
    local n_onnx n_trt
    n_onnx=$(ls "$HERE"/onnx_models/*.onnx 2>/dev/null | wc -l | tr -d ' ')
    n_trt=$(ls "$HERE/trt_engines_${trt_suffix}"/*.engine 2>/dev/null | wc -l | tr -d ' ')
    local missing=0
    [ "$n_onnx" -ge 9 ] && ok "onnx: $n_onnx/9" || { echo "  --    onnx: $n_onnx/9"; missing=1; }
    [ "$n_trt" -ge 9 ]  && ok "engines: $n_trt/9" || { echo "  --    engines: $n_trt/9"; missing=1; }
    ls "$HERE/tllm_engines_${PRECISION}"/*.engine >/dev/null 2>&1 \
        && ok "gpt engine" || { echo "  --    gpt engine"; missing=1; }
    [ "$missing" -eq 1 ] && hint "bash backends/trt/run.sh build"

    echo ""
    if [ "$fail" -ne 0 ]; then
        echo "not ready — fix the items above"
        return 1
    fi
    [ "$missing" -eq 1 ] && echo "environment ok — artifacts not built yet" || echo "ready"
    return 0
}

# --- dispatch ----------------------------------------------------------------
[ $# -eq 0 ] && { usage; exit 1; }
CMD="$1"; shift

case "$CMD" in -h|--help|help) usage; exit 0 ;; esac

setup

for a in "$@"; do
    case "$a" in -h|--help)
        case "$CMD" in
            infer) exec python "$HERE/infer.py" --help ;;
            serve) exec python "$HERE/serving/triton_server.py" --help ;;
        esac ;;
    esac
done

if [ "$CMD" != "check" ] && [ "${SKIP_CHECK:-0}" != "1" ]; then
    check >/dev/null 2>&1 || { check; echo ""; echo "Set SKIP_CHECK=1 to run anyway." >&2; exit 1; }
fi

case "$CMD" in
    check)
        check
        ;;
    build)
        bash "$HERE/scripts/export_models.sh"
        PRECISION="$PRECISION" bash "$HERE/scripts/convert_checkpoint.sh"
        PRECISION="$PRECISION" MAX_BATCH_SIZE="$MAX_BATCH_SIZE" bash "$HERE/scripts/build_engines.sh"
        ;;
    infer)
        exec python "$HERE/infer.py" --precision "$PRECISION" "$@"
        ;;
    serve)
        exec python "$HERE/serving/triton_server.py" --precision "$PRECISION" "$@"
        ;;
    *)
        echo "ERROR: unknown command '$CMD'" >&2
        echo "" >&2
        usage >&2
        exit 1
        ;;
esac
