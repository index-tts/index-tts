#!/usr/bin/env bash
# Activate the backend environment, check it, then run a command.
#
# Usage:
#   bash backends/trt/scripts/run.sh infer  --text "hi" --speaker examples/voice_01.wav --output out.wav
#   bash backends/trt/scripts/run.sh serve  --mode streaming --max_batch_size 1
#   bash backends/trt/scripts/run.sh client --mode streaming --text "hi" --speaker_audio examples/voice_01.wav
#   bash backends/trt/scripts/run.sh check
#   bash backends/trt/scripts/run.sh python -c "import tensorrt_llm"
#
# Environment:
#   PRECISION        fp32|fp16|int8|int4 (default fp16)
#   OPENMPI_PREFIX   OpenMPI install prefix, if not on the default library path
#   SKIP_PREFLIGHT   set to 1 to skip the environment check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backends/trt"
PRECISION="${PRECISION:-fp16}"
cd "$PROJECT_ROOT"

if [ $# -eq 0 ]; then
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi
SUBCMD="$1"; shift

# --- venv ---
if [ ! -x "$BACKEND_DIR/.venv/bin/python" ]; then
    echo "ERROR: no venv at $BACKEND_DIR/.venv" >&2
    echo "Run: uv sync --directory backends/trt" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$BACKEND_DIR/.venv/bin/activate"

# --- PYTHONPATH, protobuf backend, libpython ---
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh" >/dev/null

# --- OpenMPI: needed by tensorrt_llm, not installable via pip ---
_try_mpi_prefix() {
    local prefix="$1"
    [ -d "$prefix/lib/x86_64-linux-gnu" ] && local libdir="$prefix/lib/x86_64-linux-gnu" || local libdir="$prefix/lib"
    [ -e "$libdir/libmpi.so.40" ] || return 1
    export OPAL_PREFIX="$prefix"
    export PATH="$prefix/bin:$PATH"
    export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    return 0
}

if python -c "import ctypes; ctypes.CDLL('libmpi.so.40')" 2>/dev/null; then
    :  # already resolvable
elif [ -n "${OPENMPI_PREFIX:-}" ]; then
    _try_mpi_prefix "$OPENMPI_PREFIX" || {
        echo "ERROR: no libmpi.so.40 under OPENMPI_PREFIX=$OPENMPI_PREFIX" >&2; exit 1; }
else
    for _p in "$HOME/local-mpi/root/usr" /usr/lib/x86_64-linux-gnu/openmpi /usr /usr/local /opt/hpcx/ompi; do
        _try_mpi_prefix "$_p" && break
    done
fi

# --- preflight ---
if [ "${SKIP_PREFLIGHT:-0}" != "1" ] && [ "$SUBCMD" != "check" ]; then
    if ! PRECISION="$PRECISION" bash "$SCRIPT_DIR/preflight.sh" "$PRECISION"; then
        echo "" >&2
        echo "Preflight failed. Fix the items above, or set SKIP_PREFLIGHT=1 to run anyway." >&2
        exit 1
    fi
    echo ""
fi

# --- dispatch ---
case "$SUBCMD" in
    check)  exec bash "$SCRIPT_DIR/preflight.sh" "$PRECISION" ;;
    infer)  exec python "$BACKEND_DIR/infer.py" --precision "$PRECISION" "$@" ;;
    serve)  exec python "$BACKEND_DIR/serving/triton_server.py" --precision "$PRECISION" "$@" ;;
    client) exec python "$BACKEND_DIR/serving/triton_client.py" "$@" ;;
    python) exec python "$@" ;;
    *)      echo "ERROR: unknown subcommand '$SUBCMD' (expected: infer serve client check python)" >&2; exit 1 ;;
esac
