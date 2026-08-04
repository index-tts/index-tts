#!/usr/bin/env bash
# Configure runtime environment variables for faster-indextts-2.
#
# Usage:
#   source backends/trt/scripts/setup_env.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "$_SCRIPT_DIR/../../.." && pwd)"

# --- PYTHONPATH: IndexTTS-2 project root ---
case ":${PYTHONPATH:-}:" in
  *":${_PROJECT_ROOT}:"*) ;;
  *) export PYTHONPATH="${_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# --- Protobuf pure-Python backend ---
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

# --- Python LIBDIR (for PyTriton -> Triton Server's libpython3.x.so) ---
_prepend_ld() {
  local p="$1"
  [ -z "$p" ] && return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$p:"*) return 0 ;;
  esac
  export LD_LIBRARY_PATH="${p}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

_PY_BIN=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  _PY_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  _PY_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "$(command -v python 2>/dev/null)" ]; then
  _PY_BIN="$(command -v python)"
fi

if [ -n "$_PY_BIN" ]; then
  _PY_LIBDIR="$($_PY_BIN -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null)"
  _prepend_ld "$_PY_LIBDIR"
  if [ -n "${CONDA_PREFIX:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    _prepend_ld "${CONDA_PREFIX}/lib"
  fi
fi
unset _PY_BIN _PY_LIBDIR _SCRIPT_DIR _PROJECT_ROOT

unset -f _prepend_ld

echo "[setup_env] PYTHONPATH=${PYTHONPATH}"
echo "[setup_env] PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}"
echo "[setup_env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
