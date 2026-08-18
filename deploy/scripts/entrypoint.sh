#!/usr/bin/env bash
# Docker entrypoint for faster-indextts-2.

set -euo pipefail

VENV_PY="${VENV_PY:-/workspace/indextts/deploy/.venv/bin/python}"

if [ -x "$VENV_PY" ]; then
  PY_LIBDIR="$($VENV_PY -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || true)"
  if [ -n "$PY_LIBDIR" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$PY_LIBDIR:"*) ;;
      *) export LD_LIBRARY_PATH="${PY_LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
fi

exec "$@"
