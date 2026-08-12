#!/usr/bin/env python3
"""CLI wrapper for IndexTTS voice post-processing.

Usage (from project root):
  uv run python -m indextts.utils.audio_tuning --input in.wav --output out.wav
  uv run python scripts/audio_tune.py --input in.wav --preset voice-clarity
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from indextts.utils.audio_tuning import _cli


if __name__ == "__main__":
    raise SystemExit(_cli())
