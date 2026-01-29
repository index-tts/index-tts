#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shentong/index-tts-main"
PY="/home/shentong/miniconda3/envs/indextts2/bin/python"

# 必须：让 python 能 import 到仓库里的 indextts
export PYTHONPATH="$ROOT"
cd "$ROOT"

# 可选：固定 GPU（不想固定就删掉这行）
export CUDA_VISIBLE_DEVICES=0

exec "$PY" -X faulthandler axis_pack/export_single_axis.py \
  --input axis_pack/all_conversations.jsonl \
  --speaker_map axis_pack/speaker_map.json \
  --cfg checkpoints/config.yaml \
  --model_dir checkpoints/IndexTTS-2 \
  --configs_dir axis_pack/configs \
  --out_root axis_pack/output_export_by_tags \
  --with_conversation \
  "$@"
