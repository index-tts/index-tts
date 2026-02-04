#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shentong/index-tts-main"
PY="/home/shentong/miniconda3/envs/indextts2/bin/python"

export PYTHONPATH="$ROOT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1

# 用法示例：
#   bash axis_pack/run.sh --lines 1
#   bash axis_pack/run.sh --lines 1 --axes
#   bash axis_pack/run.sh --lines 1 --axes emotion,speech_rate_anomaly
#   bash axis_pack/run.sh --lines 1 --conversation_only --merge
#   bash axis_pack/run.sh --lines 1 --with_conversation --merge

exec "$PY" -X faulthandler axis_pack/export_single_axis.py \
  --input axis_pack/all_conversations.jsonl \
  --cfg checkpoints/config.yaml \
  --model_dir checkpoints/IndexTTS-2 \
  --configs_dir axis_pack \
  --out_root axis_pack/output_export_by_tags \
  --merge \
  "$@"
