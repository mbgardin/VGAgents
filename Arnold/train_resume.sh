#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR="dumped/default/latest"
WAD="full_deathmatch"
DUMP_FREQ=1000
GPU_ID=-1

mkdir -p "$RUN_DIR"

# Find the latest checkpoint in this fixed run dir
LATEST_CKPT="$(ls -t "$RUN_DIR"/periodic-*.pth 2>/dev/null | head -n 1 || true)"

if [[ -n "$LATEST_CKPT" ]]; then
  echo "Resuming training from: $LATEST_CKPT"
  python arnold.py \
    --gpu_id "$GPU_ID" \
    --wad "$WAD" \
    --dump_path "$RUN_DIR" \
    --dump_freq "$DUMP_FREQ" \
    --reload "$LATEST_CKPT"
else
  echo "No checkpoint found in $RUN_DIR — starting fresh training."
  python arnold.py \
    --gpu_id "$GPU_ID" \
    --wad "$WAD" \
    --dump_path "$RUN_DIR" \
    --dump_freq "$DUMP_FREQ"
fi