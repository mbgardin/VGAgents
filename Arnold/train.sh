#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Where all training runs go
DUMP_ROOT="dumped/default"

# Training parameters
WAD="full_deathmatch"
DUMP_FREQ=1000
GPU_ID=-1

echo "Starting training..."
echo "WAD: ${WAD}"
echo "Dump freq: ${DUMP_FREQ}"

python arnold.py \
  --gpu_id ${GPU_ID} \
  --wad ${WAD} \
  --dump_freq ${DUMP_FREQ}