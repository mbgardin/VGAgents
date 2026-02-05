#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"   # run from repo root even if launched elsewhere

# Find newest checkpoint (by modified time)
LATEST_CKPT="$(ls -t dumped/default/*/periodic-*.pth 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoints found at dumped/default/*/periodic-*.pth"
  echo "Train with something like: python arnold.py --gpu_id -1 --wad full_deathmatch --dump_freq 1000"
  exit 1
fi

echo "Using latest checkpoint: ${LATEST_CKPT}"

# Run evaluation + render
python arnold.py \
  --gpu_id -1 \
  --wad full_deathmatch \
  --dump_path . \
  --evaluate 1 \
  --visualize true \
  --reload "${LATEST_CKPT}"