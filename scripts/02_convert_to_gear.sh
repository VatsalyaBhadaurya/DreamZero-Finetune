#!/bin/bash
# ============================================================================
# Step 2: Convert your LeRobot v2 dataset -> GEAR metadata (modality.json,
# embodiment.json, stats.json, relative_stats_dreamzero.json, tasks/episodes).
# Writes new files under $DATA_ROOT/meta/.  Re-run with --force to overwrite.
#   source scripts/config.env && bash scripts/02_convert_to_gear.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
cd "$DREAMZERO_ROOT"

$PY scripts/data/convert_lerobot_to_gear.py \
    --dataset-path "$DATA_ROOT" \
    --embodiment-tag "$EMB" \
    --state-keys "$STATE_KEYS" \
    --action-keys "$ACTION_KEYS" \
    --relative-action-keys $RELATIVE_ACTION_KEYS \
    --task-key "$TASK_KEY" \
    --fps "$FPS" \
    --action-horizon 24 \
    "$@"   # pass --force to overwrite existing metadata

echo ">> Conversion done. Generated under $DATA_ROOT/meta/:"
ls -1 "$DATA_ROOT/meta/"
