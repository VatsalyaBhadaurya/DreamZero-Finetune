#!/bin/bash
# ============================================================================
# Step 1b (SO-101 HDF5 only): convert HDF5 episodes -> LeRobot v2.0 dataset.
# Run BEFORE 02_convert_to_gear.sh. Smoke-test with LIMIT=3 first.
#   source scripts/config.env && bash scripts/01b_hdf5_to_lerobot.sh
#   LIMIT=3 bash scripts/01b_hdf5_to_lerobot.sh     # quick 3-episode test
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

# lerobot provides LeRobotDataset.create (handles parquet + mp4 + meta).
$PY -c "import lerobot" 2>/dev/null || $PY -m pip install --user lerobot h5py

LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

$PY "$HERE/hdf5_to_lerobot.py" \
    --src "$HDF5_SRC" \
    --out "$DATA_ROOT" \
    --repo-id "${EMB}/pick_block" \
    --fps "$FPS" \
    --task "$TASK_INSTRUCTION" \
    $LIMIT_ARG

echo ">> LeRobot dataset ready at $DATA_ROOT"
