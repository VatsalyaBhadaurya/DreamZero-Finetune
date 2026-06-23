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

# No lerobot dependency: we write the v2.0 layout directly (pyarrow + ffmpeg).
$PY -c "import h5py, pyarrow, numpy" 2>/dev/null || $PY -m pip install --user h5py pyarrow numpy
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not on PATH (needed for H.264 video encode)"; exit 1; }

LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

$PY "$HERE/hdf5_to_lerobot.py" \
    --src "$HDF5_SRC" \
    --out "$DATA_ROOT" \
    --fps "$FPS" \
    --task "$TASK_INSTRUCTION" \
    $LIMIT_ARG

echo ">> LeRobot dataset ready at $DATA_ROOT"
