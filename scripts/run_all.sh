#!/bin/bash
# ============================================================================
# End-to-end: GEAR conversion -> register embodiment -> train.
# Assumes the LeRobot v2.0 dataset already exists at $DATA_ROOT
# (i.e. you've already run 01b_hdf5_to_lerobot.sh). If it doesn't, this also
# runs the HDF5->LeRobot conversion first.
#
#   source scripts/config.env && bash scripts/run_all.sh
# Override on the fly:
#   NUM_GPUS=1 MAX_STEPS=20000 bash scripts/run_all.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

echo "============================================================"
echo " DreamZero pipeline: $EMB"
echo "   DATA_ROOT     = $DATA_ROOT"
echo "   DREAMZERO     = $DREAMZERO_ROOT"
echo "   GPUS / STEPS  = $NUM_GPUS / $MAX_STEPS"
echo "   wandb         = $REPORT_TO ($WANDB_ENTITY/$WANDB_PROJECT)"
echo "============================================================"

# 0) HDF5 -> LeRobot v2.0 (skip if already converted)
if [ ! -f "$DATA_ROOT/meta/info.json" ]; then
    echo ">> [0/3] dataset not found — running HDF5->LeRobot conversion"
    bash "$HERE/01b_hdf5_to_lerobot.sh"
else
    echo ">> [0/3] LeRobot dataset present at $DATA_ROOT — skipping conversion"
fi

# 1) LeRobot v2.0 -> GEAR metadata (skip if already done)
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo ">> [1/3] GEAR conversion"
    bash "$HERE/02_convert_to_gear.sh"
else
    echo ">> [1/3] GEAR metadata present (meta/embodiment.json) — skipping"
fi

# 2) Register embodiment + auto-patch base config (idempotent)
echo ">> [2/3] registering embodiment + patching configs"
$PY "$HERE/03_register_embodiment.py"

# 3) Train
echo ">> [3/3] launching training"
bash "$HERE/04_train_lora_wan22.sh"

echo ">> Pipeline complete. Checkpoints in: $OUTPUT_DIR"
