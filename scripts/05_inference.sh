#!/bin/bash
# ============================================================================
# Step 5: Serve your fine-tuned model (inference socket server).
# Point --model-path at your trained LoRA output (or merged checkpoint).
#   source scripts/config.env && bash scripts/05_inference.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
cd "$DREAMZERO_ROOT"

PORT="${PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-$OUTPUT_DIR}"
SERVE_GPUS="${SERVE_GPUS:-$NUM_GPUS}"

python -m torch.distributed.run --standalone --nproc_per_node="$SERVE_GPUS" \
    socket_test_optimized_AR.py \
    --port "$PORT" \
    --enable-dit-cache \
    --model-path "$MODEL_PATH"
