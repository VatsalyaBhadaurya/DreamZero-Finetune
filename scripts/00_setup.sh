#!/bin/bash
# ============================================================================
# Step 0: Clone DreamZero, install deps, download all checkpoints.
# Run on the Linux/CUDA box where you'll train.
#   source scripts/config.env && bash scripts/00_setup.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

# --- Clone repo --------------------------------------------------------------
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo ">> Cloning dreamzero into $DREAMZERO_ROOT"
    git clone https://github.com/dreamzero0/dreamzero "$DREAMZERO_ROOT"
else
    echo ">> Repo already present at $DREAMZERO_ROOT"
fi
cd "$DREAMZERO_ROOT"

# --- Python env + deps -------------------------------------------------------
# Requires Python 3.10/3.11, PyTorch 2.8+, CUDA 12.9+ per the repo.
echo ">> Installing dependencies (pyproject.toml)"
$PY -m pip install --upgrade pip
$PY -m pip install -e .
$PY -m pip install "huggingface_hub[cli]" deepspeed wandb decord

# --- Download checkpoints -----------------------------------------------------
mkdir -p "$DREAMZERO_ROOT/checkpoints"

echo ">> Wan2.2-TI2V-5B backbone (diffusion + T5 + VAE)"
[ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null || true)" ] && \
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"

echo ">> CLIP image encoder (from Wan2.1 repo — only the CLIP .pth is used)"
[ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ] && \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        --local-dir "$IMAGE_ENCODER_DIR"

echo ">> umt5-xxl tokenizer"
[ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null || true)" ] && \
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"

echo ">> DreamZero-AgiBot base checkpoint (LoRA fine-tune starts from this, ~45GB)"
[ -z "$(ls -A "$BASE_CKPT" 2>/dev/null || true)" ] && \
    huggingface-cli download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir "$BASE_CKPT"

echo ">> Setup complete."
