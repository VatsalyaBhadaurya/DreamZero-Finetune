#!/bin/bash
# ============================================================================
# Step 4: LoRA fine-tune DreamZero on your embodiment, Wan2.2-TI2V-5B backbone.
#   source scripts/config.env && bash scripts/04_train_lora_wan22.sh
# Override anything inline, e.g.:
#   NUM_GPUS=2 MAX_STEPS=30000 bash scripts/04_train_lora_wan22.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
export HYDRA_FULL_ERROR=1
cd "$DREAMZERO_ROOT"

# --- Pre-flight checks -------------------------------------------------------
[ -d "$DREAMZERO_ROOT/groot" ] || { echo "ERROR: no groot/ in $DREAMZERO_ROOT"; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "ERROR: dataset not found at $DATA_ROOT"; exit 1; }
[ -f "$DATA_ROOT/meta/embodiment.json" ] || { echo "ERROR: run 02_convert_to_gear.sh first (meta/embodiment.json missing)"; exit 1; }
[ -f "groot/vla/configs/data/dreamzero/${EMB}_relative.yaml" ] || { echo "ERROR: run 03_register_embodiment.py first"; exit 1; }
[ -d "$BASE_CKPT" ] || { echo "ERROR: DreamZero-AgiBot base missing at $BASE_CKPT (run 00_setup.sh)"; exit 1; }

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"

python -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" \
    report_to=wandb \
    data=dreamzero/${EMB}_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=$NUM_VIEWS \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    training_args.warmup_ratio=0.05 \
    save_steps=$SAVE_STEPS \
    save_strategy=steps \
    save_total_limit=10 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=$IMG_W \
    image_resolution_height=$IMG_H \
    save_lora_only=true \
    max_chunk_size=4 \
    ${EMB}_data_root=$DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$BASE_CKPT \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true

echo ">> Training finished. LoRA weights in: $OUTPUT_DIR"
