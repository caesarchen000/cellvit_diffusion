#!/bin/bash
# Training script for CellViT Diffusion model
# 
# Usage:
#   bash scripts/train_cellvit_diffusion.sh
#
# Make sure to:
#   1. Start visdom server: visdom (or python -m visdom.server)
#   2. Organize your data in the format: data_dir/images/*.png and data_dir/masks/*.png
#   3. Adjust parameters below as needed

# Activate conda environment
# conda activate seg

# Start visdom server in background (if not already running)
# python -m visdom.server -port 8850 &

# Training parameters
DATA_DIR="../train"  # Path to training data (supports nested structure: DATA_DIR/ID/images/ and DATA_DIR/ID/masks/)
OUT_DIR="./results/cellvit_diffusion"
IMAGE_SIZE=256
BATCH_SIZE=2
LEARNING_RATE=1e-4
DIFFUSION_STEPS=6000
LOG_INTERVAL=100
SAVE_INTERVAL=5000

# Find latest checkpoint to resume from
LATEST_CHECKPOINT=$(ls -t "$OUT_DIR"/savedmodel*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_checkpoint $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting from scratch"
    RESUME_ARG=""
fi

# Run training
cd /home/caesar/Desktop/seg/MedSegDiff
python scripts/segmentation_train.py \
    --data_name "CUSTOM" \
    --data_dir "$DATA_DIR" \
    --model_arch "cellvit" \
    --image_size $IMAGE_SIZE \
    --in_ch 4 \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --diffusion_steps $DIFFUSION_STEPS \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --out_dir "$OUT_DIR" \
    --gpu_dev "0" \
    --cellvit_embed_dim 384 \
    --cellvit_depth 12 \
    --cellvit_heads 6 \
    $RESUME_ARG

