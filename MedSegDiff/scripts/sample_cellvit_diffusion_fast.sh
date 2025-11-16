#!/bin/bash

# Fast sampling script for quick testing
# Uses fewer steps and ensemble members for faster results

# Configuration
DATA_DIR="../train"
OUT_DIR="results/cellvit_diffusion/samples_fast"

# Change to script directory first
cd /home/caesar/Desktop/seg/MedSegDiff

# Automatically find the latest EMA checkpoint
LATEST_CHECKPOINT=$(ls -t results/cellvit_diffusion/emasavedmodel*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No EMA checkpoint found in results/cellvit_diffusion/"
    exit 1
fi
MODEL_PATH="$LATEST_CHECKPOINT"
IMAGE_SIZE=256
BATCH_SIZE=1
NUM_SAMPLES=1  # Just 1 sample for quick test
NUM_ENSEMBLE=3  # Reduced from 10 to 3 for speed
DIFFUSION_STEPS=1000  # Reduced from 8000 to 1000 for speed (still good quality)
USE_DDIM=false
THRESHOLD=0.3  # Lower threshold to avoid all-black results
POST_PROCESS=true

# Create output directory
mkdir -p "$OUT_DIR"

# Activate conda environment
conda activate seg

echo "Using checkpoint: $MODEL_PATH"
echo "FAST MODE: Sampling with $DIFFUSION_STEPS diffusion steps and $NUM_ENSEMBLE ensemble members"
echo "This will be much faster (~5-10 minutes) for testing."

python scripts/segmentation_sample.py \
    --data_name "CUSTOM" \
    --data_dir "$DATA_DIR" \
    --model_arch "cellvit" \
    --image_size $IMAGE_SIZE \
    --in_ch 4 \
    --batch_size $BATCH_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --model_path "$MODEL_PATH" \
    --out_dir "$OUT_DIR" \
    --gpu_dev "0" \
    --num_samples $NUM_SAMPLES \
    --num_ensemble $NUM_ENSEMBLE \
    --use_ddim $USE_DDIM \
    --clip_denoised true \
    --post_process $POST_PROCESS \
    --threshold $THRESHOLD

echo "Fast sampling complete! Results saved to: $OUT_DIR"
echo ""
echo "For maximum quality, use the regular sample_cellvit_diffusion.sh script"
echo "which uses 8000 steps and 10 ensemble members (but takes 1.5-4 hours)."

