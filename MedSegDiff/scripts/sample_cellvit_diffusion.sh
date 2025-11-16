#!/bin/bash

# Script to sample from trained CellViT diffusion model with more diffusion steps
# More steps = better denoising, less noise

# Configuration
DATA_DIR="../train"
OUT_DIR="results/cellvit_diffusion/samples"

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
NUM_SAMPLES=5
NUM_ENSEMBLE=5  # Number of samples to ensemble
DIFFUSION_STEPS=6000  # Increase this for better denoising (default is 1000, 6000-8000 for best quality)
USE_DDIM=false  # Set to true for faster sampling with fewer steps needed
THRESHOLD=0.5

# Create output directory
mkdir -p "$OUT_DIR"

# Activate conda environment
conda activate seg

echo "Using checkpoint: $MODEL_PATH"
echo "Sampling with $DIFFUSION_STEPS diffusion steps..."
echo "This will take longer but produce less noisy results."

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
    --clip_denoised true

echo "Sampling complete! Results saved to: $OUT_DIR"
echo ""
echo "Using $DIFFUSION_STEPS steps for maximum quality."
echo "If you want even better quality (but slower), change DIFFUSION_STEPS to 8000 in the script."

