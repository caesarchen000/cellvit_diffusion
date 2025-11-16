#!/usr/bin/env python3
"""
Create demo images with three panels (left to right):
  1) Input image
  2) Merged ground-truth mask
  3) Predicted mask from diffusion model

It expects:
  - Predictions in a directory like: results/cellvit_diffusion/samples_fast_debug
    with filenames: <ID>_output*.jpg
  - Training data in nested layout:
       data_dir/ID/images/ID.png
       data_dir/ID/masks/*.png   (instance masks)
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make demo triptych images.")
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing predicted masks (e.g. results/.../samples_fast_debug).",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Training data directory with nested IDs (e.g. ../train).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save triptych demo images.",
    )
    parser.add_argument(
        "--pattern",
        default="*_output*.jpg",
        help="Glob pattern for prediction files inside pred_dir.",
    )
    return parser.parse_args()


def build_mask_index(data_dir: str) -> Dict[str, List[str]]:
    """Return {image_id: [mask_path, ...]} for nested dataset layout."""
    mask_paths = glob.glob(os.path.join(data_dir, "*", "masks", "*.png"))
    mask_index: Dict[str, List[str]] = {}
    for mask_path in mask_paths:
        image_id = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))
        mask_index.setdefault(image_id, []).append(mask_path)
    return mask_index


def load_and_merge_masks(
    mask_paths: List[str], target_hw: Tuple[int, int]
) -> Image.Image:
    """Merge instance masks (pixel-wise max), resize to target_hw, return PIL image."""
    if not mask_paths:
        raise ValueError("No mask paths provided for ID.")

    resize = transforms.Resize(target_hw, interpolation=InterpolationMode.NEAREST)
    merged = None
    for m_path in mask_paths:
        mask = Image.open(m_path).convert("L")
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0).float()
        mask_tensor = resize(mask_tensor)
        merged = (
            mask_tensor if merged is None else torch.maximum(merged, mask_tensor)
        )

    merged = (merged.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    # merged shape: (1, H, W)
    merged_np = merged.squeeze(0).cpu().numpy()
    return Image.fromarray(merged_np, mode="L")


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, args.pattern)))
    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction files found in {args.pred_dir} matching {args.pattern}"
        )

    mask_index = build_mask_index(args.data_dir)
    if not mask_index:
        raise FileNotFoundError(
            f"No masks found inside {args.data_dir}. Expected nested layout."
        )

    for pred_path in pred_paths:
        base = os.path.basename(pred_path)
        sample_id = base.split("_output")[0]

        # Input image path: data_dir/ID/images/ID.png
        img_path = os.path.join(args.data_dir, sample_id, "images", f"{sample_id}.png")
        if not os.path.isfile(img_path):
            print(f"[WARN] Input image not found for ID {sample_id}, skipping.")
            continue

        if sample_id not in mask_index:
            print(f"[WARN] No masks found for ID {sample_id}, skipping.")
            continue

        # Load prediction mask (grayscale), treat values as already in [0,255] or [0,1]
        pred_img = Image.open(pred_path).convert("L")
        pred_w, pred_h = pred_img.size
        target_hw = (pred_h, pred_w)

        # Load and resize input image to prediction size
        input_img = Image.open(img_path).convert("RGB")
        input_img = input_img.resize((pred_w, pred_h), resample=Image.BILINEAR)

        # Load and merge GT masks, resized to prediction size
        gt_img = load_and_merge_masks(mask_index[sample_id], target_hw)

        # Normalize prediction into [0,255] uint8 if needed
        pred_arr = np.array(pred_img, dtype=np.float32)
        if pred_arr.max() <= 1.0:
            pred_arr = (pred_arr * 255.0).clip(0, 255)
        pred_arr = pred_arr.astype(np.uint8)
        pred_img_uint8 = Image.fromarray(pred_arr, mode="L")

        # Convert GT and prediction to 3-channel for consistent visualization
        gt_vis = gt_img.convert("RGB")
        pred_vis = pred_img_uint8.convert("RGB")

        # Create triptych canvas: [input | gt | pred]
        canvas = Image.new("RGB", (pred_w * 3, pred_h))
        canvas.paste(input_img, (0, 0))
        canvas.paste(gt_vis, (pred_w, 0))
        canvas.paste(pred_vis, (pred_w * 2, 0))

        out_path = os.path.join(args.out_dir, f"{sample_id}_demo.png")
        canvas.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


