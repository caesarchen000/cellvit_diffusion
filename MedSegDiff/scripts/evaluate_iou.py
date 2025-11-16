#!/usr/bin/env python3
"""
Compute IoU, Dice, and AP between predicted diffusion masks and merged ground-truth
instance masks.

Steps (per sample):
1. Merge all instance masks for an ID at native resolution.
2. Resize the merged mask to the prediction resolution (nearest-neighbor).
3. Use the prediction as probabilities for AP, threshold it for IoU/Dice.
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IoU for diffusion masks.")
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory that contains predicted masks (e.g., results/.../samples_fast_debug)",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Training data directory with nested IDs (../train).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to prediction after normalization.",
    )
    parser.add_argument(
        "--pattern",
        default="*_output*.jpg",
        help="Glob pattern for prediction files inside pred_dir.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for tensor ops (cpu or cuda:<id>).",
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
) -> torch.Tensor:
    """Merge instance masks (pixel-wise max) and resize to target_hw."""
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
    return merged


def load_prediction(pred_path: str, threshold: float) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Load prediction image, treat values as probabilities, also return thresholded mask.

    Returns:
        sample_id: ID string parsed from filename prefix before '_output'.
        prob_tensor: float tensor in [0,1] with shape (1, H, W).
        bin_tensor: thresholded tensor in {0,1} with shape (1, H, W).
    """
    base = os.path.basename(pred_path)
    sample_id = base.split("_output")[0]
    pred_img = Image.open(pred_path).convert("L")
    prob_tensor = transforms.ToTensor()(pred_img)  # assume already in [0,1]
    bin_tensor = (prob_tensor >= threshold).float()
    return sample_id, prob_tensor, bin_tensor


def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    assert pred.shape == gt.shape, "Prediction and GT shapes must match"
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = torch.logical_and(pred_b, gt_b).sum().item()
    union = torch.logical_or(pred_b, gt_b).sum().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    assert pred.shape == gt.shape, "Prediction and GT shapes must match"
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = torch.logical_and(pred_b, gt_b).sum().item()
    denom = pred_b.sum().item() + gt_b.sum().item()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def main():
    args = parse_args()

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

    iou_scores = []
    dice_scores = []
    ap_scores = []
    missing = []

    for pred_path in pred_paths:
        sample_id, prob_mask, bin_mask = load_prediction(pred_path, args.threshold)
        target_hw = prob_mask.shape[-2:]

        if sample_id not in mask_index:
            missing.append(sample_id)
            continue

        gt_mask = load_and_merge_masks(mask_index[sample_id], target_hw)
        gt_mask = gt_mask.to(args.device)

        # Metrics
        iou = compute_iou(bin_mask.to(args.device), gt_mask)
        dice = compute_dice(bin_mask.to(args.device), gt_mask)

        # Average precision at pixel level using probabilities
        y_true = gt_mask.view(-1).cpu().numpy()
        y_score = prob_mask.to(args.device).view(-1).detach().cpu().numpy()
        try:
            ap = average_precision_score(y_true, y_score)
        except ValueError:
            # Happens if only one class present; fall back to 1.0 or 0.0 similarly to IoU/Dice edge cases
            ap = 1.0 if y_true.sum() == 0 else 0.0

        iou_scores.append(iou)
        dice_scores.append(dice)
        ap_scores.append(ap)

        print(f"{sample_id}: IoU={iou:.4f}, Dice={dice:.4f}, AP={ap:.4f}")

    if iou_scores:
        mean_iou = sum(iou_scores) / len(iou_scores)
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_ap = sum(ap_scores) / len(ap_scores)
        print(
            f"\nEvaluated {len(iou_scores)} samples. "
            f"Mean IoU = {mean_iou:.4f}, Mean Dice = {mean_dice:.4f}, Mean AP = {mean_ap:.4f}"
        )
    else:
        print("No IoU scores computed (no matching IDs).")

    if missing:
        print(
            f"\nMissing masks for {len(missing)} prediction(s): "
            + ", ".join(sorted(set(missing)))
        )


if __name__ == "__main__":
    main()

