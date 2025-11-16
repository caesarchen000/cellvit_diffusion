"""
Quick smoke test for the CellViT diffusion backbone.

Run with:
    python scripts/example_cellvit_diffusion.py --image_size 256 --in_ch 4

This script instantiates the diffusion model with the CellViT architecture,
creates dummy data, and performs a forward pass to verify the wiring.
"""

import sys
import argparse
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).parent
medsegdiff_dir = script_dir.parent
seg_dir = medsegdiff_dir.parent
sys.path.insert(0, str(medsegdiff_dir))
sys.path.insert(0, str(seg_dir))

import torch

from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main() -> None:
    args = create_argparser().parse_args()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.eval()

    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_batch = torch.randn(
        batch_size,
        args.in_ch,
        args.image_size,
        args.image_size,
        device=device,
    )
    timesteps = torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(batch_size,),
        device=device,
    )

    with torch.no_grad():
        prediction, calibration = model(dummy_batch, timesteps)

    print("Prediction shape :", prediction.shape)
    print("Calibration shape:", calibration.shape)


def create_argparser() -> argparse.ArgumentParser:
    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            model_arch="cellvit",
            image_size=256,
            in_ch=4,
            cellvit_seg_channels=1,
            batch_size=2,
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

