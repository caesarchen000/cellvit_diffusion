"""
CellViT diffusion backbone.

This module adapts the CellViT architecture to act as the score model in a
diffusion process. A noisy segmentation sample is processed jointly with the
corresponding conditioning image at every diffusion step. The conditioning path
re-uses the ViT encoder from CellViT, and its multi-scale feature maps are fused
with the segmentation branch by adaptive gates that are modulated by the current
diffusion timestep (and optional cell-level embeddings).

The overall design follows Fig. 3 in the project description:
    * Condition encoder (green)  – ViT backbone applied to the raw image.
    * Segmentation encoder (blue) – ViT backbone applied to the noisy mask.
    * Decoder (orange) – U-Net style decoder reused from CellViT.
    * Attentive fusion – lightweight gating blocks that blend features from both
      encoders while injecting the timestep embedding.

Author: GPT-5 Codex (2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT


def sinusoidal_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of shape `(B,)` with integer or float timesteps.
        dim: Dimension of the embedding.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        (B, dim) tensor with the sinusoidal embeddings.
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=device, dtype=torch.float32))
        * torch.arange(start=0, end=half, device=device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embeddings = torch.cat(
            [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
        )
    return embeddings


class AdaptiveFusionBlock(nn.Module):
    """Fuse segmentation and condition features with timestep awareness."""

    def __init__(
        self,
        seg_channels: int,
        cond_channels: int,
        out_channels: int,
        time_embed_dim: int,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.seg_proj = nn.Conv2d(seg_channels, out_channels, kernel_size=1)
        self.cond_proj = nn.Conv2d(cond_channels, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.use_residual = use_residual and seg_channels == out_channels

    def forward(
        self,
        seg_feat: torch.Tensor,
        cond_feat: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        seg_proj = self.seg_proj(seg_feat)
        cond_proj = self.cond_proj(cond_feat)
        time_proj = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)

        fused = seg_proj + cond_proj + time_proj
        gate = self.gate(fused)
        gated = fused * gate

        if self.use_residual:
            gated = gated + seg_proj

        return self.act(self.norm(gated))


@dataclass
class CellViTDiffusionConfig:
    """Configuration container for the diffusion backbone."""

    image_channels: int = 3
    seg_channels: int = 1
    output_channels: int = 1
    embed_dim: int = 384
    condition_embed_dim: Optional[int] = None
    depth: int = 12
    num_heads: int = 6
    extract_layers: Tuple[int, int, int, int] = (3, 6, 9, 12)
    patch_size: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    regression_loss: bool = False
    cell_embedding_dim: Optional[int] = None


class CellViTDiffusion(nn.Module):
    """Diffusion backbone that reuses CellViT blocks at every timestep."""

    def __init__(self, config: CellViTDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_size = config.patch_size
        self.seg_channels = config.seg_channels
        self.image_channels = config.image_channels
        self.embed_dim = config.embed_dim
        self.condition_embed_dim = config.condition_embed_dim or config.embed_dim
        self.extract_layers = list(config.extract_layers)
        assert (
            len(self.extract_layers) == 4
        ), "extract_layers must contain 4 layer indices."

        # Encoders -----------------------------------------------------------------
        self.condition_encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=0,
            embed_dim=self.condition_embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            norm_layer=nn.LayerNorm,
            extract_layers=self.extract_layers,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            in_chans=self.image_channels,
        )

        self.segmentation_encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=0,
            embed_dim=self.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            norm_layer=nn.LayerNorm,
            extract_layers=self.extract_layers,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            in_chans=self.seg_channels,
        )

        # Decoder ------------------------------------------------------------------
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.seg_channels, 32, 3, dropout=config.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=config.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=config.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=config.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=config.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=config.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=config.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=config.drop_rate)
        )

        self.output_channels = config.output_channels or self.seg_channels
        self.segmentation_decoder = self.create_upsampling_branch(self.output_channels)
        self.calibration_decoder = self.create_upsampling_branch(1)

        # Time and optional cell embeddings ----------------------------------------
        self.model_channels = self.embed_dim
        self.time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        if config.cell_embedding_dim is not None:
            self.cell_embed_proj = nn.Linear(
                config.cell_embedding_dim, self.time_embed_dim
            )
        else:
            self.cell_embed_proj = None

        # Fusion blocks -------------------------------------------------------------
        self.input_fusion = AdaptiveFusionBlock(
            seg_channels=self.seg_channels,
            cond_channels=self.image_channels,
            out_channels=self.seg_channels,
            time_embed_dim=self.time_embed_dim,
            use_residual=False,
        )

        fusion_blocks: List[AdaptiveFusionBlock] = []
        for _ in range(len(self.extract_layers)):
            fusion_blocks.append(
                AdaptiveFusionBlock(
                    seg_channels=self.embed_dim,
                    cond_channels=self.condition_embed_dim,
                    out_channels=self.embed_dim,
                    time_embed_dim=self.time_embed_dim,
                )
            )
        self.skip_fusions = nn.ModuleList(fusion_blocks)

    # --------------------------------------------------------------------- Helpers
    def create_upsampling_branch(self, num_classes: int) -> nn.ModuleDict:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.config.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.config.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.config.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.config.drop_rate),
            Conv2DBlock(256, 256, dropout=self.config.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.config.drop_rate),
            Conv2DBlock(128, 128, dropout=self.config.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_head = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.config.drop_rate),
            Conv2DBlock(64, 64, dropout=self.config.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        return nn.ModuleDict(
            {
                "bottleneck_upsampler": bottleneck_upsampler,
                "decoder3_upsampler": decoder3_upsampler,
                "decoder2_upsampler": decoder2_upsampler,
                "decoder1_upsampler": decoder1_upsampler,
                "decoder0_head": decoder0_head,
            }
        )

    @staticmethod
    def _tokens_to_feature_map(
        tokens: torch.Tensor, embed_dim: int, patch_dim: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert ViT tokens to a `(B, C, H, W)` feature map."""
        bsz = tokens.shape[0]
        spatial_tokens = tokens[:, 1:, :].transpose(1, 2)
        return spatial_tokens.view(bsz, embed_dim, *patch_dim)

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.ModuleDict,
    ) -> torch.Tensor:
        b4 = branch_decoder["bottleneck_upsampler"](z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder["decoder3_upsampler"](torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder["decoder2_upsampler"](torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder["decoder1_upsampler"](torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        output = branch_decoder["decoder0_head"](torch.cat([b0, b1], dim=1))
        return output

    # --------------------------------------------------------------------- Forward
    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cell_embeddings: Optional[torch.Tensor] = None,
        *,
        condition_image: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]:
        """Run a diffusion step.

        Args:
            inputs: `(B, image_channels + seg_channels, H, W)` tensor at timestep `t`.
            timesteps: `(B,)` tensor with the current diffusion step.
            cell_embeddings: Optional `(B, D_cell)` tensor with cell-level
                conditioning vectors.
            return_intermediate: If True, return a dict with intermediate
                activations for analysis; otherwise only return the prediction.

        Returns:
            Either `(prediction, calibration)` tensors or a dict with additional
            intermediate states.
        """
        del y
        if condition_image is None:
            assert (
                inputs.shape[1] >= self.image_channels + self.seg_channels
            ), "Input tensor does not contain enough channels."
            condition_image = inputs[:, : self.image_channels, ...]
            noisy_mask = inputs[:, self.image_channels :, ...]
        else:
            noisy_mask = inputs
        assert (
            noisy_mask.shape[-1] % self.patch_size == 0
            and noisy_mask.shape[-2] % self.patch_size == 0
        ), "Input mask size must be divisible by patch_size."
        assert condition_image.shape[-2:] == noisy_mask.shape[-2:]

        # --- Time embedding -------------------------------------------------------
        emb = sinusoidal_embedding(timesteps.to(noisy_mask.device), self.model_channels)
        emb = self.time_embed(emb)
        if self.cell_embed_proj is not None and cell_embeddings is not None:
            emb = emb + self.cell_embed_proj(cell_embeddings)

        # --- Encode condition & segmentation -------------------------------------
        _, _, condition_tokens = self.condition_encoder(condition_image)
        _, _, segmentation_tokens = self.segmentation_encoder(noisy_mask)

        patch_dim = (
            noisy_mask.shape[-2] // self.patch_size,
            noisy_mask.shape[-1] // self.patch_size,
        )

        cond_feats = [
            self._tokens_to_feature_map(tok, self.condition_embed_dim, patch_dim)
            for tok in condition_tokens
        ]
        seg_feats = [
            self._tokens_to_feature_map(tok, self.embed_dim, patch_dim)
            for tok in segmentation_tokens
        ]

        # Fuse skip connections (ordered from low to high resolution).
        fused_skips: List[torch.Tensor] = []
        for fusion_block, seg_feat, cond_feat in zip(
            self.skip_fusions, seg_feats, cond_feats
        ):
            fused_skips.append(fusion_block(seg_feat, cond_feat, emb))

        z1, z2, z3, z4 = fused_skips
        z0 = self.input_fusion(noisy_mask, condition_image, emb)

        prediction = self._forward_upsample(
            z0=z0,
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            branch_decoder=self.segmentation_decoder,
        )
        calibration = self._forward_upsample(
            z0=z0,
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            branch_decoder=self.calibration_decoder,
        )

        if return_intermediate:
            return {
                "prediction": prediction,
                "calibration": calibration,
                "time_embedding": emb,
                "fused_skips": torch.stack(
                    [feat.mean(dim=(2, 3)) for feat in fused_skips], dim=1
                ),
            }
        return prediction, calibration

    # ------------------------------------------------------------------ Utilities
    def convert_to_fp16(self):
        self.half()

    def convert_to_fp32(self):
        self.float()

    def load_part_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except RuntimeError:
                continue


__all__ = ["CellViTDiffusion", "CellViTDiffusionConfig"]
