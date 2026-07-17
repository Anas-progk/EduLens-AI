"""
Swin-Tiny model for binary student engagement classification.

Improvements over baseline:
  1. drop_path_rate (stochastic depth):
       Randomly drops entire residual paths during training.
       Acts as a strong regulariser without hurting inference speed.

  2. Custom two-layer classification head:
       LayerNorm → Dropout(0.4) → Linear(768→256) → GELU → Dropout(0.3) → Linear(256→2)
       The baseline used a single Linear(768→2) which has no capacity to learn
       non-linear combinations of the rich 768-dim Swin features before classifying.

  3. Separate freeze / unfreeze utilities for progressive fine-tuning.

  4. Dedicated optimizer constructors:
       get_optimizer_frozen    — only optimises the head (phase 1)
       get_optimizer_unfrozen  — differential LR: backbone << head (phase 2)
"""

import torch
import torch.nn as nn
import timm


# ── Internal model class ───────────────────────────────────────────────────────

class _SwinEngagement(nn.Module):
    """Swin-Tiny backbone with a custom engagement classification head."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 768)
        return self.head(features)    # (B, num_classes)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_engagement_model(
    num_classes    : int   = 2,
    pretrained     : bool  = True,
    drop_rate      : float = 0.3,
    drop_path_rate : float = 0.1,
) -> _SwinEngagement:
    """
    Creates a Swin-Tiny model with a custom classification head.

    Args:
        num_classes:    Number of output classes (2 for binary engagement).
        pretrained:     Load ImageNet pretrained weights.
        drop_rate:      Dropout inside Swin attention / MLP layers.
        drop_path_rate: Stochastic depth rate (0 = disabled, 0.1 is a good default).
    """
    # num_classes=0 removes timm's default head → returns feature vectors
    backbone = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained     = pretrained,
        num_classes    = 0,
        drop_rate      = drop_rate,
        drop_path_rate = drop_path_rate,
    )

    in_features = backbone.num_features   # 768 for Swin-Tiny

    head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    model = _SwinEngagement(backbone, head)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: Swin-Tiny + custom head")
    print(f"  Backbone features  : {in_features}")
    print(f"  Total params       : {total/1e6:.1f}M")
    print(f"  Trainable params   : {trainable/1e6:.1f}M")
    print(f"  drop_rate          : {drop_rate}")
    print(f"  drop_path_rate     : {drop_path_rate}")
    print(f"  Output classes     : {num_classes}")

    return model


# ── Freeze / unfreeze ─────────────────────────────────────────────────────────

def freeze_backbone(model: _SwinEngagement) -> _SwinEngagement:
    """
    Freeze all backbone layers; only the head will receive gradients.
    Use for the first N epochs so the head stabilises before full fine-tuning.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"  Backbone FROZEN  |  Head trainable: {head_params/1e6:.3f}M params")
    return model


def unfreeze_all(model: _SwinEngagement) -> _SwinEngagement:
    """
    Unfreeze every layer for full fine-tuning.
    MUST use differential learning rates after calling this
    (see get_optimizer_unfrozen below).
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"  All layers UNFROZEN  |  Total trainable: {total/1e6:.1f}M params")
    return model


# ── Optimizer constructors ────────────────────────────────────────────────────

def get_optimizer_frozen(
    model        : _SwinEngagement,
    lr           : float,
    weight_decay : float,
) -> torch.optim.Optimizer:
    """
    AdamW only over head parameters.
    Used during the frozen backbone phase (phase 1).
    """
    return torch.optim.AdamW(
        model.head.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )


def get_optimizer_unfrozen(
    model        : _SwinEngagement,
    backbone_lr  : float,
    head_lr      : float,
    weight_decay : float,
) -> torch.optim.Optimizer:
    """
    AdamW with DIFFERENTIAL learning rates:
      backbone_lr (very small) → preserves pretrained ImageNet features
      head_lr     (larger)     → keeps the head adapting quickly

    Typical values: backbone_lr=3e-5, head_lr=2e-4.
    Using a single LR for both tends to either under-train the head or
    catastrophically forget the backbone's pretrained features.
    """
    return torch.optim.AdamW(
        [
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': model.head.parameters(),     'lr': head_lr},
        ],
        weight_decay = weight_decay,
    )
