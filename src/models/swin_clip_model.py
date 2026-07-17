"""
SwinClipModel -- Swin-Tiny backbone + Temporal Transformer for clip-level engagement.

Architecture:
  1. Shared Swin-Tiny backbone extracts per-frame features  (B*T, 768)
  2. TemporalTransformer aggregates across T frames via CLS-token attention  (B, 768)
  3. Classification head maps to 2 classes  (B, 2)

Why this architecture:
  - Engagement is BEHAVIORAL: gaze drift, posture shifts, attention fluctuations.
    These dynamics are invisible in single frames but apparent across 8 consecutive frames.
  - Shared backbone: weight tying means all T frames see the same feature extractor,
    which is efficient and regularises temporal feature alignment.
  - CLS token: standard, proven approach (BERT, ViT) for sequence-level classification.
    The model learns which frames to attend to via self-attention weights.
  - Pre-LN TransformerEncoder (norm_first=True): more stable training with lower LRs.

Memory:
  batch_size=4 clips * 8 frames = 32 forward passes through Swin-Tiny backbone.
  Identical memory to frame-level batch_size=32. No extra GPU RAM required.
"""

import math
import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# TemporalTransformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    CLS-token Transformer for aggregating T frame features into one clip feature.

    Input:  (B, T, D)   -- T per-frame feature vectors of dimension D
    Output: (B, D)      -- single clip-level feature vector (CLS token output)

    Design choices:
      - CLS token prepended: clip classification via CLS output (avoids pooling ambiguity)
      - Learnable positional embedding: encodes frame order within clip
      - Pre-LN (norm_first=True): more stable gradients vs post-LN at low LRs
      - 2 transformer layers: captures within-clip dynamics without overfitting small data
      - d_ff = D (not 4D): smaller FFN due to limited clip training data (~333 NE clips)
    """

    def __init__(
        self,
        d_model   : int   = 768,
        n_heads   : int   = 8,
        n_layers  : int   = 2,
        dropout   : float = 0.1,
        max_len   : int   = 9,    # CLS + up to 8 frames
    ):
        super().__init__()

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable positional encoding  (CLS + T frame positions)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = d_model,   # compact FFN
            dropout     = dropout,
            activation  = "gelu",
            batch_first = True,
            norm_first  = True,          # Pre-LN: more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)

        # Initialise weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)  per-frame features
        Returns:
            (B, D)  CLS token output = clip representation
        """
        B, T, D = x.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)            # (B, T+1, D)

        # Add positional embedding (truncate to actual sequence length)
        x = x + self.pos_embed[:, : T + 1, :]

        # Transformer encoding
        x = self.encoder(x)                         # (B, T+1, D)

        # Return CLS token (index 0)
        return self.norm(x[:, 0])                   # (B, D)


# ---------------------------------------------------------------------------
# Full clip model
# ---------------------------------------------------------------------------

class SwinClipModel(nn.Module):
    """
    Swin-Tiny + TemporalTransformer + classification head.

    forward() accepts (B, T, C, H, W) clips and returns (B, num_classes) logits.
    The backbone is shared across all T frames (same weights, called T times per clip).
    """

    def __init__(self, backbone: nn.Module, temporal: TemporalTransformer, head: nn.Module):
        super().__init__()
        self.backbone = backbone     # Swin-Tiny, returns (B*T, 768)
        self.temporal = temporal     # TemporalTransformer, returns (B, 768)
        self.head     = head         # Classification head, returns (B, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) -- batch of B clips, each with T frames
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Extract per-frame features via shared backbone
        frames   = x.view(B * T, C, H, W)           # (B*T, C, H, W)
        features = self.backbone(frames)             # (B*T, 768)
        features = features.view(B, T, -1)           # (B, T, 768)

        # Aggregate over time
        clip_feat = self.temporal(features)          # (B, 768)

        # Classify
        return self.head(clip_feat)                  # (B, num_classes)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_clip_model(
    num_classes    : int   = 2,
    pretrained     : bool  = True,
    drop_rate      : float = 0.25,
    drop_path_rate : float = 0.1,
    n_heads        : int   = 8,
    n_layers       : int   = 2,
    temporal_drop  : float = 0.1,
    n_frames       : int   = 8,
) -> SwinClipModel:
    """
    Builds the full clip-level engagement classification model.

    Args:
        num_classes:    Output classes (2 for binary).
        pretrained:     Load ImageNet weights for Swin backbone.
        drop_rate:      Dropout inside Swin.
        drop_path_rate: Stochastic depth rate.
        n_heads:        Attention heads in TemporalTransformer.
        n_layers:       Transformer encoder layers.
        temporal_drop:  Dropout in TemporalTransformer.
        n_frames:       Expected clip length (for pos embedding size = n_frames+1).
    """
    # Swin-Tiny backbone (no classification head -- returns 768-dim features)
    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained     = pretrained,
        num_classes    = 0,
        drop_rate      = drop_rate,
        drop_path_rate = drop_path_rate,
    )
    d_model = backbone.num_features   # 768

    temporal = TemporalTransformer(
        d_model  = d_model,
        n_heads  = n_heads,
        n_layers = n_layers,
        dropout  = temporal_drop,
        max_len  = n_frames + 1,   # CLS + frames
    )

    head = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Dropout(p=0.4),
        nn.Linear(d_model, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    model = SwinClipModel(backbone, temporal, head)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: Swin-Tiny + TemporalTransformer (clip-level)")
    print(f"  Backbone features  : {d_model}")
    print(f"  Temporal layers    : {n_layers}  heads={n_heads}")
    print(f"  Clip length        : {n_frames} frames")
    print(f"  Total params       : {total/1e6:.1f}M")
    print(f"  Trainable params   : {trainable/1e6:.1f}M")
    print(f"  drop_rate          : {drop_rate}")
    print(f"  drop_path_rate     : {drop_path_rate}")

    return model


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def freeze_backbone(model: SwinClipModel) -> SwinClipModel:
    """
    Freeze Swin backbone. Only temporal transformer + head receive gradients.
    Use during Phase 1 (BOOTSTRAP) to let the temporal head learn from ImageNet features.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.temporal.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    frozen    = sum(p.numel() for p in model.backbone.parameters())
    trainable = sum(p.numel() for p in model.temporal.parameters()) +                 sum(p.numel() for p in model.head.parameters())
    print(f"  Backbone FROZEN ({frozen/1e6:.1f}M)  |  "
          f"Temporal+Head trainable ({trainable/1e6:.3f}M)")
    return model


def unfreeze_all(model: SwinClipModel) -> SwinClipModel:
    """
    Unfreeze all layers for full fine-tuning.
    Always pair with differential learning rates (backbone_lr << head_lr).
    """
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  All layers UNFROZEN  |  Total trainable: {total/1e6:.1f}M")
    return model


def partial_unfreeze_last_stages(model: SwinClipModel, n_stages: int = 1) -> SwinClipModel:
    """
    Selectively unfreeze only the LAST n_stages of the Swin backbone.

    WHY selective unfreeze (instead of full unfreeze)?
    ─────────────────────────────────────────────────
    Full unfreeze → 27.5M backbone params trainable on ~1000 clips
    → catastrophic overfitting (train F1 → 0.98 while val F1 drops from 0.85 → 0.60).

    Selective unfreeze → only the topmost, most semantic Swin stage adapts.
    Early stages (patch_embed, layers[0-2]) remain frozen — they provide stable
    low-level features (edges, textures, shapes) that generalise across domains.
    Only the final 768-dim stage gets engagement-specific fine-tuning.

    Swin-Tiny stage breakdown:
      layers[0]: Stage 0  (96-dim,  2 blocks,  ~0.8M params)  ← always frozen
      layers[1]: Stage 1  (192-dim, 2 blocks,  ~2.1M params)  ← always frozen
      layers[2]: Stage 2  (384-dim, 6 blocks, ~13.8M params)  ← frozen unless n_stages≥3
      layers[3]: Stage 3  (768-dim, 2 blocks,  ~8.6M params)  ← unfrozen with n_stages=1

    n_stages=1 (recommended): ~8.6M backbone + 7.3M temporal+head = ~15.9M total trainable
    n_stages=2:              ~22.4M backbone + 7.3M temporal+head = ~29.7M total trainable

    Args:
        model:    SwinClipModel instance.
        n_stages: How many final Swin stages to unfreeze (1 recommended for small datasets).
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Always unfreeze temporal transformer + classification head
    for param in model.temporal.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    # 3. Unfreeze the last n_stages of the Swin backbone
    total_stages = len(model.backbone.layers)   # 4 for Swin-Tiny
    start_stage  = max(0, total_stages - n_stages)

    for i in range(start_stage, total_stages):
        for param in model.backbone.layers[i].parameters():
            param.requires_grad = True

    # 4. Also unfreeze the Swin final LayerNorm (processes stage-3 output)
    if hasattr(model.backbone, "norm"):
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Backbone: last {n_stages} stage(s) UNFROZEN  |  "
        f"Frozen={frozen/1e6:.1f}M  Trainable={trainable/1e6:.1f}M"
    )
    return model


# ---------------------------------------------------------------------------
# Optimizer constructors
# ---------------------------------------------------------------------------

def get_optimizer_frozen(
    model        : SwinClipModel,
    lr           : float,
    weight_decay : float,
) -> torch.optim.Optimizer:
    """
    AdamW only over temporal transformer + head (backbone frozen).
    Phase 1: BOOTSTRAP.
    """
    params = list(model.temporal.parameters()) + list(model.head.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_optimizer_unfrozen(
    model        : SwinClipModel,
    backbone_lr  : float,
    temporal_lr  : float,
    head_lr      : float,
    weight_decay : float,
) -> torch.optim.Optimizer:
    """
    AdamW with three-level differential learning rates.
    Use for FULL backbone unfreeze (not recommended for small datasets).

    Typical: backbone_lr=5e-6, temporal_lr=1e-5, head_lr=2e-5
    """
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.temporal.parameters(), "lr": temporal_lr},
            {"params": model.head.parameters(),     "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def get_optimizer_partial(
    model        : SwinClipModel,
    backbone_lr  : float,
    temporal_lr  : float,
    head_lr      : float,
    weight_decay : float,
) -> torch.optim.Optimizer:
    """
    AdamW for PARTIAL unfreeze (after partial_unfreeze_last_stages()).

    Only includes backbone params that have requires_grad=True (the unfrozen last stages).
    This avoids creating wasteful optimizer state for frozen backbone layers.

    Three param groups with differential LRs:
      backbone_params (last stage only, tiny LR) -- slow domain adaptation
      temporal_params (small LR)                 -- intermediate adaptation
      head_params     (slightly larger LR)        -- fastest adaptation

    Typical: backbone_lr=2e-6, temporal_lr=8e-6, head_lr=1.5e-5
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    temporal_params = list(model.temporal.parameters())
    head_params     = list(model.head.parameters())

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": temporal_params, "lr": temporal_lr})
    param_groups.append({"params": head_params,     "lr": head_lr})

    print(
        f"  Optimizer [partial]: "
        f"backbone_last={len(backbone_params)} params (lr={backbone_lr:.1e})  "
        f"temporal={len(temporal_params)} params (lr={temporal_lr:.1e})  "
        f"head={len(head_params)} params (lr={head_lr:.1e})"
    )
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
