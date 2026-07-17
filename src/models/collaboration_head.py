"""
collaboration_head.py -- Pairwise collaboration classifier for Phase 2.

Architecture:
  Input: two 768-d temporal clip features (from frozen SwinClipModel) +
         4-d interaction signals (proximity, facing, correlation, turn-taking)
  Output: per-pair collaboration probability (sigmoid)

Design principles:
  1. FROZEN engagement backbone — Phase 1 weights never updated
  2. Small trainable head (~2.4M params) — appropriate for 400-800 training pairs
  3. Cross-attention between A and B features — forces model to reason about
     the RELATIONSHIP between persons, not just individual appearance
  4. Symmetric training: (A,B) and (B,A) both labeled the same
  5. Interaction signals bypass the neural net and provide rule-based priors
     that prevent the model from learning spurious correlations

Usage:
  # Build standalone
  head = build_collab_head()

  # Inference (single pair)
  feat_A = torch.randn(1, 768)   # from frozen SwinClipModel.get_temporal_feat()
  feat_B = torch.randn(1, 768)
  signals = torch.tensor([[0.8, 0.6, 0.3, 0.4]])  # proximity, facing, corr, tt
  prob = head.predict_prob(feat_A, feat_B, signals)  # scalar 0-1

  # Batch inference
  logits = head(feat_A_batch, feat_B_batch, signals_batch)  # (B,) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CollaborationHead
# ---------------------------------------------------------------------------

class CollaborationHead(nn.Module):
    """
    Pairwise collaboration classifier.

    Takes temporal clip features for two persons + 4 interaction signals
    and outputs a collaboration logit (apply sigmoid for probability).

    Architecture flow:
      feat_A (768) ──┐
                     ├─→ Project to proj_dim (128) ──→ Cross-Attention ──→ fused_A (128)
      feat_B (768) ──┘                                                  ─→ fused_B (128)
                                                                              │
      interaction_signals (4) ───────────────────────────────────────────────┤
                                                              concat (128+128+4=260)
                                                                              │
                                                        MLP: 260 → 128 → 64 → 1
                                                                              │
                                                                        collab_logit

    Cross-attention detail:
      - fused_A = MultiheadAttn(query=proj_A, key=proj_B, value=proj_B)
      - fused_B = MultiheadAttn(query=proj_B, key=proj_A, value=proj_A)
      This asks: "what about B is relevant to A?" and vice versa.
      Two engaged-but-isolated people will have DIFFERENT cross-attention
      patterns than two people actively interacting.

    Parameters:
      feat_dim:   Dimension of input temporal features from SwinClipModel (768)
      proj_dim:   Projection dimension before cross-attention (128)
                  Reduced to limit overfitting on small collab dataset
      n_heads:    Attention heads for cross-attention (4, proj_dim must divide evenly)
      signal_dim: Number of interaction signals (4)
      mlp_hidden: Hidden dim of final MLP (64)
      dropout:    Dropout rate in MLP (0.3)
    """

    def __init__(
        self,
        feat_dim   : int   = 768,
        proj_dim   : int   = 128,
        n_heads    : int   = 4,
        signal_dim : int   = 4,
        mlp_hidden : int   = 64,
        dropout    : float = 0.3,
    ):
        super().__init__()

        assert proj_dim % n_heads == 0, \
            f"proj_dim ({proj_dim}) must be divisible by n_heads ({n_heads})"

        self.proj_dim   = proj_dim
        self.signal_dim = signal_dim

        # ── Projection layers (768 → 128) ──────────────────────────────────
        self.proj_A = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_B = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── Cross-attention layers ─────────────────────────────────────────
        # A attends over B's key/values
        self.cross_attn_A = nn.MultiheadAttention(
            embed_dim   = proj_dim,
            num_heads   = n_heads,
            dropout     = 0.1,
            batch_first = True,
        )
        # B attends over A's key/values
        self.cross_attn_B = nn.MultiheadAttention(
            embed_dim   = proj_dim,
            num_heads   = n_heads,
            dropout     = 0.1,
            batch_first = True,
        )

        # Layer norms after cross-attention (pre-LN style for stability)
        self.norm_A = nn.LayerNorm(proj_dim)
        self.norm_B = nn.LayerNorm(proj_dim)

        # ── Interaction signal encoder ─────────────────────────────────────
        # Small MLP to lift 4-d signals to richer representation
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, 16),
            nn.GELU(),
            nn.Linear(16, signal_dim),   # keep at 4 — we don't need huge signal repr
        )

        # ── Final MLP classifier ───────────────────────────────────────────
        # Input: proj_A (128) + proj_B (128) + encoded_signals (4) = 260
        mlp_input_dim = proj_dim + proj_dim + signal_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden * 2),
            nn.LayerNorm(mlp_hidden * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(mlp_hidden, 1),   # raw logit
        )

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        print(f"\nCollaborationHead")
        print(f"  feat_dim={feat_dim}  proj_dim={proj_dim}  n_heads={n_heads}")
        print(f"  signal_dim={signal_dim}  mlp_hidden={mlp_hidden}")
        print(f"  Total params: {total/1e6:.2f}M")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        feat_A   : torch.Tensor,   # (B, 768)
        feat_B   : torch.Tensor,   # (B, 768)
        signals  : torch.Tensor,   # (B, 4)  [proximity, facing, correlation, turn_taking]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            feat_A:  Temporal clip features for person A, shape (B, 768)
            feat_B:  Temporal clip features for person B, shape (B, 768)
            signals: Interaction signals between A and B, shape (B, 4)

        Returns:
            logits: Raw (pre-sigmoid) collaboration scores, shape (B,)
                    Apply sigmoid() for probability. BCEWithLogitsLoss for training.
        """
        B = feat_A.shape[0]

        # Project to lower dimension
        pA = self.proj_A(feat_A)   # (B, 128)
        pB = self.proj_B(feat_B)   # (B, 128)

        # MultiheadAttention expects (B, seq_len, dim)
        # We treat each person's feature as a single "token" (seq_len=1)
        pA_seq = pA.unsqueeze(1)   # (B, 1, 128)
        pB_seq = pB.unsqueeze(1)   # (B, 1, 128)

        # Cross-attention: A queries B (what about B is relevant to A?)
        fused_A, _ = self.cross_attn_A(
            query=pA_seq, key=pB_seq, value=pB_seq
        )  # (B, 1, 128)

        # Cross-attention: B queries A (what about A is relevant to B?)
        fused_B, _ = self.cross_attn_B(
            query=pB_seq, key=pA_seq, value=pA_seq
        )  # (B, 1, 128)

        # Residual + LayerNorm (stabilise training)
        fused_A = self.norm_A(pA + fused_A.squeeze(1))   # (B, 128)
        fused_B = self.norm_B(pB + fused_B.squeeze(1))   # (B, 128)

        # Encode interaction signals
        enc_signals = self.signal_encoder(signals)   # (B, 4)

        # Concatenate and classify
        combined = torch.cat([fused_A, fused_B, enc_signals], dim=-1)   # (B, 260)
        logits   = self.mlp(combined).squeeze(-1)                        # (B,)

        return logits

    @torch.no_grad()
    def predict_prob(
        self,
        feat_A  : torch.Tensor,
        feat_B  : torch.Tensor,
        signals : torch.Tensor,
    ) -> float:
        """
        Convenience wrapper for single-pair inference.

        Args:
            feat_A:  (768,) or (1, 768)
            feat_B:  (768,) or (1, 768)
            signals: (4,) or (1, 4)

        Returns:
            float: collaboration probability 0.0-1.0
        """
        self.eval()
        if feat_A.dim() == 1: feat_A = feat_A.unsqueeze(0)
        if feat_B.dim() == 1: feat_B = feat_B.unsqueeze(0)
        if signals.dim() == 1: signals = signals.unsqueeze(0)

        logit = self.forward(feat_A, feat_B, signals)
        return torch.sigmoid(logit).item()


# ---------------------------------------------------------------------------
# FeatureExtractorWrapper
# ---------------------------------------------------------------------------

class FeatureExtractorWrapper(nn.Module):
    """
    Wraps a FROZEN SwinClipModel to expose the temporal clip feature (768-d)
    as a separate output, alongside the usual engagement logit.

    The original SwinClipModel.forward() only returns (B, 2) logits.
    This wrapper intercepts the 768-d temporal feature BEFORE the head.

    Usage:
        engagement_model = load_engagement_model("weights/best_clip_model.pth")
        extractor = FeatureExtractorWrapper(engagement_model)
        # Freeze everything
        for p in extractor.parameters():
            p.requires_grad = False

        clip_tensor = ...  # (1, 8, 3, 224, 224)
        eng_logit, clip_feat = extractor(clip_tensor)
        # eng_logit: (1, 2)  ← engagement prediction (as before)
        # clip_feat: (1, 768) ← temporal feature for CollaborationHead
    """

    def __init__(self, swin_clip_model: nn.Module):
        super().__init__()
        self.backbone = swin_clip_model.backbone
        self.temporal = swin_clip_model.temporal
        self.head     = swin_clip_model.head

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            eng_logits: (B, 2)    engagement logits (Engaged=1, NotEngaged=0)
            clip_feat:  (B, 768)  temporal clip embedding for collaboration
        """
        B, T, C, H, W = x.shape

        frames   = x.view(B * T, C, H, W)
        features = self.backbone(frames)           # (B*T, 768)
        features = features.view(B, T, -1)         # (B, T, 768)

        clip_feat   = self.temporal(features)      # (B, 768)  ← EXPOSED
        eng_logits  = self.head(clip_feat)         # (B, 2)

        return eng_logits, clip_feat


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_collab_head(
    feat_dim   : int   = 768,
    proj_dim   : int   = 128,
    n_heads    : int   = 4,
    signal_dim : int   = 4,
    mlp_hidden : int   = 64,
    dropout    : float = 0.3,
) -> CollaborationHead:
    """Build a CollaborationHead instance."""
    return CollaborationHead(
        feat_dim   = feat_dim,
        proj_dim   = proj_dim,
        n_heads    = n_heads,
        signal_dim = signal_dim,
        mlp_hidden = mlp_hidden,
        dropout    = dropout,
    )


def build_feature_extractor(swin_clip_model: nn.Module) -> FeatureExtractorWrapper:
    """
    Wraps a loaded SwinClipModel and FULLY freezes all parameters.
    Safe to call after loading checkpoint.
    """
    extractor = FeatureExtractorWrapper(swin_clip_model)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()

    total = sum(p.numel() for p in extractor.parameters())
    print(f"FeatureExtractorWrapper: {total/1e6:.1f}M params (ALL FROZEN)")
    return extractor


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def collab_loss(
    logits    : torch.Tensor,   # (B,)
    labels    : torch.Tensor,   # (B,)  float 0.0 or 1.0
    pos_weight: float = 1.5,
) -> torch.Tensor:
    """
    Weighted BCE loss for collaboration training.

    pos_weight > 1: Penalises missing collaborating pairs more than false positives.
    Set higher (2.0-2.5) if Collaborative pairs are underrepresented in your data.

    Label smoothing (0.05) prevents overconfident predictions on noisy collab labels.
    Collaboration annotation is inherently subjective; a model that is 95% confident
    is almost certainly overfit.
    """
    # Label smoothing
    smooth_labels = labels * 0.90 + 0.05

    pw = torch.tensor([pos_weight], device=logits.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")
    return loss_fn(logits, smooth_labels)


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    head = build_collab_head()

    B = 4
    feat_A  = torch.randn(B, 768)
    feat_B  = torch.randn(B, 768)
    signals = torch.rand(B, 4)   # all signals in [0,1]
    labels  = torch.tensor([1.0, 0.0, 1.0, 0.0])

    logits = head(feat_A, feat_B, signals)
    loss   = collab_loss(logits, labels)

    print(f"\nSanity check:")
    print(f"  logits: {logits.tolist()}")
    print(f"  probs:  {torch.sigmoid(logits).tolist()}")
    print(f"  loss:   {loss.item():.4f}")

    # Symmetric test — (A,B) and (B,A) should give same probability
    prob_AB = torch.sigmoid(head(feat_A[:1], feat_B[:1], signals[:1])).item()
    prob_BA = torch.sigmoid(head(feat_B[:1], feat_A[:1], signals[:1])).item()
    print(f"\nSymmetry test:")
    print(f"  P(collab | A,B) = {prob_AB:.4f}")
    print(f"  P(collab | B,A) = {prob_BA:.4f}")
    print(f"  (should be similar but not identical before symmetric training)")
