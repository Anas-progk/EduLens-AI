"""
ClipDataset -- PyTorch Dataset for clip-level (temporal) engagement classification.

Each sample is a sequence of T consecutive frames from a single video clip.
The model sees temporal context, not just a single snapshot.

Key design decisions:
  - Frames grouped by clip_id
  - Clip label = majority label of constituent frames
  - Class-conditional augmentation: NE clips get stronger augmentation on ALL frames
  - Consistent random state per clip: same crop/flip applied to all T frames (temporal coherence)
  - build_sampler() for balanced NE clip exposure during training

v2 changes (clip_dataset):
  - ClipAugmentor strong mode parameters reduced to prevent "distortion = NE" learning.
    Run 5 showed that combining aggressive ClipAugmentor + aggressive synthetic data
    dropped NE_F1 from 0.778 -> 0.628. The strong mode is still clearly stronger than
    the engaged mode, but parameters are realistic for a seated classroom setting.
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Validation transforms (no augmentation)
# ---------------------------------------------------------------------------

def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation/test -- no augmentation."""
    return T.Compose([
        T.Resize(int(image_size * 1.143)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Temporally-coherent augmentation helpers
# ---------------------------------------------------------------------------

class ClipAugmentor:
    """
    Applies the SAME random spatial transform to every frame in a clip,
    then independent photometric noise per frame.

    Why coherence matters:
      If we crop frame 1 to the face and frame 2 to the desk,
      the temporal model cannot learn meaningful dynamics.
      Spatial transforms must be consistent; photometric can vary.
    """

    def __init__(self, image_size: int = 224, strong: bool = False):
        self.image_size = image_size
        self.strong = strong
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if strong:
            # NE clips -- stronger augmentation than Engaged, but NOT so aggressive
            # that the model learns "distortion = NE" instead of "behaviour = NE".
            #
            # v2 reduction rationale (from run 4 vs run 5 analysis):
            #   Run 5 (aggressive synthetic + strong aug) gave NE_F1=0.628 vs
            #   run 4 (no synthetic, original strong aug) giving NE_F1=0.778.
            #   The combination of ClipAugmentor strong mode + extra synthetic data
            #   created a feedback loop: too many distorted NE clips -> model learns
            #   photometric artifacts as the NE signal.
            #   These reduced params keep augmentation meaningful but realistic.
            self.crop_scale   = (0.72, 1.0)   # was (0.55, 1.0) -- face must stay in frame
            self.crop_ratio   = (0.85, 1.15)  # was (0.75, 1.33)
            self.rot_degrees  = 12            # was 20 -- classroom cameras are stable
            self.cj_bright    = 0.35          # was 0.5
            self.cj_contrast  = 0.35          # was 0.5
            self.cj_sat       = 0.25          # was 0.4
            self.cj_hue       = 0.08          # was 0.1
            self.gray_p       = 0.10          # was 0.15
            self.blur_p       = 0.25          # was 0.4
            self.blur_sigma   = (0.1, 1.5)    # was (0.1, 2.5)
            self.persp_p      = 0.15          # was 0.35 -- rare viewpoint change
            self.persp_scale  = 0.12          # was 0.25
            self.erase_p      = 0.20          # was 0.35
            self.erase_scale  = (0.02, 0.12)  # was (0.02, 0.20)
        else:
            # Engaged clips -- standard augmentation
            self.crop_scale   = (0.70, 1.0)
            self.crop_ratio   = (0.85, 1.15)
            self.rot_degrees  = 10
            self.cj_bright    = 0.3
            self.cj_contrast  = 0.3
            self.cj_sat       = 0.2
            self.cj_hue       = 0.05
            self.gray_p       = 0.05
            self.blur_p       = 0.2
            self.blur_sigma   = (0.1, 1.5)
            self.persp_p      = 0.0
            self.persp_scale  = 0.0
            self.erase_p      = 0.15
            self.erase_scale  = (0.02, 0.12)

        # Per-frame independent erasing (applied after ToTensor)
        self.eraser = T.RandomErasing(
            p=self.erase_p,
            scale=self.erase_scale,
            ratio=(0.3, 3.3),
            value=0,
        )

    def _sample_spatial_params(self, img: Image.Image):
        """Sample spatial augmentation parameters once per clip."""
        w, h = img.size

        crop_params = T.RandomResizedCrop.get_params(
            img, scale=self.crop_scale, ratio=self.crop_ratio,
        )
        do_hflip = random.random() < 0.5
        angle    = random.uniform(-self.rot_degrees, self.rot_degrees)

        do_persp    = self.persp_p > 0 and random.random() < self.persp_p
        persp_params = None
        if do_persp:
            persp_params = T.RandomPerspective.get_params(w, h, self.persp_scale)

        return crop_params, do_hflip, angle, do_persp, persp_params

    def _apply_spatial(self, img: Image.Image, params) -> Image.Image:
        """Apply pre-sampled spatial params to one frame."""
        crop_params, do_hflip, angle, do_persp, persp_params = params

        i, j, th, tw = crop_params
        img = TF.resized_crop(img, i, j, th, tw, (self.image_size, self.image_size))
        if do_hflip:
            img = TF.hflip(img)
        img = TF.rotate(img, angle)
        if do_persp and persp_params is not None:
            img = TF.perspective(img, *persp_params)
        return img

    def _apply_photometric(self, img: Image.Image) -> Image.Image:
        """Per-frame independent photometric augmentation."""
        cj = T.ColorJitter(
            brightness=self.cj_bright,
            contrast=self.cj_contrast,
            saturation=self.cj_sat,
            hue=self.cj_hue,
        )
        img = cj(img)
        if random.random() < self.gray_p:
            img = TF.to_grayscale(img, num_output_channels=3)
        if random.random() < self.blur_p:
            sigma = random.uniform(*self.blur_sigma)
            img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img

    def __call__(self, frames: list) -> torch.Tensor:
        """
        Args:
            frames: list of PIL Images (length T)
        Returns:
            tensor: (T, C, H, W) float32, normalised
        """
        spatial_params = self._sample_spatial_params(frames[0])
        tensors = []
        for img in frames:
            img = self._apply_spatial(img, spatial_params)
            img = self._apply_photometric(img)
            t   = TF.to_tensor(img)
            t   = self.normalize(t)
            t   = self.eraser(t)
            tensors.append(t)
        return torch.stack(tensors, dim=0)   # (T, C, H, W)


# ---------------------------------------------------------------------------
# ClipDataset
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    """
    Groups frame-level CSV rows into clips.

    Expected CSV columns:
        image_path  -- absolute path to frame file
        label       -- 0 = Not Engaged, 1 = Engaged
        clip_id     -- string identifying the source video clip
                       (if absent, uses first 6 chars of image basename)

    Each __getitem__ returns:
        frames: torch.Tensor  (T, C, H, W)  float32
        label:  torch.Tensor  scalar int64

    Training: class-conditional augmentation via ClipAugmentor
    Validation/Test: deterministic get_val_transforms per frame
    """

    def __init__(
        self,
        csv_file   : str,
        split      : str = "train",
        image_size : int = 224,
        n_frames   : int = 8,
        min_frames : int = 4,
    ):
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["image_path", "label"])
        df["label"] = df["label"].astype(int)

        if "clip_id" not in df.columns:
            import os
            df["clip_id"] = df["image_path"].apply(
                lambda p: os.path.basename(p)[:6]
            )

        grouped = (
            df.groupby("clip_id")
            .agg(image_paths=("image_path", list), labels=("label", list))
            .reset_index()
        )
        grouped["label"] = grouped["labels"].apply(
            lambda ls: int(np.bincount(ls).argmax())
        )
        grouped = grouped[
            grouped["image_paths"].apply(len) >= min_frames
        ].reset_index(drop=True)

        self.clips      = grouped["image_paths"].tolist()
        self.labels     = grouped["label"].values
        self.clip_ids   = grouped["clip_id"].tolist()
        self.split      = split
        self.image_size = image_size
        self.n_frames   = n_frames

        if split == "train":
            self.aug_majority = ClipAugmentor(image_size, strong=False)
            self.aug_minority = ClipAugmentor(image_size, strong=True)
        else:
            self.val_transform = get_val_transforms(image_size)

        ne_count  = int((self.labels == 0).sum())
        eng_count = int((self.labels == 1).sum())
        aug_note  = "  (NE->strong clip aug)" if split == "train" else ""
        print(
            f"  [{split.upper():5s}] {len(self):>6,} clips  |  "
            f"Engaged={eng_count:>5,}  Not Engaged={ne_count:>5,}{aug_note}"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def _sample_frames(self, paths: list) -> list:
        """Uniformly subsample or pad a clip to exactly self.n_frames frames."""
        T = self.n_frames
        if len(paths) >= T:
            indices = np.linspace(0, len(paths) - 1, T, dtype=int)
        else:
            indices = list(range(len(paths)))
            while len(indices) < T:
                indices.append(indices[-1])
            indices = np.array(indices)

        frames = []
        for idx in indices:
            try:
                p   = str(paths[idx]).replace('\\', '/')
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.image_size, self.image_size), color=0)
            frames.append(img)
        return frames

    def __getitem__(self, idx: int):
        paths  = self.clips[idx]
        label  = self.labels[idx]
        frames = self._sample_frames(paths)

        if self.split == "train":
            augmentor   = self.aug_minority if label == 0 else self.aug_majority
            clip_tensor = augmentor(frames)
        else:
            tensors     = [self.val_transform(f) for f in frames]
            clip_tensor = torch.stack(tensors, dim=0)

        return clip_tensor, torch.tensor(label, dtype=torch.long)

    @classmethod
    def build_sampler(cls, labels_array, ne_proportion: float = 0.5):
        """
        WeightedRandomSampler for clip-level balanced training.

        ne_proportion=0.50 -> 50% NE clips per batch
        ne_proportion=0.25 -> 25% NE clips per batch

        CRITICAL: Do NOT combine with strong focal alpha weights.
        Use sampler + UNIFORM focal alpha (focal_alpha_pw=0.0).
        """
        labels_array = np.asarray(labels_array)
        counts  = np.bincount(labels_array, minlength=2).astype(float)
        desired = np.array([ne_proportion, 1.0 - ne_proportion])
        cw      = desired / counts
        sw      = cw[labels_array]
        sampler = WeightedRandomSampler(
            weights     = torch.FloatTensor(sw),
            num_samples = len(sw),
            replacement = True,
        )
        natural_ne_pct = counts[0] / counts.sum() * 100
        print(
            f"  Clip Sampler: {ne_proportion*100:.0f}% NE  "
            f"(natural={natural_ne_pct:.1f}%  |  "
            f"~{int(ne_proportion*4)} NE + ~{4-int(ne_proportion*4)} E per batch of 4 clips)"
        )
        return sampler

    def get_class_weights(self) -> torch.Tensor:
        counts  = np.bincount(self.labels, minlength=2).astype(float)
        weights = len(self.labels) / (2.0 * counts)
        return torch.FloatTensor(weights)
