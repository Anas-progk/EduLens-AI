"""
EngagementDataset -- PyTorch Dataset for binary engagement classification.

v4 addition: build_sampler() classmethod for three-phase balanced training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Standard augmentation -- majority class (Engaged)."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.0), value=0),
    ])


def get_train_transforms_strong(image_size: int = 224) -> T.Compose:
    """
    STRONG augmentation -- minority class (Not Engaged, label=0).

    NE frames are only 5.8% of data (~2,656 frames).
    With balanced sampling (Phase 1), each NE frame appears ~16x per epoch.
    Stronger augmentation prevents the model from memorising those 2,656 frames.
    Variety amplified: crop scale, rotation, color, blur, perspective, erasing.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.15),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))], p=0.4),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.25, p=1.0)], p=0.35),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for VALIDATION and TEST. No augmentation."""
    return T.Compose([
        T.Resize(int(image_size * 1.143)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class EngagementDataset(Dataset):
    """
    Loads engagement frames from a CSV file.
    Labels: 0 = Not Engaged, 1 = Engaged

    Training uses class-conditional augmentation:
      label=0 (NE) -> strong pipeline (more variety per scarce frame)
      label=1 (E)  -> standard pipeline
    """

    def __init__(self, csv_file: str, split: str = 'train', image_size: int = 224):
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['image_path', 'label'])
        self.paths      = df['image_path'].values
        self.labels     = df['label'].astype(int).values
        self.split      = split
        self.image_size = image_size

        if split == 'train':
            self.transform_majority = get_train_transforms(image_size)
            self.transform_minority = get_train_transforms_strong(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        ne_count  = int((self.labels == 0).sum())
        eng_count = int((self.labels == 1).sum())
        aug_note  = "  (NE->strong aug)" if split == 'train' else ""
        print(f"  [{split.upper():5s}] {len(self):>8,} frames  |  "
              f"Engaged={eng_count:>6,}  Not Engaged={ne_count:>6,}{aug_note}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path  = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.image_size, self.image_size), color=0)

        if self.split == 'train':
            transform = (self.transform_minority if label == 0
                         else self.transform_majority)
        else:
            transform = self.transform
        return transform(img), torch.tensor(label, dtype=torch.long)

    # -------------------------------------------------------------------------
    # SAMPLER FACTORY  (v4)
    # -------------------------------------------------------------------------

    @classmethod
    def build_sampler(cls, labels_array, ne_proportion: float = 0.5):
        """
        Build a WeightedRandomSampler that achieves ne_proportion NE per batch.

        ne_proportion=0.50  -> 50% NE per batch (16 NE in batch_size=32)
        ne_proportion=0.25  -> 25% NE per batch (8 NE in batch_size=32)
        vs. natural: 0.058  -> 5.8% NE per batch (~2 NE in batch_size=32)

        CRITICAL -- No Double Correction:
          This sampler changes the SAMPLING distribution.
          If you also pass strong alpha weights to the loss (e.g. power=0.75),
          you amplify both sampling AND gradient -- exactly the double-correction
          that caused the v1 crash (NaN loss, val F1 = 0.04).
          When using this sampler: set focal_alpha_power=0.0 (uniform alpha).

        Math:
          sample_weight[i] = desired_fraction[class_i] / count[class_i]
          This makes E[class_fraction in batch] = desired_fraction.
        """
        labels_array = np.asarray(labels_array)
        counts  = np.bincount(labels_array, minlength=2).astype(float)
        desired = np.array([ne_proportion, 1.0 - ne_proportion])
        class_weights  = desired / counts
        sample_weights = class_weights[labels_array]
        sampler = WeightedRandomSampler(
            weights     = torch.FloatTensor(sample_weights),
            num_samples = len(sample_weights),
            replacement = True,
        )
        n_ne = int(ne_proportion * 32)
        n_e  = 32 - n_ne
        print(f"  Sampler: {ne_proportion*100:.0f}% NE per batch  "
              f"(~{n_ne} NE + ~{n_e} E per batch of 32, "
              f"vs ~2 NE natural)")
        return sampler

    def get_class_weights(self) -> torch.Tensor:
        counts  = np.bincount(self.labels, minlength=2).astype(float)
        weights = len(self.labels) / (2.0 * counts)
        return torch.FloatTensor(weights)
