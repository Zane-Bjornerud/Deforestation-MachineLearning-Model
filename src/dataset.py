import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow `from band_names import ...` whether this module is imported as
# `src.dataset` (e.g. from scripts/, with the repo root on sys.path) or as
# a bare sibling module (e.g. `python src/train.py`, with src/ on sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_names import CANONICAL_BAND_SET


def _validate_band_names(metadata):
    """Raise early if any chip's band_names aren't the canonical 18 bands."""
    for item in metadata:
        band_set = set(item["band_names"])
        if band_set != CANONICAL_BAND_SET:
            raise ValueError(
                f"chip {item.get('chip_id')} has non-canonical band_names "
                f"{item['band_names']}; expected the 18 canonical bands "
                f"(mismatch: {sorted(band_set ^ CANONICAL_BAND_SET)}). "
                "Re-run the TFRecord processor to canonicalize band names."
            )


class DeforestationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        metadata_file,
        normalization_stats_file,
        augment=False,
        filter_positives=False,
    ):
        """
        Args:
            data_dir: Directory containing chips and masks
            metadata_file: Pickle file with chip metadata
            normalization_stats_file: Pickle file with normalization stats
            augment: Whether to apply data augmentation
            filter_positives: If True, only include chips with deforestation
        """
        self.data_dir = data_dir

        # Load metadata
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)

        _validate_band_names(self.metadata)

        # Filter for positive samples if requested
        if filter_positives:
            self.metadata = [x for x in self.metadata if x["has_deforestation"]]

        # Load normalization stats
        with open(normalization_stats_file, "rb") as f:
            stats = pickle.load(f)
            self.means = stats["means"]
            self.stds = stats["stds"]

        # Set up augmentations
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                ],
                additional_targets={"mask": "mask"},
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.metadata)

    def normalize(self, chip):
        """Normalize chip using pre-computed statistics."""
        # chip shape: (n_bands, H, W)
        normalized = np.zeros_like(chip)
        n_bands = min(chip.shape[0], len(self.means))  # Handle potential band mismatch
        for i in range(n_bands):
            normalized[i] = (chip[i] - self.means[i]) / (self.stds[i] + 1e-8)
        # If chip has more bands than stats, leave them as-is
        if chip.shape[0] > len(self.means):
            normalized[len(self.means) :] = chip[len(self.means) :]
        return normalized

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load chip and mask
        chip = np.load(f"{self.data_dir}/{item['chip_path']}")  # (n_bands, H, W)
        mask = np.load(f"{self.data_dir}/{item['mask_path']}")  # (1, H, W)

        # Normalize chip
        chip = self.normalize(chip)

        # Convert to HWC format for albumentations
        chip_hwc = chip.transpose(1, 2, 0)  # (H, W, n_bands)
        mask_hw = mask[0]  # (H, W)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=chip_hwc, mask=mask_hw)
            chip_hwc = transformed["image"]
            mask_hw = transformed["mask"]

        # Convert back to CHW format
        chip = chip_hwc.transpose(2, 0, 1)  # (n_bands, H, W)
        mask = mask_hw[np.newaxis, :, :]  # (1, H, W)

        # Convert to tensors
        chip_tensor = torch.from_numpy(chip.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return chip_tensor, mask_tensor


class BalancedDeforestationDataset(Dataset):
    """Dataset that balances positive and negative samples."""

    def __init__(
        self,
        data_dir,
        metadata_file,
        normalization_stats_file,
        augment=False,
        pos_fraction=0.5,
    ):
        """
        Args:
            pos_fraction: Fraction of samples that should be positive (have deforestation)
        """
        self.data_dir = data_dir
        self.pos_fraction = pos_fraction

        # Load metadata
        with open(metadata_file, "rb") as f:
            all_metadata = pickle.load(f)

        _validate_band_names(all_metadata)

        # Split into positive and negative samples
        self.pos_samples = [x for x in all_metadata if x["has_deforestation"]]
        self.neg_samples = [x for x in all_metadata if not x["has_deforestation"]]

        print(
            f"Found {len(self.pos_samples)} positive and {len(self.neg_samples)} negative samples"
        )

        # Load normalization stats
        with open(normalization_stats_file, "rb") as f:
            stats = pickle.load(f)
            self.means = stats["means"]
            self.stds = stats["stds"]

        # Set up augmentations
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                ],
                additional_targets={"mask": "mask"},
            )
        else:
            self.transform = None

    def __len__(self):
        # Return length based on positive samples (since they're usually fewer)
        return len(self.pos_samples) * 2  # 2x to account for negative samples

    def normalize(self, chip):
        """Normalize chip using pre-computed statistics."""
        normalized = np.zeros_like(chip)
        for i in range(chip.shape[0]):
            normalized[i] = (chip[i] - self.means[i]) / (self.stds[i] + 1e-8)
        return normalized

    def __getitem__(self, idx):
        # Decide whether to return positive or negative sample
        if np.random.random() < self.pos_fraction:
            # Return positive sample
            item = np.random.choice(self.pos_samples)
        else:
            # Return negative sample
            item = np.random.choice(self.neg_samples)

        # Load chip and mask
        chip = np.load(f"{self.data_dir}/{item['chip_path']}")
        mask = np.load(f"{self.data_dir}/{item['mask_path']}")

        # Normalize chip
        chip = self.normalize(chip)

        # Convert to HWC format for albumentations
        chip_hwc = chip.transpose(1, 2, 0)
        mask_hw = mask[0]

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=chip_hwc, mask=mask_hw)
            chip_hwc = transformed["image"]
            mask_hw = transformed["mask"]

        # Convert back to CHW format
        chip = chip_hwc.transpose(2, 0, 1)
        mask = mask_hw[np.newaxis, :, :]

        # Convert to tensors
        chip_tensor = torch.from_numpy(chip.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return chip_tensor, mask_tensor
