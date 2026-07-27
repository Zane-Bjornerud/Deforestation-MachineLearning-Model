#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp

from src.dataset import DeforestationDataset
from src.band_names import CANONICAL_BAND_ORDER


DATA_DIR = "data/processed"
METADATA_FILE = f"{DATA_DIR}/val_metadata.pkl"
NORM_STATS_FILE = f"{DATA_DIR}/normalization_stats.pkl"
CHECKPOINT_FILE = "outputs/checkpoints/best_model.pth"
OUTPUT_ROOT = Path("artifacts/inspection")
SAMPLE_INDEX = 0

# The canonical 18 bands, in the order a fresh export (scripts/gee_export_chips.py)
# produces them. Chips processed from the legacy TFRecords have the same 18
# canonical names (see src/band_names.py) but not necessarily this same order,
# since their positions are frozen to match the already-trained checkpoint.
EXPECTED_CHANNELS = CANONICAL_BAND_ORDER


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stats_for_array(array: np.ndarray):
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def write_stats_csv(path: Path, band_names, arrays):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "channel", "min", "max", "mean", "std"])
        for index, (band_name, array) in enumerate(zip(band_names, arrays)):
            s = stats_for_array(array)
            writer.writerow(
                [index, band_name, s["min"], s["max"], s["mean"], s["std"]]
            )


def pick_rgb_indices(band_names):
    preferred = ["B4_pre", "B3_pre", "B2_pre"]
    indices = [band_names.index(name) for name in preferred if name in band_names]
    if len(indices) != 3:
        raise ValueError(
            f"Expected canonical bands {preferred} in band_names, got {band_names}"
        )
    return indices


def make_rgb(chip_tensor, band_names):
    indices = pick_rgb_indices(band_names)
    rgb = chip_tensor[indices].permute(1, 2, 0).cpu().numpy()
    rgb = rgb.astype(np.float32)
    rgb = rgb - np.nanmin(rgb)
    denom = np.nanmax(rgb) - np.nanmin(rgb) + 1e-8
    rgb = np.clip(rgb / denom, 0, 1)
    return rgb, indices


def save_preview(path: Path, image, title=None, cmap=None, vmin=None, vmax=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    safe_mkdir(OUTPUT_ROOT)
    sample_dir = OUTPUT_ROOT / f"sample_{SAMPLE_INDEX:03d}"
    safe_mkdir(sample_dir)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DeforestationDataset(
        DATA_DIR,
        METADATA_FILE,
        NORM_STATS_FILE,
        augment=False,
    )

    if SAMPLE_INDEX >= len(dataset):
        raise IndexError(f"sample index {SAMPLE_INDEX} out of range for dataset size {len(dataset)}")

    item = dataset.metadata[SAMPLE_INDEX]
    band_names = item["band_names"]

    print("Source metadata:")
    print(json.dumps(item, indent=2, default=str))

    print("\nChannel names:")
    for index, name in enumerate(band_names):
        print(f"{index:02d}: {name}")

    raw_chip = np.load(f"{DATA_DIR}/{item['chip_path']}")
    raw_mask = np.load(f"{DATA_DIR}/{item['mask_path']}")

    print(f"\nRaw chip shape: {raw_chip.shape}")
    print(f"Raw mask shape: {raw_mask.shape}")

    raw_arrays = [raw_chip[i] for i in range(raw_chip.shape[0])]
    write_stats_csv(sample_dir / "input_statistics.csv", band_names, raw_arrays)

    processed_chip, target_mask = dataset[SAMPLE_INDEX]
    processed_np = processed_chip.cpu().numpy()

    print(f"\nProcessed tensor shape: {processed_chip.shape}")
    print(f"Target mask shape: {target_mask.shape}")

    processed_arrays = [processed_np[i] for i in range(processed_np.shape[0])]
    write_stats_csv(sample_dir / "processed_statistics.csv", band_names, processed_arrays)

    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "sample_index": SAMPLE_INDEX,
                "chip_id": item.get("chip_id"),
                "chip_path": item.get("chip_path"),
                "mask_path": item.get("mask_path"),
                "source_tfrecord": item.get("source_tfrecord"),
                "source_record_index": item.get("source_record_index"),
                "band_names": band_names,
                "expected_channels": EXPECTED_CHANNELS,
                "raw_shape": list(raw_chip.shape),
                "processed_shape": list(processed_chip.shape),
                "mask_shape": list(target_mask.shape),
                "device": str(device),
                "checkpoint": CHECKPOINT_FILE,
            },
            f,
            indent=2,
            default=str,
        )

    rgb, rgb_indices = make_rgb(processed_chip, band_names)
    print(f"\nRGB preview indices: {rgb_indices}")

    save_preview(
        sample_dir / "input_preview.png",
        rgb,
        title="Input preview",
    )

    save_preview(
        sample_dir / "target_mask.png",
        target_mask[0].cpu().numpy(),
        title="Target mask",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    model = smp.Unet(encoder_name="resnet34", in_channels=processed_chip.shape[0], classes=1)
    state_dict = torch.load(CHECKPOINT_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        input_batch = processed_chip.unsqueeze(0).to(device)
        print(f"\nModel input batch shape: {input_batch.shape}")

        logits = model(input_batch)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_binary = (probs > 0.5).astype(np.float32)

    print(f"Model output shape: {logits.shape}")
    print(f"Prediction probability shape: {probs.shape}")
    print(f"Prediction binary shape: {pred_binary.shape}")

    save_preview(
        sample_dir / "prediction_probability.png",
        probs,
        title="Prediction probability",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    save_preview(
        sample_dir / "prediction_binary.png",
        pred_binary,
        title="Prediction binary",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    print(f"\nSaved inspection package to: {sample_dir}")


if __name__ == "__main__":
    main()