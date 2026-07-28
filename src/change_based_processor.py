import numpy as np
import os
import sys
import pickle
from pathlib import Path
import tensorflow as tf
from scipy import ndimage
import math
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_names import CANONICAL_BAND_ORDER, legacy_to_canonical, raw_bands_to_canonical_order
from dataset_contract import (
    assert_no_label_source_conflict,
    load_contract,
    validate_processor_identity,
    write_dataset_manifest,
)

PROCESSOR_NAME = "change_based_processor"


def inspect_processed_sample(
    metadata_file,
    sample_index=0,
    output_dir="debug/channels",
):
    """Print per-channel stats and save an 18-panel diagnostic figure."""

    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(metadata_file)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    item = metadata[sample_index]
    image = np.load(os.path.join(base_dir, item["chip_path"]))
    mask = np.load(os.path.join(base_dir, item["mask_path"]))
    band_names = item["band_names"]

    print(f"Chip id: {item['chip_id']}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print("Band order:")
    for channel_index, band_name in enumerate(band_names):
        print(f"  {channel_index:02d}: {band_name}")

    print("\nChannel statistics:")
    for channel_index in range(image.shape[0]):
        channel = image[channel_index]
        band_name = band_names[channel_index] if channel_index < len(band_names) else f"channel_{channel_index}"

        print(
            f"{channel_index:02d} {band_name:10s} "
            f"min={float(channel.min()):10.4f} "
            f"max={float(channel.max()):10.4f} "
            f"mean={float(channel.mean()):10.4f} "
            f"std={float(channel.std()):10.4f}"
        )

    print("\nMask statistics:")
    print(
        f"mask min={float(mask.min()):.4f} "
        f"max={float(mask.max()):.4f} "
        f"mean={float(mask.mean()):.4f} "
        f"std={float(mask.std()):.4f} "
        f"sum={float(mask.sum()):.1f}"
    )

    n_channels = image.shape[0]
    n_cols = 3
    n_rows = math.ceil(n_channels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for channel_index in range(n_channels):
        channel = image[channel_index]
        band_name = band_names[channel_index] if channel_index < len(band_names) else f"channel_{channel_index}"

        ax = axes[channel_index]
        im = ax.imshow(channel, cmap="gray")
        ax.set_title(f"{channel_index:02d}: {band_name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

        plt.imsave(
            f"{output_dir}/channel_{channel_index:02d}_{band_name}.png",
            channel,
            cmap="gray",
        )

    for ax in axes[n_channels:]:
        ax.axis("off")

    plt.tight_layout()
    figure_path = f"{output_dir}/all_channels_{item['chip_id']}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved channel panels to: {output_dir}")
    print(f"Saved combined figure to: {figure_path}")

def resize_array(array, target_size):
    """Resize array to target size using bilinear interpolation."""
    if array.shape == target_size:
        return array

    # Calculate zoom factors
    zoom_factors = (target_size[0] / array.shape[0], target_size[1] / array.shape[1])

    # Resize using scipy
    resized = ndimage.zoom(array, zoom_factors, order=1)  # order=1 for bilinear

    return resized


def create_deforestation_labels_from_change(
    chip_data, band_names, dnbr_threshold=-0.15, dndvi_threshold=-0.2
):
    """
    Create deforestation labels based on change indices instead of GFC data.

    Args:
        chip_data: Array of shape (n_bands, H, W)
        band_names: List of band names
        dnbr_threshold: dNBR threshold for deforestation (more negative = more disturbance)
        dndvi_threshold: dNDVI threshold for deforestation (more negative = more vegetation loss)
    """

    try:
        # Find change index bands
        dnbr_idx = band_names.index("dNBR")
        dndvi_idx = band_names.index("dNDVI")

        dnbr = chip_data[dnbr_idx]
        dndvi = chip_data[dndvi_idx]

        # Create deforestation mask based on change thresholds
        # Both conditions must be met for strong deforestation signal
        defor_mask = (dnbr < dnbr_threshold) & (dndvi < dndvi_threshold)

        # Alternative: Use either condition (more sensitive)
        # defor_mask = (dnbr < dnbr_threshold) | (dndvi < dndvi_threshold)

        return defor_mask.astype(np.float32)

    except ValueError as e:
        print(f"Could not find change indices: {e}")
        return np.zeros((chip_data.shape[1], chip_data.shape[2]), dtype=np.float32)


def process_tfrecords_with_change_labels(dataset_id):
    """Process TFRecords using change-based deforestation labels.

    Args:
        dataset_id: Dataset contract id, e.g. "legacy_threshold_v1". Looked
            up under configs/datasets/<dataset_id>.yaml, which supplies the
            input glob (raw_path/*.tfrecord), output_dir (processed_path),
            and the dNBR/dNDVI thresholds recorded alongside every chip.
    """

    print("=== Change-Based Deforestation Processor ===\n")

    contract = load_contract(dataset_id)
    validate_processor_identity(contract, PROCESSOR_NAME)

    input_glob = f"{contract.raw_path}/*.tfrecord"
    output_dir = contract.processed_path

    # Find TFRecord files matching the explicit input glob only
    tfrecord_files = sorted(Path().glob(input_glob))
    print(f"Found {len(tfrecord_files)} files:")
    for f in tfrecord_files:
        print(f"  {f.name}")

    # Create output directories
    os.makedirs(f"{output_dir}/chips", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    all_metadata = []
    chip_count = 0
    target_size = (256, 256)

    dnbr_thresh = contract.extra["dnbr_threshold"]
    dndvi_thresh = contract.extra["dndvi_threshold"]
    threshold_profile = contract.extra.get("threshold_profile", "custom")

    print(f"Using {threshold_profile} thresholds (from contract {dataset_id}):")
    print(f"  dNBR < {dnbr_thresh}")
    print(f"  dNDVI < {dndvi_thresh}")

    assert_no_label_source_conflict(output_dir, contract.label_source)

    # Process each TFRecord file
    for tfrecord_file in tfrecord_files:
        print(f"\nProcessing {tfrecord_file.name}...")

        dataset = tf.data.TFRecordDataset(str(tfrecord_file))

        file_count = 0
        for raw_record_index, raw_record in enumerate(dataset):
            try:
                # Parse the TF Example
                example = tf.train.Example.FromString(raw_record.numpy())
                features = example.features.feature

                if chip_count == 0 and file_count == 0:
                    for key in features.keys():
                        if key != "label":
                            print(key)

                # Extract feature bands into a dict keyed by raw band name.
                # Order is NOT decided here -- dict/protobuf map iteration
                # order is not a channel-order guarantee. Channel order is
                # fixed below via raw_bands_to_canonical_order.
                raw_bands = {}

                # Process all features
                for key, feature in features.items():
                    if key == "label":  # Skip the original label
                        continue

                    if feature.HasField("float_list"):
                        values = feature.float_list.value
                        band_array = np.array(values, dtype=np.float32)

                        # Determine band size and reshape
                        band_size = int(np.sqrt(len(band_array)))
                        band_reshaped = band_array.reshape(band_size, band_size)

                        # Resize to target size if needed
                        if (band_size, band_size) != target_size:
                            band_resized = resize_array(band_reshaped, target_size)
                            if chip_count == 0 and len(raw_bands) == 0:
                                print(
                                    f"Resizing features from {band_size}x{band_size} to {target_size[0]}x{target_size[1]}"
                                )
                        else:
                            band_resized = band_reshaped

                        raw_bands[key] = band_resized

                    elif feature.HasField("bytes_list"):
                        # Handle bytes features (shouldn't be many besides
                        # original label, which is skipped above). GEE
                        # byte-encodes with .toByte() -> UINT8, not float32 --
                        # a float32 read silently corrupts the data (see
                        # GFC_process_tfrecords4.py's label parsing for the
                        # concrete failure mode this caused there).
                        data = feature.bytes_list.value[0]
                        band_array = np.frombuffer(data, dtype=np.uint8).astype(
                            np.float32
                        )

                        band_size = int(np.sqrt(len(band_array)))
                        band_reshaped = band_array.reshape(band_size, band_size)

                        if (band_size, band_size) != target_size:
                            band_resized = resize_array(band_reshaped, target_size)
                        else:
                            band_resized = band_reshaped

                        raw_bands[key] = band_resized

                # Check if we have valid data
                if len(raw_bands) > 0:
                    # Fix channel order explicitly from src/band_names.py --
                    # never from feature/dict iteration order. Handles both
                    # legacy bare/_1 raw names and already-canonical names.
                    chip_bands, band_names = raw_bands_to_canonical_order(raw_bands)

                    for i, name in enumerate(band_names):
                        print(f"{i}: {name}")

                    # Stack bands: (n_bands, height, width)
                    chip = np.stack(chip_bands, axis=0)
                    assert chip.shape == (len(CANONICAL_BAND_ORDER), *target_size), (
                        f"chip {chip_count}: expected shape "
                        f"{(len(CANONICAL_BAND_ORDER), *target_size)}, got {chip.shape}"
                    )
                    assert band_names == CANONICAL_BAND_ORDER, (
                        f"chip {chip_count}: band_names {band_names} != "
                        f"CANONICAL_BAND_ORDER {CANONICAL_BAND_ORDER}"
                    )

                    # Create deforestation mask from change indices
                    defor_mask = create_deforestation_labels_from_change(
                        chip, band_names, dnbr_thresh, dndvi_thresh
                    )

                    # Convert to (1, H, W) format
                    mask = defor_mask[np.newaxis, :, :]

                    # Save chip and mask
                    chip_id = f"chip_{chip_count:05d}"
                    chip_path = f"{output_dir}/chips/{chip_id}.npy"
                    mask_path = f"{output_dir}/masks/{chip_id}.npy"

                    np.save(chip_path, chip.astype(np.float32))
                    np.save(mask_path, mask.astype(np.float32))

                    # Create metadata
                    has_defor = np.sum(mask) > 0
                    defor_fraction = float(np.mean(mask))

                    all_metadata.append(
    {
        "chip_id": chip_id,
        "chip_path": f"chips/{chip_id}.npy",
        "mask_path": f"masks/{chip_id}.npy",
        "source_tfrecord": str(tfrecord_file),
        "source_record_index": raw_record_index,
        "has_deforestation": has_defor,
        "deforestation_fraction": defor_fraction,
        "patch_size": target_size[0],
        "n_bands": len(chip_bands),
        "band_names": band_names,
        "dataset_id": contract.dataset_id,
        "label_mode": contract.label_mode,
        "label_source": contract.label_source,
        "label_contract_version": contract.label_contract_version,
        "threshold_profile": threshold_profile,
        "thresholds_used": {
            "dnbr": dnbr_thresh,
            "dndvi": dndvi_thresh,
        },
    }
)

                    chip_count += 1
                    file_count += 1

                    # Progress update
                    if chip_count % 100 == 0:
                        current_defor = sum(
                            1 for x in all_metadata if x["has_deforestation"]
                        )
                        print(
                            f"Processed {chip_count} chips, {current_defor} with deforestation ({current_defor/chip_count*100:.1f}%)"
                        )

                    # Show first few chips info
                    if chip_count <= 5:
                        print(
                            f"Chip {chip_count}: {len(chip_bands)} bands, {target_size[0]}x{target_size[1]}"
                        )
                        print(f"Has deforestation: {has_defor} ({defor_fraction:.3%})")
                        if chip_count == 1:
                            print(f"      Bands: {band_names}")

            except Exception as e:
                if file_count < 3:  # Only show errors for first few
                    print(f"    Error: {e}")
                continue

        print(f"Extracted {file_count} chips from {tfrecord_file.name}")

    # Save metadata
    metadata_path = f"{output_dir}/metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    # Print summary
    print("PROCESSING SUMMARY")
    print(f"Total chips processed: {chip_count}")

    if all_metadata:
        n_with_defor = sum(1 for x in all_metadata if x["has_deforestation"])
        n_without_defor = len(all_metadata) - n_with_defor

        print(
            f"Chips with deforestation: {n_with_defor} ({n_with_defor/len(all_metadata)*100:.1f}%)"
        )
        print(
            f"Chips without deforestation: {n_without_defor} ({n_without_defor/len(all_metadata)*100:.1f}%)"
        )

        # Show threshold effectiveness
        if n_with_defor > 0:
            defor_chips = [x for x in all_metadata if x["has_deforestation"]]
            defor_fractions = [x["deforestation_fraction"] for x in defor_chips]
            print(f"\nDeforestation coverage in positive chips:")
            print(f"  Min: {min(defor_fractions):.3%}")
            print(f"  Max: {max(defor_fractions):.3%}")
            print(f"  Mean: {np.mean(defor_fractions):.3%}")

        print(f"\nLabeling method: Change indices with {threshold_profile} thresholds")
        print(f"Thresholds used: dNBR < {dnbr_thresh}, dNDVI < {dndvi_thresh}")

    print(f"\nFiles saved:")
    print(f"  Metadata: {metadata_path}")
    print(f"  Chips: {output_dir}/chips/ ({chip_count} files)")
    print(f"  Masks: {output_dir}/masks/ ({chip_count} files)")

    if chip_count > 0 and n_with_defor > 0:
        print("\nProcessing successful with deforestation detected!")
        return all_metadata
    else:
        print(
            "\nProcessing complete but no deforestation detected. Try adjusting thresholds."
        )
        return None


def compute_normalization_stats_final(metadata_file, n_samples=100):
    """Compute normalization statistics."""

    print(f"\nComputing normalization statistics...")

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    if not metadata:
        print("No metadata found!")
        return None

    # Sample chips for statistics
    sample_size = min(n_samples, len(metadata))
    sample_indices = np.random.choice(len(metadata), sample_size, replace=False)

    print(f"Using {sample_size} chips for statistics...")

    base_dir = os.path.dirname(metadata_file)

    all_chips = []
    for idx in sample_indices:
        chip_path = os.path.join(base_dir, metadata[idx]["chip_path"])
        chip = np.load(chip_path)
        all_chips.append(chip)

    # Stack all chips: (n_samples, n_bands, height, width)
    all_chips = np.stack(all_chips, axis=0)

    # Compute per-band statistics
    means = np.mean(all_chips, axis=(0, 2, 3))
    stds = np.std(all_chips, axis=(0, 2, 3))

    # Avoid division by zero
    stds = np.where(stds == 0, 1.0, stds)

    # Print per-band statistics
    print("Per-band statistics:")
    band_names = metadata[0]["band_names"] if metadata else []
    for i, (mean, std) in enumerate(zip(means, stds)):
        band_name = band_names[i] if i < len(band_names) else f"Band_{i}"
        print(f"  {band_name}: mean={mean:.3f}, std={std:.3f}")

    stats = {
        "means": means,
        "stds": stds,
        "n_samples_used": sample_size,
        "n_bands": len(means),
        "band_names": band_names,
    }

    # Save statistics
    stats_path = os.path.join(base_dir, "normalization_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nNormalization statistics saved to: {stats_path}")

    return stats

def load_raw_feature_arrays(example, target_size):
    """Extract raw feature arrays from one TFRecord example, keyed by canonical band name."""
    raw_arrays = {}

    for key, feature in example.features.feature.items():
        if key == "label":
            continue

        if feature.HasField("float_list"):
            values = np.array(feature.float_list.value, dtype=np.float32)
        elif feature.HasField("bytes_list"):
            values = np.frombuffer(feature.bytes_list.value[0], dtype=np.uint8).astype(
                np.float32
            )
        else:
            continue

        band_size = int(np.sqrt(len(values)))
        band = values.reshape(band_size, band_size)

        if band.shape != target_size:
            band = resize_array(band, target_size)

        raw_arrays[legacy_to_canonical(key)] = band

    return raw_arrays


def confirm_one_chip_against_raw(
    metadata_file,
    sample_index=0,
    channel_pairs=[
        ("B4_post", "B4_post"),   # visible
        ("B8_post", "B8_post"),   # infrared
        ("dNBR", "dNBR"),         # derived
    ],
    output_dir="debug/raw_confirmation",
):
    """
    Manually confirm that selected raw TFRecord bands match processed channels.

    channel_pairs should be a list of:
        (raw_band_name, processed_band_name)
    Both are canonical names now (load_raw_feature_arrays renames raw keys
    the same way the processor does), so in practice the pair is usually
    the same name twice.
    Example:
        [("B4_post", "B4_post"), ("B8_post", "B8_post"), ("dNBR", "dNBR")]
    """

    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(metadata_file)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    item = metadata[sample_index]
    image = np.load(os.path.join(base_dir, item["chip_path"]))
    band_names = item["band_names"]
    target_size = image.shape[1:]

    raw_tfrecord_path = item["source_tfrecord"]
    raw_record_index = item["source_record_index"]

    dataset = tf.data.TFRecordDataset(str(raw_tfrecord_path))

    raw_record = None
    for record_index, raw in enumerate(dataset):
        if record_index == raw_record_index:
            raw_record = raw
            break

    if raw_record is None:
        raise IndexError(
            f"Could not find record {raw_record_index} in {raw_tfrecord_path}"
        )

    example = tf.train.Example.FromString(raw_record.numpy())
    raw_arrays = load_raw_feature_arrays(example, target_size)

    if channel_pairs is None:
        channel_pairs = [
            ("B4_post", "B4_post"),
            ("B8_post", "B8_post"),
            ("dNBR", "dNBR"),
        ]

    print("\n=== RAW TO PROCESSED CHANNEL CONFIRMATION ===")
    print(f"Chip id: {item['chip_id']}")
    print(f"Processed chip path: {os.path.join(base_dir, item['chip_path'])}")
    print(f"Raw TFRecord: {raw_tfrecord_path}")
    print(f"Raw record index: {raw_record_index}")
    print("\nProcessed band order:")
    for channel_index, band_name in enumerate(band_names):
        print(f"  {channel_index:02d}: {band_name}")

    for raw_band_name, processed_band_name in channel_pairs:
        if raw_band_name not in raw_arrays:
            print(f"\nMissing raw band: {raw_band_name}")
            continue

        if processed_band_name not in band_names:
            print(f"\nMissing processed band: {processed_band_name}")
            continue

        processed_index = band_names.index(processed_band_name)
        raw_channel = raw_arrays[raw_band_name]
        processed_channel = image[processed_index]

        diff = np.abs(raw_channel - processed_channel)
        mean_abs_diff = float(diff.mean())
        max_abs_diff = float(diff.max())
        matches = bool(
            np.allclose(raw_channel, processed_channel, atol=1e-6, rtol=1e-6)
        )

        print(
            f"\nraw {raw_band_name} -> processed channel {processed_index} "
            f"({processed_band_name})"
        )
        print(
            f"  raw stats:       min={float(raw_channel.min()):.4f} "
            f"max={float(raw_channel.max()):.4f} "
            f"mean={float(raw_channel.mean()):.4f} "
            f"std={float(raw_channel.std()):.4f}"
        )
        print(
            f"  processed stats: min={float(processed_channel.min()):.4f} "
            f"max={float(processed_channel.max()):.4f} "
            f"mean={float(processed_channel.mean()):.4f} "
            f"std={float(processed_channel.std()):.4f}"
        )
        print(
            f"  difference:      mean_abs={mean_abs_diff:.8f} "
            f"max_abs={max_abs_diff:.8f} "
            f"allclose={matches}"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(raw_channel, cmap="gray")
        axes[0].set_title(f"Raw: {raw_band_name}")
        axes[0].axis("off")

        axes[1].imshow(processed_channel, cmap="gray")
        axes[1].set_title(f"Processed {processed_index}: {processed_band_name}")
        axes[1].axis("off")

        axes[2].imshow(diff, cmap="magma")
        axes[2].set_title("Absolute difference")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{item['chip_id']}_{processed_index:02d}_{processed_band_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Change-based deforestation TFRecord processor"
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset contract id under configs/datasets/, e.g. "
        "legacy_threshold_v1. Supplies the input glob, output dir, and "
        "dNBR/dNDVI thresholds.",
    )
    args = parser.parse_args()

    # Install scipy if needed
    try:
        from scipy import ndimage
    except ImportError:
        print("Installing scipy for image resizing...")
        import subprocess

        subprocess.check_call(["pip", "install", "scipy"])
        from scipy import ndimage

    # Process TFRecords
    metadata = process_tfrecords_with_change_labels(args.dataset_id)

    if metadata:
        contract = load_contract(args.dataset_id)
        output_dir = contract.processed_path
        metadata_path = f"{output_dir}/metadata.pkl"

        # Inspect processed sample
        inspect_processed_sample(metadata_path, sample_index=0)

        confirm_one_chip_against_raw(metadata_path, sample_index=0)

        # Compute normalization stats
        stats = compute_normalization_stats_final(metadata_path)

        manifest_path = write_dataset_manifest(output_dir, contract, metadata)
        print(f"Dataset manifest: {manifest_path}")

        print(f"\nData processing complete")
        print("\nNext steps:")
        print(f"1. python src/split_data.py --dataset-id {args.dataset_id}")
        print(f"2. python src/train.py --experiment <experiment_id>")
        print("3. jupyter notebook notebooks/explore_data.ipynb")
    else:
        print("No deforestation detected. Consider adjusting thresholds")
