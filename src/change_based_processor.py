import numpy as np
import os
import pickle
from pathlib import Path
import tensorflow as tf
from scipy import ndimage


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


def process_tfrecords_with_change_labels():
    """Process TFRecords using change-based deforestation labels."""

    print("=== Change-Based Deforestation Processor ===\n")

    # Find TFRecord files
    tfrecord_files = list(Path("data").glob("*.tfrecord"))
    print(f"Found {len(tfrecord_files)} files:")
    for f in tfrecord_files:
        print(f"  {f.name}")

    # Create output directories
    os.makedirs("data/processed/chips", exist_ok=True)
    os.makedirs("data/processed/masks", exist_ok=True)

    all_metadata = []
    chip_count = 0
    target_size = (256, 256)

    # Different thresholds to try
    thresholds = {
        "conservative": {"dnbr": -0.2, "dndvi": -0.25},  # Strict criteria
        "moderate": {"dnbr": -0.15, "dndvi": -0.2},  # Balanced
        "sensitive": {"dnbr": -0.1, "dndvi": -0.15},  # Catches more cases
    }

    # Use moderate thresholds
    selected_threshold = "sensitive"
    dnbr_thresh = thresholds[selected_threshold]["dnbr"]
    dndvi_thresh = thresholds[selected_threshold]["dndvi"]

    print(f"Using {selected_threshold} thresholds:")
    print(f"  dNBR < {dnbr_thresh}")
    print(f"  dNDVI < {dndvi_thresh}")

    # Process each TFRecord file
    for tfrecord_file in tfrecord_files:
        print(f"\nProcessing {tfrecord_file.name}...")

        dataset = tf.data.TFRecordDataset(str(tfrecord_file))

        file_count = 0
        for raw_record in dataset:
            try:
                # Parse the TF Example
                example = tf.train.Example.FromString(raw_record.numpy())
                features = example.features.feature

                # Extract feature bands
                chip_bands = []
                band_names = []

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
                            if chip_count == 0 and len(chip_bands) == 0:
                                print(
                                    f"    Resizing features from {band_size}x{band_size} to {target_size[0]}x{target_size[1]}"
                                )
                        else:
                            band_resized = band_reshaped

                        chip_bands.append(band_resized)
                        band_names.append(key)

                    elif feature.HasField("bytes_list"):
                        # Handle bytes features (shouldn't be many besides original label)
                        data = feature.bytes_list.value[0]
                        band_array = np.frombuffer(data, dtype=np.float32)

                        band_size = int(np.sqrt(len(band_array)))
                        band_reshaped = band_array.reshape(band_size, band_size)

                        if (band_size, band_size) != target_size:
                            band_resized = resize_array(band_reshaped, target_size)
                        else:
                            band_resized = band_reshaped

                        chip_bands.append(band_resized)
                        band_names.append(key)

                # Check if we have valid data
                if len(chip_bands) > 0:

                    # Stack bands: (n_bands, height, width)
                    chip = np.stack(chip_bands, axis=0)

                    # Create deforestation mask from change indices
                    defor_mask = create_deforestation_labels_from_change(
                        chip, band_names, dnbr_thresh, dndvi_thresh
                    )

                    # Convert to (1, H, W) format
                    mask = defor_mask[np.newaxis, :, :]

                    # Save chip and mask
                    chip_id = f"chip_{chip_count:05d}"
                    chip_path = f"data/processed/chips/{chip_id}.npy"
                    mask_path = f"data/processed/masks/{chip_id}.npy"

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
                            "has_deforestation": has_defor,
                            "deforestation_fraction": defor_fraction,
                            "patch_size": target_size[0],
                            "n_bands": len(chip_bands),
                            "band_names": band_names,
                            "labeling_method": f"change_indices_{selected_threshold}",
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
                            f"    Processed {chip_count} chips, {current_defor} with deforestation ({current_defor/chip_count*100:.1f}%)"
                        )

                    # Show first few chips info
                    if chip_count <= 5:
                        print(
                            f"    Chip {chip_count}: {len(chip_bands)} bands, {target_size[0]}x{target_size[1]}"
                        )
                        print(
                            f"      Has deforestation: {has_defor} ({defor_fraction:.3%})"
                        )
                        if chip_count == 1:
                            print(f"      Bands: {band_names}")

            except Exception as e:
                if file_count < 3:  # Only show errors for first few
                    print(f"    Error: {e}")
                continue

        print(f"    Extracted {file_count} chips from {tfrecord_file.name}")

    # Save metadata
    metadata_path = "data/processed/metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
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

        print(f"\nLabeling method: Change indices with {selected_threshold} thresholds")
        print(f"Thresholds used: dNBR < {dnbr_thresh}, dNDVI < {dndvi_thresh}")

    print(f"\nFiles saved:")
    print(f"  Metadata: {metadata_path}")
    print(f"  Chips: data/processed/chips/ ({chip_count} files)")
    print(f"  Masks: data/processed/masks/ ({chip_count} files)")

    if chip_count > 0 and n_with_defor > 0:
        print("\n✅ Processing successful with deforestation detected!")
        return all_metadata
    else:
        print(
            "\n⚠️ Processing complete but no deforestation detected. Try adjusting thresholds."
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

    all_chips = []
    for idx in sample_indices:
        chip_path = f"data/processed/{metadata[idx]['chip_path']}"
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
    stats_path = "data/processed/normalization_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nNormalization statistics saved to: {stats_path}")

    return stats


if __name__ == "__main__":
    # Install scipy if needed
    try:
        from scipy import ndimage
    except ImportError:
        print("Installing scipy for image resizing...")
        import subprocess

        subprocess.check_call(["pip", "install", "scipy"])
        from scipy import ndimage

    # Process TFRecords
    metadata = process_tfrecords_with_change_labels()

    if metadata:
        # Compute normalization stats
        stats = compute_normalization_stats_final("data/processed/metadata.pkl")

        print(f"\n🎉 Data processing complete!")
        print("\nNext steps:")
        print("1. python src/split_data.py")
        print("2. python src/train.py")
        print("3. jupyter notebook notebooks/explore_data.ipynb")
    else:
        print("❌ No deforestation detected. Consider adjusting thresholds.")
