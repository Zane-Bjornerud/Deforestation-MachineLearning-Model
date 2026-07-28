import numpy as np
import os
import pickle
from pathlib import Path
import tensorflow as tf
from scipy import ndimage

from band_names import CANONICAL_BAND_ORDER, raw_bands_to_canonical_order
from dataset_contract import (
    assert_no_label_source_conflict,
    load_contract,
    validate_processor_identity,
    write_dataset_manifest,
)

PROCESSOR_NAME = "GFC_process_tfrecords4"


def resize_array(array, target_size):
    """Resize array to target size using bilinear interpolation."""
    if array.shape == target_size:
        return array

    # Calculate zoom factors
    zoom_factors = (target_size[0] / array.shape[0], target_size[1] / array.shape[1])

    # Resize using scipy
    resized = ndimage.zoom(array, zoom_factors, order=1)  # order=1 for bilinear

    return resized


def process_tfrecords_resolution_fixed(dataset_id):
    """GFC-labeled TFRecord processor that handles resolution mismatches.

    Args:
        dataset_id: Dataset contract id, e.g. "gee_full_gfc_v1". Looked up
            under configs/datasets/<dataset_id>.yaml, which supplies the
            input glob (raw_path/*.tfrecord), output_dir (processed_path),
            and the Hansen label parameters recorded alongside every chip.
    """

    print("=== Resolution-Fixed TFRecord Processor (GFC labels) ===\n")

    contract = load_contract(dataset_id)
    validate_processor_identity(contract, PROCESSOR_NAME)

    input_glob = f"{contract.raw_path}/*.tfrecord"
    output_dir = contract.processed_path

    assert_no_label_source_conflict(output_dir, contract.label_source)

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

    # QC counters, folded into the dataset manifest at the end -- lets the
    # QC report (scripts/qc_report.py) show shard/record/failure counts
    # without re-parsing the raw TFRecords a second time.
    qc = {
        "shards_discovered": len(tfrecord_files),
        "shards_processed": 0,
        "total_records_read": 0,
        "records_skipped_no_label": 0,
        "records_skipped_no_bands": 0,
        "records_failed": 0,
    }

    # We'll use 256x256 as our target resolution (since that's what the features are)
    target_size = (256, 256)

    print(f"Target resolution: {target_size[0]}x{target_size[1]}")
    print(
        f"Label config: dataset_id={contract.dataset_id} "
        f"hansen_asset={contract.extra.get('gfc_asset')} "
        f"forest_cover_threshold={contract.extra.get('forest_cover_threshold')} "
        f"target_year={contract.target_year}"
    )

    # Process each TFRecord file
    for tfrecord_file in tfrecord_files:
        print(f"\nProcessing {tfrecord_file.name}...")

        dataset = tf.data.TFRecordDataset(str(tfrecord_file))

        file_count = 0
        for raw_record in dataset:
            qc["total_records_read"] += 1
            try:
                # Parse the TF Example
                example = tf.train.Example.FromString(raw_record.numpy())
                features = example.features.feature

                # Extract label first
                label_data = None
                if "label" in features:
                    label_feature = features["label"]
                    if label_feature.HasField("bytes_list"):
                        data = label_feature.bytes_list.value[0]
                        # The label band is exported via .toByte() (see
                        # scripts/gee_export_chips.py build_label()) -- one
                        # UINT8 byte per pixel, NOT float32. Decoding it as
                        # float32 misreads the buffer (4x too few "pixels",
                        # garbage values) and silently zeroes out almost
                        # every positive label after thresholding.
                        label_array = np.frombuffer(data, dtype=np.uint8).astype(
                            np.float32
                        )

                        # Determine original label size
                        label_size = int(np.sqrt(len(label_array)))
                        label_reshaped = label_array.reshape(label_size, label_size)

                        # Resize label to target size if needed
                        if (label_size, label_size) != target_size:
                            label_data = resize_array(label_reshaped, target_size)
                            if chip_count == 0:
                                print(
                                    f"Resizing labels from {label_size}x{label_size} to {target_size[0]}x{target_size[1]}"
                                )
                        else:
                            label_data = label_reshaped

                if label_data is None:
                    qc["records_skipped_no_label"] += 1
                    continue

                # Extract feature bands into a dict keyed by raw band name.
                # Order is NOT decided here -- dict/protobuf map iteration
                # order is not a channel-order guarantee. Channel order is
                # fixed below via raw_bands_to_canonical_order.
                raw_bands = {}

                # Process all non-label features
                for key, feature in features.items():
                    if key == "label":
                        continue

                    if feature.HasField("float_list"):
                        values = feature.float_list.value
                        band_array = np.array(values, dtype=np.float32)

                        # Determine band size
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
                        # label, which is skipped above). GEE byte-encodes
                        # with .toByte() -> UINT8, not float32 -- see the
                        # label parsing above for why this distinction
                        # matters (a float32 read silently corrupts the data).
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
                if len(raw_bands) > 0 and label_data is not None:

                    # Fix channel order explicitly from src/band_names.py --
                    # never from feature/dict iteration order.
                    chip_bands, band_names = raw_bands_to_canonical_order(raw_bands)

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
                    # Label: (1, height, width)
                    mask = label_data[np.newaxis, :, :]

                    # Convert label to binary (0 or 1) and handle potential floating point labels
                    mask = (mask > 0.5).astype(np.float32)

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
                            "has_deforestation": has_defor,
                            "deforestation_fraction": defor_fraction,
                            "patch_size": target_size[0],
                            "n_bands": len(chip_bands),
                            "band_names": band_names,
                            "dataset_id": contract.dataset_id,
                            "label_mode": contract.label_mode,
                            "label_source": contract.label_source,
                            "label_contract_version": contract.label_contract_version,
                            "target_year": contract.target_year,
                        }
                    )

                    chip_count += 1
                    file_count += 1

                    # Progress update
                    if chip_count % 100 == 0:
                        print(f"    Processed {chip_count} chips...")

                    # Show first few chips info
                    if chip_count <= 3:
                        print(
                            f"Chip {chip_count}: {len(chip_bands)} bands, {target_size[0]}x{target_size[1]}"
                        )
                        print(f"Shape: {chip.shape}")
                        print(f"Mask shape: {mask.shape}")
                        print(f"Has deforestation: {has_defor} ({defor_fraction:.3%})")
                        if chip_count == 1:
                            print(f"Bands: {band_names}")

                else:
                    qc["records_skipped_no_bands"] += 1
                    if file_count < 3:  # Only show warnings for first few
                        print(
                            f"Skipping: bands={len(raw_bands)}, label={label_data is not None}"
                        )

            except Exception as e:
                qc["records_failed"] += 1
                if file_count < 3:  # Only show errors for first few
                    print(f"Error: {e}")
                continue

        qc["shards_processed"] += 1
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

        # Sample chip info
        sample = all_metadata[0]
        sample_chip = np.load(f"{output_dir}/{sample['chip_path']}")
        sample_mask = np.load(f"{output_dir}/{sample['mask_path']}")

        print(f"\nFinal data format:")
        print(f"  Chip shape: {sample_chip.shape}")
        print(f"  Mask shape: {sample_mask.shape}")
        print(f"  Patch size: {sample['patch_size']}x{sample['patch_size']}")
        print(f"  Number of bands: {sample['n_bands']}")
        print(f"  Band names: {sample['band_names']}")

        # Show data ranges
        print(f"\nData ranges (first chip):")
        print(f"  Chip values: [{sample_chip.min():.3f}, {sample_chip.max():.3f}]")
        print(f"  Mask values: [{sample_mask.min():.3f}, {sample_mask.max():.3f}]")

        # Deforestation statistics
        if n_with_defor > 0:
            defor_chips = [x for x in all_metadata if x["has_deforestation"]]
            defor_fractions = [x["deforestation_fraction"] for x in defor_chips]
            print(f"\nDeforestation coverage in positive chips:")
            print(f"  Min: {min(defor_fractions):.3%}")
            print(f"  Max: {max(defor_fractions):.3%}")
            print(f"  Mean: {np.mean(defor_fractions):.3%}")

    print(f"\nFiles saved:")
    print(f"  Metadata: {metadata_path}")
    print(f"  Chips: {output_dir}/chips/ ({chip_count} files)")
    print(f"  Masks: {output_dir}/masks/ ({chip_count} files)")

    print(f"\nQC counters: {qc}")

    if chip_count > 0:
        print("\n Processing successful!")
        return all_metadata, qc
    else:
        print("\n No chips processed.")
        return None, qc


def compute_normalization_stats_final(metadata_file, n_samples=100):
    """Compute normalization statistics."""

    print(f"\nComputing normalization statistics...")

    base_dir = os.path.dirname(metadata_file)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GFC (Hansen Global Forest Change) labeled TFRecord processor"
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset contract id under configs/datasets/, e.g. "
        "gee_full_gfc_v1. Supplies the input glob, output dir, and Hansen "
        "label parameters.",
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
    metadata, qc = process_tfrecords_resolution_fixed(args.dataset_id)

    if metadata:
        contract = load_contract(args.dataset_id)
        output_dir = contract.processed_path
        metadata_path = f"{output_dir}/metadata.pkl"

        # Compute normalization stats
        stats = compute_normalization_stats_final(metadata_path)

        manifest_path = write_dataset_manifest(
            output_dir, contract, metadata, extra_manifest_fields={"qc": qc}
        )
        print(f"Dataset manifest: {manifest_path}")

        print(f"\n Data processing complete!")
        print("\nNext steps:")
        print(f"1. python src/split_data.py --dataset-id {args.dataset_id}")
        print(f"2. python src/train.py --experiment <experiment_id>")
        print("3. jupyter notebook notebooks/explore_data.ipynb")
    else:
        print(" No data processed successfully.")
