import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def create_data_splits(
    metadata_file,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by_deforestation=True,
):
    """
    Create train/validation/test splits from metadata.

    Args:
        metadata_file: Path to metadata pickle file
        output_dir: Directory to save split files
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
        stratify_by_deforestation: Whether to stratify by presence of deforestation
    """

    # Load metadata
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    print(f"Total samples: {len(metadata)}")

    # Create arrays for splitting
    indices = np.arange(len(metadata))

    if stratify_by_deforestation:
        # Use deforestation presence for stratification
        stratify_labels = [int(item["has_deforestation"]) for item in metadata]
        print(f"Samples with deforestation: {sum(stratify_labels)}")
        print(
            f"Samples without deforestation: {len(stratify_labels) - sum(stratify_labels)}"
        )
    else:
        stratify_labels = None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=stratify_labels,
        random_state=42,
    )

    # Second split: val vs test
    if stratify_by_deforestation:
        temp_labels = [stratify_labels[i] for i in temp_indices]
    else:
        temp_labels = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=42,
    )

    # Create split metadata
    train_metadata = [metadata[i] for i in train_indices]
    val_metadata = [metadata[i] for i in val_indices]
    test_metadata = [metadata[i] for i in test_indices]

    # Save splits
    splits = {"train": train_metadata, "val": val_metadata, "test": test_metadata}

    for split_name, split_data in splits.items():
        with open(f"{output_dir}/{split_name}_metadata.pkl", "wb") as f:
            pickle.dump(split_data, f)

        # Print statistics
        n_total = len(split_data)
        n_positive = sum(1 for item in split_data if item["has_deforestation"])
        print(
            f"{split_name}: {n_total} samples, {n_positive} positive ({n_positive/n_total*100:.1f}%)"
        )

    print("\nData splits created successfully!")
    return splits


if __name__ == "__main__":
    # Dataset directory: chips/, masks/, and metadata.pkl all live here.
    # Swap to data/processed/gee_canary_gfc_v1 or gee_full_gfc_v1 once those
    # are populated by scripts/gee_export_chips.py + GFC_process_tfrecords4.py.
    dataset_dir = "data/processed/legacy_threshold_v1"

    metadata_file = f"{dataset_dir}/metadata.pkl"
    output_dir = dataset_dir

    splits = create_data_splits(metadata_file, output_dir)
