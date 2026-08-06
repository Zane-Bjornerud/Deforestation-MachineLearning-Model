import pickle
import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from dataset_contract import load_contract, validate_metadata_matches_contract


def create_data_splits(
    metadata_file,
    output_dir,
    contract,
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
        contract: DatasetContract this metadata must match (see
            src/dataset_contract.py) -- validated before splitting so a
            wrong-processor or stale metadata.pkl fails loudly here rather
            than silently propagating into training.
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
        stratify_by_deforestation: Whether to stratify by presence of deforestation
    """

    # Load metadata
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    print(f"Total samples: {len(metadata)}")

    validate_metadata_matches_contract(metadata, contract)

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


def create_block_splits(
    metadata_file,
    output_dir,
    contract,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    buffer_bands=1,
    seed=42,
):
    """Create train/validation/test splits as contiguous geographic bands
    of spatial blocks.

    Unlike create_data_splits (random, chip-level, stratified only by
    deforestation presence), this assigns whole spatial blocks --
    block_size_tiles x block_size_tiles groups of chips, see
    src/spatial_blocks.py -- to a single split, as three contiguous bands
    along one grid axis (block_row or block_col, chosen from `seed`), sized
    by cumulative chip count to hit the target ratios. A `buffer_bands`-wide
    strip of blocks is dropped entirely at each of the two band boundaries.

    Note: an earlier version of this function assigned each block
    independently via a random per-block hash and buffered any block
    adjacent to a differently-assigned neighbor. That doesn't work -- with
    i.i.d. random per-block labels, the great majority of blocks end up
    touching at least one differently-labeled neighbor (same reason a
    random 0/1 grid looks like static, not solid regions), so the buffer
    step ate 80-90% of the data. Contiguous banding keeps the boundary
    between splits to two lines instead of scattering it across the whole
    grid.

    This avoids the leakage inherent to chip-level random splitting: two
    neighboring chips show near-identical imagery/context (same forest
    edge, same clouds, same acquisition day), so a random split can put one
    in train and its neighbor in val/test, letting the model score well by
    having effectively already seen that immediate area. Reported metrics
    from this split reflect performance on genuinely unseen geography,
    matching the leakage caveat documented in docs/model-card.md.

    Trade-off: because train/val/test are contiguous bands rather than
    scattered blocks, each split can only be as representative of the whole
    AOI as the band it landed in -- e.g. if deforestation activity is
    concentrated in one part of the AOI, whichever band contains it will
    have a very different positive rate than the others. The per-split
    composition printed below (and the empty/zero-positive check) is how
    that gets caught rather than silently trained on.

    Requires metadata produced with spatial block fields (block_id,
    block_row, block_col) -- i.e. processed with a mixer.json present. Raises
    if any record is missing them; callers should fall back to
    create_data_splits otherwise (see __main__ below).
    """
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    print(f"Total samples: {len(metadata)}")
    validate_metadata_matches_contract(metadata, contract)

    missing = [m["chip_id"] for m in metadata if "block_id" not in m]
    if missing:
        raise ValueError(
            f"{len(missing)} chips are missing block_id (e.g. {missing[0]!r}) "
            "-- this metadata wasn't processed with mixer.json spatial "
            "metadata available. Use create_data_splits instead, or "
            "re-process this dataset with a mixer.json present in raw_path."
        )

    blocks = defaultdict(list)  # block_id -> list of chip metadata dicts
    for m in metadata:
        blocks[m["block_id"]].append(m)

    # Band along block_row or block_col, picked from the seed so repeated
    # datasets don't all hold out the same edge of their AOI.
    axis = "block_row" if random.Random(seed).random() < 0.5 else "block_col"
    axis_value_of = {bid: chips[0][axis] for bid, chips in blocks.items()}
    print(f"Banding along {axis}")

    chip_count_by_axis_value = defaultdict(int)
    for bid, chips in blocks.items():
        chip_count_by_axis_value[axis_value_of[bid]] += len(chips)

    sorted_values = sorted(chip_count_by_axis_value)
    total_chips = sum(chip_count_by_axis_value.values())

    band_assignment = {}
    cumulative = 0
    for v in sorted_values:
        cumulative += chip_count_by_axis_value[v]
        frac = cumulative / total_chips
        if frac <= train_ratio:
            band_assignment[v] = "train"
        elif frac <= train_ratio + val_ratio:
            band_assignment[v] = "val"
        else:
            band_assignment[v] = "test"

    # Buffer: drop buffer_bands axis-values on each side of any boundary
    # where the band assignment changes.
    buffered_values = set()
    for i in range(len(sorted_values) - 1):
        if band_assignment[sorted_values[i]] != band_assignment[sorted_values[i + 1]]:
            for k in range(buffer_bands):
                if i - k >= 0:
                    buffered_values.add(sorted_values[i - k])
                if i + 1 + k < len(sorted_values):
                    buffered_values.add(sorted_values[i + 1 + k])

    for v in buffered_values:
        band_assignment[v] = "buffer"

    assignment = {bid: band_assignment[axis_value_of[bid]] for bid in blocks}

    splits = {"train": [], "val": [], "test": [], "buffer": []}
    for bid, chips in blocks.items():
        splits[assignment[bid]].extend(chips)

    n_blocks = {name: sum(1 for a in assignment.values() if a == name) for name in splits}
    print(
        f"Blocks: {len(blocks)} total -- train={n_blocks['train']} "
        f"val={n_blocks['val']} test={n_blocks['test']} "
        f"buffer={n_blocks['buffer']} (buffer blocks are dropped, not trained "
        "or evaluated on)"
    )

    for split_name in ("train", "val", "test"):
        split_chips = splits[split_name]
        with open(f"{output_dir}/{split_name}_metadata.pkl", "wb") as f:
            pickle.dump(split_chips, f)

        n_total = len(split_chips)
        n_positive = sum(1 for item in split_chips if item["has_deforestation"])
        if n_total:
            print(
                f"{split_name}: {n_total} chips, {n_positive} positive "
                f"({n_positive/n_total*100:.1f}%)"
            )
        else:
            print(f"{split_name}: 0 chips")
        if n_total == 0 or n_positive == 0:
            raise ValueError(
                f"{split_name} split is empty or has zero positive chips -- "
                "try a smaller buffer_bands, a larger block_size_tiles "
                "(fewer, coarser bands), or check whether deforestation is "
                "concentrated in one part of the AOI."
            )

    print("\nBlock splits created successfully!")
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split a processed dataset into train/val/test"
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset contract id under configs/datasets/, e.g. "
        "gee_full_gfc_v1 or legacy_threshold_v1.",
    )
    args = parser.parse_args()

    contract = load_contract(args.dataset_id)
    dataset_dir = contract.processed_path

    metadata_file = f"{dataset_dir}/metadata.pkl"
    output_dir = dataset_dir

    # Use the block-level spatial split whenever the processed metadata
    # carries block_id (i.e. was processed with mixer.json available -- see
    # src/spatial_blocks.py); otherwise fall back to the legacy random
    # stratified split (older processed datasets, or datasets without a
    # mixer.json, e.g. change_based_processor output).
    with open(metadata_file, "rb") as f:
        sample_metadata = pickle.load(f)
    has_block_metadata = bool(sample_metadata) and "block_id" in sample_metadata[0]

    if has_block_metadata:
        print("block_id found in metadata -- using block-level spatial split.")
        splits = create_block_splits(metadata_file, output_dir, contract)
    else:
        print(
            "No block_id in metadata -- falling back to the random "
            "stratified split (chip-level; known geographic leakage risk, "
            "see docs/model-card.md)."
        )
        splits = create_data_splits(metadata_file, output_dir, contract)
