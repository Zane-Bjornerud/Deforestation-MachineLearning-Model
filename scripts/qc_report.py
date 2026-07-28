#!/usr/bin/env python
"""
Automated QC report for a processed dataset, run before trusting its output
for training or gating a new GEE export.

Usage:
    python scripts/qc_report.py --dataset-id existing_gfc_recovery_v0

Reads dataset_manifest.json + metadata.pkl from the dataset's processed_path
(see configs/datasets/<dataset_id>.yaml), streams through every chip/mask
.npy once to compute dataset-wide statistics without loading everything into
memory at once, and renders a fixed number of randomly selected chips as
multi-panel visual inspections (pre/post RGB, NDVI/NBR, dNDVI/dNBR, GFC mask).

Writes:
    <processed_path>/qc/qc_report.json   -- machine-readable report
    <processed_path>/qc/visual/*.png     -- visual inspection panels
Prints a human-readable summary to stdout, including an automatic pass/fail
read on the objective Gate A criteria (order, shape, dtype, no unexplained
extremes). "Masks are semantically correct" still requires a human to look
at the visual panels -- this script cannot certify that on its own.
"""

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from band_names import CANONICAL_BAND_ORDER
from dataset_contract import load_contract

NEARLY_FULL_MASK_THRESHOLD = 0.9  # fraction of positive pixels counted as "nearly full"


def _load_dataset(dataset_id):
    contract = load_contract(dataset_id)
    processed_path = contract.processed_path

    manifest_path = os.path.join(processed_path, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found -- run the processor for {dataset_id} first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    metadata_path = os.path.join(processed_path, "metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return contract, manifest, metadata


def _check_duplicates(metadata):
    chip_ids = [item["chip_id"] for item in metadata]
    seen = set()
    duplicates = set()
    for chip_id in chip_ids:
        if chip_id in seen:
            duplicates.add(chip_id)
        seen.add(chip_id)
    return sorted(duplicates)


def _check_channel_order(metadata):
    """Every chip's band_names must equal CANONICAL_BAND_ORDER exactly."""
    bad = [
        item["chip_id"]
        for item in metadata
        if item.get("band_names") != CANONICAL_BAND_ORDER
    ]
    return bad


def _streaming_chip_stats(processed_path, metadata):
    """Single pass over every chip+mask .npy: shape/dtype consistency, NaN/inf
    counts, per-channel min/max/mean/std (running accumulators, not stacked
    in memory), and mask dtype/unique-value tracking."""
    n_channels = len(CANONICAL_BAND_ORDER)
    expected_shape = None
    expected_mask_shape = None
    chip_dtypes = set()
    mask_dtypes = set()
    mask_unique_values = set()

    shape_mismatches = []
    mask_shape_mismatches = []

    count = 0
    running_sum = np.zeros(n_channels, dtype=np.float64)
    running_sumsq = np.zeros(n_channels, dtype=np.float64)
    running_min = np.full(n_channels, np.inf, dtype=np.float64)
    running_max = np.full(n_channels, -np.inf, dtype=np.float64)
    nan_counts = np.zeros(n_channels, dtype=np.int64)
    inf_counts = np.zeros(n_channels, dtype=np.int64)

    for item in metadata:
        chip = np.load(os.path.join(processed_path, item["chip_path"]))
        mask = np.load(os.path.join(processed_path, item["mask_path"]))

        if expected_shape is None:
            expected_shape = chip.shape
        elif chip.shape != expected_shape:
            shape_mismatches.append((item["chip_id"], chip.shape))

        if expected_mask_shape is None:
            expected_mask_shape = mask.shape
        elif mask.shape != expected_mask_shape:
            mask_shape_mismatches.append((item["chip_id"], mask.shape))

        chip_dtypes.add(str(chip.dtype))
        mask_dtypes.add(str(mask.dtype))
        mask_unique_values.update(np.unique(mask).tolist())

        finite = np.isfinite(chip)
        nan_counts += np.isnan(chip).sum(axis=(1, 2))
        inf_counts += np.isinf(chip).sum(axis=(1, 2))

        chip_finite = np.where(finite, chip, np.nan)
        running_sum += np.nansum(chip_finite, axis=(1, 2))
        running_sumsq += np.nansum(chip_finite**2, axis=(1, 2))
        chan_min = np.nanmin(np.where(finite, chip, np.inf), axis=(1, 2))
        chan_max = np.nanmax(np.where(finite, chip, -np.inf), axis=(1, 2))
        running_min = np.minimum(running_min, chan_min)
        running_max = np.maximum(running_max, chan_max)

        count += 1

    pixels_per_channel = count * expected_shape[1] * expected_shape[2]
    mean = running_sum / pixels_per_channel
    var = running_sumsq / pixels_per_channel - mean**2
    std = np.sqrt(np.maximum(var, 0))

    return {
        "chips_scanned": count,
        "chip_shape": list(expected_shape) if expected_shape else None,
        "mask_shape": list(expected_mask_shape) if expected_mask_shape else None,
        "chip_shape_mismatches": shape_mismatches,
        "mask_shape_mismatches": mask_shape_mismatches,
        "chip_dtypes": sorted(chip_dtypes),
        "mask_dtypes": sorted(mask_dtypes),
        "mask_unique_values": sorted(mask_unique_values),
        "per_channel_stats": [
            {
                "band": band,
                "min": float(running_min[i]),
                "max": float(running_max[i]),
                "mean": float(mean[i]),
                "std": float(std[i]),
                "nan_count": int(nan_counts[i]),
                "inf_count": int(inf_counts[i]),
            }
            for i, band in enumerate(CANONICAL_BAND_ORDER)
        ],
        "total_nan_count": int(nan_counts.sum()),
        "total_inf_count": int(inf_counts.sum()),
    }


def _mask_distribution(metadata):
    fractions = np.array([item["deforestation_fraction"] for item in metadata])
    n = len(fractions)
    return {
        "n_chips": n,
        "min": float(fractions.min()),
        "max": float(fractions.max()),
        "mean": float(fractions.mean()),
        "median": float(np.median(fractions)),
        "p90": float(np.percentile(fractions, 90)),
        "p99": float(np.percentile(fractions, 99)),
        "empty_mask_pct": float((fractions == 0).mean() * 100),
        "nearly_full_mask_pct": float(
            (fractions > NEARLY_FULL_MASK_THRESHOLD).mean() * 100
        ),
        "nearly_full_mask_threshold": NEARLY_FULL_MASK_THRESHOLD,
    }


def _percentile_stretch(band, lo=2, hi=98):
    lo_val, hi_val = np.percentile(band, [lo, hi])
    if hi_val <= lo_val:
        return np.zeros_like(band)
    stretched = (band - lo_val) / (hi_val - lo_val)
    return np.clip(stretched, 0, 1)


def _render_visual_panels(processed_path, metadata, n_visual, output_dir, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    sample = rng.sample(metadata, min(n_visual, len(metadata)))

    idx = {name: i for i, name in enumerate(CANONICAL_BAND_ORDER)}
    saved_paths = []

    for item in sample:
        chip = np.load(os.path.join(processed_path, item["chip_path"]))
        mask = np.load(os.path.join(processed_path, item["mask_path"]))[0]

        pre_rgb = np.stack(
            [
                _percentile_stretch(chip[idx["B4_pre"]]),
                _percentile_stretch(chip[idx["B3_pre"]]),
                _percentile_stretch(chip[idx["B2_pre"]]),
            ],
            axis=-1,
        )
        post_rgb = np.stack(
            [
                _percentile_stretch(chip[idx["B4_post"]]),
                _percentile_stretch(chip[idx["B3_post"]]),
                _percentile_stretch(chip[idx["B2_post"]]),
            ],
            axis=-1,
        )

        panels = [
            ("pre RGB", pre_rgb, None),
            ("post RGB", post_rgb, None),
            ("pre NDVI", chip[idx["NDVI_pre"]], "RdYlGn"),
            ("pre NBR", chip[idx["NBR_pre"]], "RdYlGn"),
            ("post NDVI", chip[idx["NDVI_post"]], "RdYlGn"),
            ("post NBR", chip[idx["NBR_post"]], "RdYlGn"),
            ("dNDVI", chip[idx["dNDVI"]], "RdBu_r"),
            ("dNBR", chip[idx["dNBR"]], "RdBu_r"),
            ("GFC reference mask", mask, "gray"),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for ax, (title, data, cmap) in zip(axes.ravel(), panels):
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            if cmap is not None:
                fig.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(
            f"chip_id={item['chip_id']}  source_shard={os.path.basename(item.get('source_tfrecord') or 'unknown')}\n"
            f"channel_order={CANONICAL_BAND_ORDER}",
            fontsize=7,
            wrap=True,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out_path = os.path.join(output_dir, f"{item['chip_id']}.png")
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Automated QC report for a processed dataset")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--n-visual", type=int, default=25)
    args = parser.parse_args()

    contract, manifest, metadata = _load_dataset(args.dataset_id)
    processed_path = contract.processed_path
    qc_dir = os.path.join(processed_path, "qc")
    visual_dir = os.path.join(qc_dir, "visual")
    os.makedirs(qc_dir, exist_ok=True)

    print(f"=== QC report: {args.dataset_id} ===")
    print(f"Processed path: {processed_path}")

    duplicates = _check_duplicates(metadata)
    bad_order = _check_channel_order(metadata)
    stream_stats = _streaming_chip_stats(processed_path, metadata)
    mask_dist = _mask_distribution(metadata)
    visual_paths = _render_visual_panels(processed_path, metadata, args.n_visual, visual_dir)

    export_qc = manifest.get("qc", {})

    report = {
        "dataset_id": args.dataset_id,
        "label_mode": contract.label_mode,
        "shards_discovered": export_qc.get("shards_discovered"),
        "shards_processed": export_qc.get("shards_processed"),
        "total_records_read": export_qc.get("total_records_read"),
        "records_skipped_no_label": export_qc.get("records_skipped_no_label"),
        "records_skipped_no_bands": export_qc.get("records_skipped_no_bands"),
        "records_failed": export_qc.get("records_failed"),
        "total_chips_written": manifest.get("chip_count"),
        "duplicate_chip_ids": duplicates,
        "chips_with_wrong_channel_order": bad_order,
        **stream_stats,
        "mask_positive_pixel_distribution": mask_dist,
        "visual_inspection_count": len(visual_paths),
        "visual_inspection_dir": visual_dir,
    }

    report_path = os.path.join(qc_dir, "qc_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Console summary + automatic (partial) Gate A read ---
    print("\n--- Counts ---")
    print(f"Shards discovered/processed: {report['shards_discovered']}/{report['shards_processed']}")
    print(f"Total records read: {report['total_records_read']}")
    print(f"Skipped (no label): {report['records_skipped_no_label']}")
    print(f"Skipped (no bands): {report['records_skipped_no_bands']}")
    print(f"Failed records: {report['records_failed']}")
    print(f"Total chips written: {report['total_chips_written']}")

    print("\n--- Shape / dtype / order ---")
    print(f"Chip shape: {stream_stats['chip_shape']} (expected [18, 256, 256])")
    print(f"Mask shape: {stream_stats['mask_shape']}")
    print(f"Chip shape mismatches: {len(stream_stats['chip_shape_mismatches'])}")
    print(f"Mask shape mismatches: {len(stream_stats['mask_shape_mismatches'])}")
    print(f"Chip dtypes: {stream_stats['chip_dtypes']}")
    print(f"Mask dtypes: {stream_stats['mask_dtypes']}, unique values: {stream_stats['mask_unique_values']}")
    print(f"Chips with wrong channel order: {len(bad_order)}")
    print(f"Duplicate chip_ids: {len(duplicates)}")
    print(f"Total NaN: {stream_stats['total_nan_count']}, total Inf: {stream_stats['total_inf_count']}")

    print("\n--- Positive-pixel distribution ---")
    for k, v in mask_dist.items():
        print(f"  {k}: {v}")

    print(f"\nVisual inspection panels: {len(visual_paths)} written to {visual_dir}")
    print(f"Full report: {report_path}")

    mask_is_binary = set(stream_stats["mask_unique_values"]).issubset({0.0, 1.0})

    print("\n--- Automatic Gate A read (objective checks only) ---")
    print(f"[{'PASS' if not stream_stats['chip_shape_mismatches'] and stream_stats['chip_shape'] == [18, 256, 256] else 'FAIL'}] output consistently [18, 256, 256]")
    print(f"[{'PASS' if not bad_order else 'FAIL'}] canonical channel order verified")
    print(f"[{'PASS' if mask_is_binary else 'FAIL'}] masks are binary (values: {stream_stats['mask_unique_values']})")
    print(f"[{'PASS' if not duplicates else 'FAIL'}] no duplicate chip identifiers")
    print(f"[{'PASS' if stream_stats['total_nan_count'] == 0 and stream_stats['total_inf_count'] == 0 else 'FAIL'}] no NaN/Inf values")
    print("[MANUAL] masks are semantically correct -- review visual panels")
    print("[MANUAL] a small training run can load the processed output -- see smoke test")


if __name__ == "__main__":
    main()
