#!/usr/bin/env python
"""
Gate A check: "a small training run can load the processed output."

Does NOT run a full training loop (too slow on CPU for a gate check) --
instead proves the exact path train.py uses actually works end to end:
contract validation, DeforestationDataset construction, one DataLoader
batch, and one forward pass through the real model class (smp.Unet).

Usage:
    python scripts/smoke_test_training.py --dataset-id existing_gfc_recovery_v0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from dataset_contract import load_contract


def main():
    parser = argparse.ArgumentParser(description="Smoke test: can training load this dataset?")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    contract = load_contract(args.dataset_id)
    data_dir = contract.processed_path
    train_metadata = f"{data_dir}/train_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"

    if not os.path.exists(train_metadata):
        print(f"FAIL: {train_metadata} not found -- run split_data.py --dataset-id {args.dataset_id} first")
        sys.exit(1)

    print(f"Loading DeforestationDataset for {args.dataset_id} (label_mode={contract.label_mode})...")
    dataset = DeforestationDataset(data_dir, train_metadata, norm_stats, contract, augment=False)
    print(f"PASS: dataset loaded and validated against contract, {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    xb, yb = next(iter(loader))
    print(f"PASS: one batch loaded, chip batch shape={tuple(xb.shape)}, mask batch shape={tuple(yb.shape)}")

    expected_shape = (2, 18, 256, 256)
    if tuple(xb.shape) != expected_shape:
        print(f"FAIL: expected chip batch shape {expected_shape}, got {tuple(xb.shape)}")
        sys.exit(1)

    model = smp.Unet(encoder_name="resnet34", in_channels=xb.shape[1], classes=1)
    model.eval()
    with torch.no_grad():
        logits = model(xb)
    print(f"PASS: forward pass through smp.Unet succeeded, logits shape={tuple(logits.shape)}")

    print(f"\nGate A training-load check: PASS ({args.dataset_id} loads and forward-passes cleanly)")


if __name__ == "__main__":
    main()
