
import os

# Set before torch loads OpenMP, matching train.py. Without this the script
# aborts with OMP Error #15 on Mac.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless save; no display needed                                                                                                
import matplotlib.pyplot as plt 

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from band_names import CANONICAL_BAND_ORDER
from dataset_contract import load_contract
from train import DEVICE, dice_loss, focal_loss, load_experiment_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a checkpoint on the held-out test split."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment id under configs/experiments/ (same value used with train.py).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pth checkpoint, typically best_model.pth from a run's "
        "per-stamp checkpoint dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Eval batch size. Doesn't affect the metrics, only wall time.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Override the default output path.",
    )
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation (4x compute).")
    parser.add_argument(
        "--inference-dataset-id",
        required=True,
        help="Dataset id to use for inference.",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    experiment = load_experiment_config(args.experiment)
    contract = load_contract(experiment["dataset_id"])

    data_dir = contract.processed_path
    split_metadata = f"{data_dir}/{args.split}_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"
    if not os.path.exists(split_metadata):
        raise FileNotFoundError(
            f"{split_metadata} not found; run split_data.py --dataset-id "
            f"{contract.dataset_id} first"
        )

    print(f"=== {args.split}: {experiment['experiment_id']} ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Split: {split_metadata}")
    print(f"Device: {DEVICE}  TTA: {args.tta}  Threshold: {args.threshold}")
    if args.split == "test":
        print(
            "REMINDER: only run --split test ONCE, after all hyperparameter "
            "decisions are locked in."
        )

    dataset = DeforestationDataset(
        data_dir, split_metadata, norm_stats, contract, augment=False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Samples: {len(dataset)}  batches: {len(loader)}")

    sample_x, _ = dataset[0]
    in_ch = sample_x.shape[0]
    model = smp.Unet(encoder_name="resnet34", in_channels=in_ch, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    metrics = evaluate_test(model, loader, tta=args.tta, threshold=args.threshold)
    metrics.update({
        "split": args.split,
        "tta": args.tta,
        "threshold": args.threshold,
        "n_samples": len(dataset),
        "checkpoint": str(ckpt),
        "experiment_id": experiment["experiment_id"],
        "dataset_id": contract.dataset_id,
        "device": str(DEVICE),
        "batch_size": args.batch_size,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    })

    out_path = Path(args.out_json) if args.out_json else resolve_out_path(
        ckpt, experiment["experiment_id"], args.split, args.tta, args.threshold
    )

    if args.baseline:
        baseline = evaluate_baseline(dataset)
        baseline.update({
            "split": args.split,
            "n_samples": len(dataset),
            "experiment_id": experiment["experiment_id"],
            "dataset_id": contract.dataset_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })
        baseline_path = out_path.parent / f"{args.split}_baseline_metrics.json"
        
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        baseline_cm_path = baseline_path.parent / f"{baseline_path.stem}_confusion.png"                                                                          
        save_confusion_matrix(                                                                                                                                   
            baseline, baseline_cm_path,                                                                                                                          
            title=f"{args.split} baseline (dNBR<{baseline['dnbr_threshold']} & dNDVI<{baseline['dndvi_threshold']})",                                            
        )                                                                                                                                                        
        print(f"Baseline confusion matrix: {baseline_cm_path}")
        print(f"\nBaseline metrics (dNBR<{baseline['dnbr_threshold']} AND dNDVI<{baseline['dndvi_threshold']}):")
        print(f"  IoU:       {baseline['iou']:.4f}")
        print(f"  F1:        {baseline['f1']:.4f}")
        print(f"  Precision: {baseline['precision']:.4f}")
        print(f"  Recall:    {baseline['recall']:.4f}")
        print(f"\nBaseline results: {baseline_path}")
    
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    cm_path = out_path.parent / f"{out_path.stem}_confusion.png"                                                                                             
    save_confusion_matrix(                                                                                                                                   
        metrics, cm_path,                                                                                                                                    
        title=f"{args.split} — {experiment['experiment_id']}",                                                                                               
    )                                                                                                                                                        
    print(f"Confusion matrix: {cm_path}")

    print(f"\n{args.split} metrics{' (TTA)' if args.tta else ''}:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"\nResults: {out_path}")
