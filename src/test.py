"""
Evaluate a trained checkpoint against the held-out test split.

    python src/test.py --experiment gee_full_gfc_v1 \\
        --checkpoint /Volumes/LaCie/deforest_outputs/checkpoints/gee_full_gfc_v1/<stamp>/best_model.pth

Loads test_metadata.pkl for the experiment's dataset contract and runs the
same forward pass + 0.5 threshold + IoU/F1/Precision/Recall math that
train.py's val loop uses. Writes test_metrics.json into the run's metrics
dir (outputs/metrics/<experiment_id>/<stamp>/) when the checkpoint lives
under a per-run stamp folder, so the test number stays paired with the
training run that produced it.

Test-set contract: only run this ONCE, after all hyperparameter decisions
are locked in. Every look at the test number during tuning contaminates it
and degrades its ability to estimate real-world performance. Use the val
metrics already logged during training for tuning decisions.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from dataset_contract import load_contract
from train import DEVICE, dice_loss, focal_loss, load_experiment_config


def evaluate_test(model, loader):
    """Same aggregation as train.train_model's val loop, isolated so this
    file doesn't depend on the training loop internals."""
    inter = union = total_loss = pred_pos = actual_pos = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = focal_loss(logits, yb) + dice_loss(logits, yb)
            total_loss += loss.item()
            pb = torch.sigmoid(logits) > 0.5
            yb_bool = yb.bool()
            inter += (pb & yb_bool).sum().item()
            union += (pb | yb_bool).sum().item()
            pred_pos += pb.sum().item()
            actual_pos += yb_bool.sum().item()

    return {
        "loss": total_loss / max(1, len(loader)),
        "iou": inter / max(1, union),
        "f1": 2 * inter / max(1, pred_pos + actual_pos),
        "precision": inter / max(1, pred_pos),
        "recall": inter / max(1, actual_pos),
        "n_batches": len(loader),
    }


def resolve_out_path(ckpt: Path, experiment_id: str) -> Path:
    """If the checkpoint sits under a per-run stamp folder (the layout
    setup_run_dir writes into), put test_metrics.json into the matching
    outputs/metrics/<experiment_id>/<stamp>/ dir so the test number stays
    beside its training curve. Otherwise drop it next to the checkpoint."""
    run_stamp = ckpt.parent.name
    default_dir = Path("outputs/metrics") / experiment_id / run_stamp
    if default_dir.exists():
        return default_dir / "test_metrics.json"
    return ckpt.parent / "test_metrics.json"


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
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    experiment = load_experiment_config(args.experiment)
    contract = load_contract(experiment["dataset_id"])

    data_dir = contract.processed_path
    test_metadata = f"{data_dir}/test_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"
    if not os.path.exists(test_metadata):
        raise FileNotFoundError(
            f"{test_metadata} not found; run split_data.py --dataset-id "
            f"{contract.dataset_id} first"
        )

    print(f"=== Test: {experiment['experiment_id']} ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Test split: {test_metadata}")
    print(f"Device: {DEVICE}")
    print(
        "REMINDER: this should only be run ONCE, after all hyperparameter "
        "decisions are locked in. Re-running with different checkpoints "
        "contaminates the test set."
    )

    dataset = DeforestationDataset(
        data_dir, test_metadata, norm_stats, contract, augment=False
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

    metrics = evaluate_test(model, loader)
    metrics.update({
        "split": "test",
        "n_samples": len(dataset),
        "checkpoint": str(ckpt),
        "experiment_id": experiment["experiment_id"],
        "dataset_id": contract.dataset_id,
        "device": str(DEVICE),
        "batch_size": args.batch_size,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    })

    out_path = Path(args.out_json) if args.out_json else resolve_out_path(
        ckpt, experiment["experiment_id"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest metrics:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"\nResults: {out_path}")
