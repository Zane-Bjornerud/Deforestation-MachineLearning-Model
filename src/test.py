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

import os

# Set before torch loads OpenMP, matching train.py. Without this the script
# aborts with OMP Error #15 on Mac.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from dataset_contract import load_contract
from train import DEVICE, dice_loss, focal_loss, load_experiment_config


def evaluate_test(model, loader, tta=False, threshold=0.5):
    """Same aggregation as train.train_model's val loop, isolated so this
    file doesn't depend on the training loop internals."""
    inter = union = total_loss = pred_pos = actual_pos = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = focal_loss(logits, yb) + dice_loss(logits, yb)
            total_loss += loss.item()
            if tta:
                # Average sigmoid probs over identity + H-flip + V-flip + HV-flip
                prob = torch.sigmoid(logits)
                prob = prob + torch.sigmoid(model(xb.flip(-1))).flip(-1)
                prob = prob + torch.sigmoid(model(xb.flip(-2))).flip(-2)
                prob = prob + torch.sigmoid(model(xb.flip(-1, -2))).flip(-1, -2)
                pb = (prob / 4) > threshold
            else:
                pb = torch.sigmoid(logits) > threshold
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

def evaluate_baseline(dataset, dnbr_thresh=-0.1, dndvi_thresh=-0.15):
        dndvi_idx = CANONICAL_BAND_ORDER.index("dNDVI")
        dnbr_idx = CANONICAL_BAND_ORDER.index("dNBR")
        inter = union = pred_pos = actual_pos = 0
        for item in dataset.metadata:
            chip= np.load(f"{dataset.data_dir}/{item['chip_path']}")
            mask= np.load(f"{dataset.data_dir}/{item['mask_path']}")[0].astype(bool)
            pb = (chip[dnbr_idx] < dnbr_thresh) & (chip[dndvi_idx] < dndvi_thresh)
            inter += int((pb&mask).sum())
            union += int((pb | mask).sum())
            pred_pos += int(pb.sum())
            actual_pos += int(mask.sum())
        return {
            "iou":  inter / max(1, union),
            "f1":   2*inter / max(1, pred_pos + actual_pos),
            "precision": inter / max(1, pred_pos),
            "recall": inter / max(1, actual_pos),
            "n_samples": len(dataset.metadata),
            "dnbr_threshold": dnbr_thresh,
            "dndvi_threshold": dndvi_thresh,
        }


def resolve_out_path(ckpt: Path, experiment_id: str, split: str = "test", tta: bool = False, threshold: float = 0.5) -> Path:
    """If the checkpoint sits under a per-run stamp folder (the layout
    setup_run_dir writes into), put test_metrics.json into the matching
    outputs/metrics/<experiment_id>/<stamp>/ dir so the test number stays
    beside its training curve. Otherwise drop it next to the checkpoint.

    Non-default thresholds are encoded in the filename so a t=0.35 run
    doesn't clobber the canonical t=0.5 result."""
    suffix = ("_tta" if tta else "") + (f"_t{threshold:g}" if threshold != 0.5 else "")
    run_stamp = ckpt.parent.name
    default_dir = Path("outputs/metrics") / experiment_id / run_stamp
    if default_dir.exists():
        return default_dir / f"{split}_metrics{suffix}.json"
    return ckpt.parent / f"{split}_metrics{suffix}.json"


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
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to sigmoid probabilities. Default "
        "0.5 matches train.py's val scoring. Use the val-optimal cut from "
        "src/threshold_sweep.py when it materially beats 0.5 -- but only "
        "pick it from a val sweep, never by tuning against test.",
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{args.split} metrics{' (TTA)' if args.tta else ''}:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"\nResults: {out_path}")
