"""
Sweep decision thresholds on a trained checkpoint to find the IoU-optimal
cut. train.py hardcodes `sigmoid(logits) > 0.5` for scoring; the actual
IoU-optimal threshold depends on how the model's probability distribution
lands under its training loss and class balance.

    python src/threshold_sweep.py --experiment gee_full_gfc_v1 \\
        --checkpoint /Volumes/LaCie/deforest_outputs/checkpoints/gee_full_gfc_v1/<stamp>/best_model.pth

Runs one forward pass per val chip and evaluates every threshold against
the same predictions (much cheaper than N passes). Writes
threshold_sweep.json into the run's metrics dir so the result stays paired
with the checkpoint that produced it.
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
from train import DEVICE, load_experiment_config


def sweep(model, loader, thresholds):
    """One forward pass per batch; update per-threshold counters. Avoids
    running the model N times when N thresholds are being compared."""
    stats = {t: {"inter": 0, "union": 0, "pred_pos": 0} for t in thresholds}
    actual_pos = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            prob = torch.sigmoid(model(xb))
            yb_bool = yb.bool()
            actual_pos += yb_bool.sum().item()
            for t in thresholds:
                pb = prob > t
                stats[t]["inter"] += (pb & yb_bool).sum().item()
                stats[t]["union"] += (pb | yb_bool).sum().item()
                stats[t]["pred_pos"] += pb.sum().item()

    results = []
    for t in thresholds:
        s = stats[t]
        inter, union, pp = s["inter"], s["union"], s["pred_pos"]
        results.append({
            "threshold": t,
            "iou": inter / max(1, union),
            "f1": 2 * inter / max(1, pp + actual_pos),
            "precision": inter / max(1, pp),
            "recall": inter / max(1, actual_pos),
            "pred_positive_pixels": pp,
        })
    return results, actual_pos


def resolve_out_path(ckpt: Path, experiment_id: str) -> Path:
    """Put threshold_sweep.json into the matching run metrics dir when
    possible, so it stays with the run's config.json + metrics.jsonl."""
    run_stamp = ckpt.parent.name
    default_dir = Path("outputs/metrics") / experiment_id / run_stamp
    if default_dir.exists():
        return default_dir / "threshold_sweep.json"
    return ckpt.parent / "threshold_sweep.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep decision thresholds on val to find the IoU-optimal cut."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated thresholds to evaluate. Default sweeps 0.30 to "
        "0.70 in 0.05 steps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    experiment = load_experiment_config(args.experiment)
    contract = load_contract(experiment["dataset_id"])
    data_dir = contract.processed_path
    val_metadata = f"{data_dir}/val_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"
    if not os.path.exists(val_metadata):
        raise FileNotFoundError(f"{val_metadata} not found")

    print(f"=== Threshold sweep: {experiment['experiment_id']} ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Val split: {val_metadata}")
    print(f"Device: {DEVICE}")
    print(f"Thresholds: {thresholds}")

    dataset = DeforestationDataset(
        data_dir, val_metadata, norm_stats, contract, augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Samples: {len(dataset)}  batches: {len(loader)}")

    sample_x, _ = dataset[0]
    in_ch = sample_x.shape[0]
    model = smp.Unet(encoder_name="resnet34", in_channels=in_ch, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    results, actual_pos = sweep(model, loader, thresholds)
    best = max(results, key=lambda r: r["iou"])

    payload = {
        "experiment_id": experiment["experiment_id"],
        "dataset_id": contract.dataset_id,
        "split": "val",
        "n_samples": len(dataset),
        "actual_positive_pixels": actual_pos,
        "checkpoint": str(ckpt),
        "device": str(DEVICE),
        "batch_size": args.batch_size,
        "thresholds": thresholds,
        "results": results,
        "best_threshold": best["threshold"],
        "best_iou": best["iou"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.out_json) if args.out_json else resolve_out_path(
        ckpt, experiment["experiment_id"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'thr':>5}  {'iou':>7}  {'f1':>7}  {'prec':>7}  {'recall':>7}  best?")
    for r in results:
        star = " <-- best" if r["threshold"] == best["threshold"] else ""
        print(
            f"{r['threshold']:>5.2f}  {r['iou']:>7.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>7.4f}  {r['recall']:>7.4f}{star}"
        )
    print(f"\nBest threshold: {best['threshold']:.2f}  IoU: {best['iou']:.4f}")
    print(f"Results: {out_path}")
