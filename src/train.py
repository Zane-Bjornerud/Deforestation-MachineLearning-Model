import os

# Must be set before torch (which loads OpenMP) is imported, otherwise the
# duplicate-libomp initialization aborts the process with OMP Error #15 on
# this Mac. Setdefault so callers can still override from the shell.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import atexit
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import yaml
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from dataset_contract import load_contract

EXPERIMENT_DIR = "configs/experiments"
METRICS_ROOT = "outputs/metrics"


def load_experiment_config(experiment_id_or_path):
    """Load an experiment config: which dataset contract to train on, where
    to write checkpoints, and hyperparameters. Keeping this separate from the
    dataset contract is what lets hansen_loss and change_based be run and
    compared as two distinct experiments against the same code."""
    if os.path.sep in experiment_id_or_path or experiment_id_or_path.endswith(
        (".yaml", ".yml")
    ):
        path = experiment_id_or_path
    else:
        path = os.path.join(EXPERIMENT_DIR, f"{experiment_id_or_path}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No experiment config found at {path}. Add one under "
            f"{EXPERIMENT_DIR}/ (see hansen_loss_v1.yaml / change_based_v1.yaml)."
        )

    with open(path) as f:
        return yaml.safe_load(f)


def _git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        return commit, dirty
    except Exception:
        return None, None


def setup_run_dir(experiment, contract, device_str, actual_channels):
    """Create per-run dirs for this training run: a metrics dir under
    outputs/metrics/<experiment_id>/<UTC timestamp>/ and a checkpoint dir
    under <checkpoint_dir>/<same UTC timestamp>/. Both share the stamp so
    metrics and weights for one run stay correlated, and neither overwrites
    prior runs. Drops config.json in the metrics dir. Returns
    (metrics_dir, run_checkpoint_dir)."""
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    metrics_dir = os.path.join(METRICS_ROOT, experiment["experiment_id"], stamp)
    os.makedirs(metrics_dir, exist_ok=True)
    run_checkpoint_dir = os.path.join(experiment["checkpoint_dir"], stamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    commit, dirty = _git_info()
    config_snapshot = {
        "experiment_id": experiment["experiment_id"],
        "dataset_id": contract.dataset_id,
        "label_mode": contract.label_mode,
        "processed_path": str(contract.processed_path),
        "checkpoint_dir": experiment["checkpoint_dir"],
        "run_checkpoint_dir": run_checkpoint_dir,
        "run_stamp": stamp,
        "epochs": experiment.get("epochs", 20),
        "batch_size": experiment.get("batch_size", 2),
        "learning_rate": experiment.get("learning_rate", 1e-4),
        "scheduler": experiment.get("scheduler"),
        "eta_min": experiment.get("eta_min", 1e-6),
        "num_workers": experiment.get("num_workers", 0),
        "device": device_str,
        "input_channels": actual_channels,
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at": started_at.isoformat(),
    }
    with open(os.path.join(metrics_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)
    return metrics_dir, run_checkpoint_dir


class _Tee:
    """Mirror writes to multiple streams. Used to send stdout to both the
    terminal and the run's train.log without needing an external `| tee`."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _tee_stdout_to(log_path):
    """Duplicate stdout+stderr to log_path so `python src/train.py ...` on
    its own captures the same output that piping through `| tee` used to."""
    f = open(log_path, "w")
    atexit.register(f.close)
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# Model configuration
IN_CH = 18  # data has 18 bands
OUT_CH = 1


# Local focal loss impl. smp.losses.FocalLoss calls target.type(output.type())
# internally, which raises "invalid type: 'torch.mps.FloatTensor'" on MPS
# because .type() only understands CPU/CUDA type strings. Everything else in
# the smp focal path is MPS-safe, so we reproduce it here with .to(dtype).
def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    target = target.to(logits.dtype)
    logpt = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-logpt)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * logpt
    loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.mean()


bce = focal_loss


def dice_loss(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3))
    den = (p + y).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def build_scheduler(optimizer, experiment):
    """Build an LR scheduler from the experiment config, or return None if
    none is requested (preserves the constant-LR behavior of older runs).

    Currently supported:
        scheduler: cosine   -> CosineAnnealingLR(T_max=epochs, eta_min=eta_min)

    Cosine annealing was picked as the default because it's a deterministic,
    smooth schedule that doesn't react to noisy val metrics (unlike
    ReduceLROnPlateau) and doesn't introduce discontinuities (unlike StepLR).
    T_max matches the run's epoch budget so LR completes its decay exactly at
    the last epoch; eta_min=1e-6 leaves the model at 1% of initial LR so the
    final epochs still fine-tune rather than freezing outright."""
    kind = experiment.get("scheduler")
    if kind in (None, "none", "None", ""):
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=experiment.get("epochs", 20),
            eta_min=experiment.get("eta_min", 1e-6),
        )
    raise ValueError(f"Unknown scheduler {kind!r}. Supported: 'cosine' or omit.")


# Training loop
def train_model(
    train_loader,
    val_loader,
    checkpoint_dir,
    epochs=50,
    metrics_path=None,
    scheduler=None,
):
    best_iou = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        # Capture the LR used for this epoch's steps BEFORE any post-epoch
        # scheduler.step(). Logged in the JSONL row so metrics.jsonl records
        # exactly what LR each epoch's weights were trained with.
        lr_this_epoch = opt.param_groups[0]["lr"]
        model.train()
        train_loss = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            try:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                if epoch == 0 and batch_idx == 0:
                    print(type(xb))
                    print("Image shape:", xb.shape)
                    print("Image dtype:", xb.dtype)
                    print("Image range:", xb.min().item(), xb.max().item())

                    print("Mask shape:", yb.shape)
                    print("Mask dtype:", yb.dtype)
                    print("Mask values:", torch.unique(yb))

                logits = model(xb)
                loss = bce(logits, yb) + dice_loss(logits, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Validation
        model.eval()
        inter = union = val_loss = pred_pos = actual_pos = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                try:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = bce(logits, yb) + dice_loss(logits, yb)
                    val_loss += loss.item()

                    pb = torch.sigmoid(logits) > 0.5
                    yb_bool = yb.bool()
                    inter += (pb & yb_bool).sum().item()
                    union += (pb | yb_bool).sum().item()
                    pred_pos += pb.sum().item()
                    actual_pos += yb_bool.sum().item()
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        iou = inter / max(1, union)
        f1 = 2 * inter / max(1, pred_pos + actual_pos)
        precision = inter / max(1, pred_pos)
        recall = inter / max(1, actual_pos)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, IoU={iou:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}"
        )

        # Save best model
        is_new_best = iou > best_iou
        if is_new_best:
            best_iou = iou
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print(f"  New best IoU: {best_iou:.3f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch}.pth"
            )

        # Per-epoch JSONL row. Written after checkpointing so a mid-epoch
        # crash never leaves a row claiming a checkpoint that isn't on disk.
        if metrics_path is not None:
            row = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "iou": iou,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "learning_rate": lr_this_epoch,
                "epoch_seconds": time.time() - epoch_start,
                "is_new_best": is_new_best,
                "best_iou_so_far": best_iou,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"  WARNING: failed to append metrics row: {e}")

        # Step the scheduler after the epoch has been fully scored + logged,
        # so the LR that trained this epoch is the one recorded above.
        if scheduler is not None:
            scheduler.step()

    print(f"Training complete! Best IoU: {best_iou:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the deforestation model on a specific experiment"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment id under configs/experiments/ (e.g. hansen_loss_v1 "
        "or change_based_v1), or an explicit path to an experiment yaml. "
        "Each experiment pins a dataset contract, checkpoint dir, and "
        "hyperparameters, so hansen_loss and change_based runs stay "
        "separate and comparable.",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(args.experiment)
    contract = load_contract(experiment["dataset_id"])

    print(f"=== Experiment: {experiment['experiment_id']} ===")
    print(f"Dataset: {contract.dataset_id} (label_mode={contract.label_mode})")
    print(f"Processed data: {contract.processed_path}")

    # Create output directories
    checkpoint_dir = experiment["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Paths
    data_dir = contract.processed_path
    train_metadata = f"{data_dir}/train_metadata.pkl"
    val_metadata = f"{data_dir}/val_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"

    # Check if files exist
    if not os.path.exists(train_metadata):
        print(f"Train metadata not found: {train_metadata}")
        print(
            f"Please run: python src/split_data.py --dataset-id {contract.dataset_id} first"
        )
        exit(1)

    # Create datasets
    print("Loading datasets...")
    train_dataset = DeforestationDataset(
        data_dir, train_metadata, norm_stats, contract, augment=True
    )
    val_dataset = DeforestationDataset(
        data_dir, val_metadata, norm_stats, contract, augment=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("No training samples found!")
        exit(1)

    # Check actual input channels from data
    sample_x, sample_y = train_dataset[0]
    actual_channels = sample_x.shape[0]
    print(f"Actual input channels: {actual_channels}")
    print(f"Sample chip shape: {sample_x.shape}")
    print(f"Sample mask shape: {sample_y.shape}")

    # Update model if needed
    if actual_channels != IN_CH:
        print(f"Updating model input channels from {IN_CH} to {actual_channels}")
        IN_CH = actual_channels

    # Create model and optimizer
    model = smp.Unet(encoder_name="resnet34", in_channels=IN_CH, classes=OUT_CH).to(
        DEVICE
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=experiment.get("learning_rate", 1e-4)
    )
    scheduler = build_scheduler(opt, experiment)
    print(f"Scheduler: {experiment.get('scheduler') or 'none'}")

    # Create data loaders with smaller batch size due to 256x256 images
    batch_size = experiment.get("batch_size", 2)
    # num_workers=0 (synchronous, single-process loading) was the long-standing
    # default here -- safe on Mac but means zero overlap between disk I/O and
    # model compute, which matters a lot once processed_path lives on a slow
    # external drive. Configurable per-experiment (not hardcoded) so it can be
    # tested/tuned without another code change, and easily reverted to 0 if a
    # given machine hits Mac multiprocessing DataLoader issues.
    num_workers = experiment.get("num_workers", 0)
    # persistent_workers keeps worker subprocesses alive across epochs instead
    # of respawning them every epoch. Without it, each new worker (macOS uses
    # the "spawn" start method) has to fully re-import this module -- re-running
    # every module-level statement, including re-importing torch/smp from
    # scratch -- once per worker, per epoch. Only valid when num_workers > 0.
    persistent_workers = num_workers > 0
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # Test loading one batch
        print("Testing data loading...")
        test_batch = next(iter(train_loader))
        print(
            f"Batch loaded successfully: {test_batch[0].shape}, {test_batch[1].shape}"
        )

        # Per-run dirs (metrics + checkpoints share the same UTC stamp) so
        # successive runs stay comparable and don't overwrite each other.
        metrics_dir, run_checkpoint_dir = setup_run_dir(
            experiment, contract, str(DEVICE), actual_channels
        )
        metrics_path = os.path.join(metrics_dir, "metrics.jsonl")
        log_path = os.path.join(metrics_dir, "train.log")
        _tee_stdout_to(log_path)
        print(f"Run metrics: {metrics_dir}")
        print(f"Run checkpoints: {run_checkpoint_dir}")
        print(f"Run log: {log_path}")

        # Train model
        print("Starting training...")
        train_model(
            train_loader,
            val_loader,
            run_checkpoint_dir,
            epochs=experiment.get("epochs", 20),
            metrics_path=metrics_path,
            scheduler=scheduler,
        )

        # Auto-render training curves so every completed run has a plot
        # ready to look at. Failure here should not fail the run itself
        # (metrics + checkpoints are already on disk).
        try:
            from plot_run import plot_run

            out = plot_run(metrics_dir)
            print(f"Wrote {out}")
        except Exception as e:
            print(f"WARNING: could not render curves.png: {e}")

    except Exception as e:
        print(f"Error during training setup: {e}")
        import traceback

        traceback.print_exc()
