"""
Render per-epoch curves for a training run into <run_dir>/curves.png.

Standalone:
    python src/plot_run.py --run-dir outputs/metrics/gee_full_gfc_v1/<stamp>

Auto-invoked at the end of train.py so every run gets a plot without an
extra step. Reads metrics.jsonl (produced by train_model), plots one panel
each for train+val loss (overlaid to expose overfitting), IoU, F1,
precision, recall, and learning rate.

Losses are shown on the same axes because the gap between train and val
loss is more informative than either curve alone -- if val loss starts
climbing while train loss keeps falling, that's overfitting, and you only
see it by looking at them together.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless save; no display needed
import matplotlib.pyplot as plt


def _read_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} does not exist")
    with open(metrics_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{metrics_path} is empty; nothing to plot")
    return rows


def _read_run_stamp(run_dir: Path) -> str:
    """Prefer the run_stamp recorded in config.json, fall back to the dir
    name. Both should match for train.py-generated runs; backfill dirs may
    have a different config.run_stamp than folder name."""
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("run_stamp", run_dir.name)
    return run_dir.name


def plot_run(run_dir: Path, out_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    rows = _read_metrics(run_dir)
    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    stamp = _read_run_stamp(run_dir)
    fig.suptitle(f"Training curves — {stamp}", fontsize=14)

    # Panel 1: losses (overlaid)
    ax = axes[0][0]
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train", color="tab:blue")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="val", color="tab:orange")
    ax.set_title("Loss (focal + dice)")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: IoU with best-epoch marker
    ax = axes[0][1]
    ious = [r["iou"] for r in rows]
    ax.plot(epochs, ious, color="tab:green")
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["iou"])
    best = rows[best_idx]
    ax.scatter([best["epoch"]], [best["iou"]], color="red", zorder=5,
               label=f"best {best['iou']:.4f} @ ep {best['epoch']}")
    ax.set_title("IoU (val)")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: F1
    ax = axes[0][2]
    ax.plot(epochs, [r["f1"] for r in rows], color="tab:purple")
    ax.set_title("F1 (val)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)

    # Panel 4: Precision
    ax = axes[1][0]
    ax.plot(epochs, [r["precision"] for r in rows], color="tab:red")
    ax.set_title("Precision (val)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)

    # Panel 5: Recall
    ax = axes[1][1]
    ax.plot(epochs, [r["recall"] for r in rows], color="tab:brown")
    ax.set_title("Recall (val)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)

    # Panel 6: Learning rate (log y). learning_rate is null in old runs
    # (backfilled from stdout logs that didn't record it) -- show a note.
    ax = axes[1][2]
    lrs = [r.get("learning_rate") for r in rows]
    if any(lr is not None for lr in lrs):
        ax.plot(epochs, lrs, color="tab:gray")
        ax.set_yscale("log")
        ax.set_title("Learning rate")
    else:
        ax.text(0.5, 0.5, "learning_rate not recorded\n(pre-scheduler run)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Learning rate")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3, which="both")

    if out_path is None:
        out_path = run_dir / "curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves for a run")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to outputs/metrics/<experiment_id>/<run_stamp>/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override the default output path (<run_dir>/curves.png).",
    )
    args = parser.parse_args()

    out = plot_run(Path(args.run_dir), Path(args.out) if args.out else None)
    print(f"Wrote {out}")
