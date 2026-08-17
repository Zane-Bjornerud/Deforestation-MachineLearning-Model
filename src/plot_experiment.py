"""
Cross-run comparison plot for an experiment. Reads every run under
outputs/metrics/<experiment_id>/ and overlays their per-epoch val IoU,
running-best IoU, and val loss curves so you can see which config is
winning at each epoch without opening N separate curves.png files.

Standalone:
    python src/plot_experiment.py --experiment-dir outputs/metrics/gee_full_gfc_v1

Auto-invoked at the end of train.py so this always reflects the latest
completed run.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short_label(run_stamp, cfg):
    """Compact per-run legend label. Includes the values most likely to
    differ between experiments so runs are distinguishable at a glance."""
    parts = [run_stamp]
    if "scheduler" in cfg and cfg["scheduler"] not in (None, "none"):
        parts.append(cfg["scheduler"])
    if "alpha" in cfg:
        parts.append(f"a={cfg['alpha']}")
    parts.append(f"{cfg.get('epochs', '?')}ep")
    return " ".join(parts)


def _load_runs(experiment_dir):
    """Discover all runs (each subdir with metrics.jsonl + config.json)."""
    runs = []
    for sub in sorted(experiment_dir.iterdir()):
        if not sub.is_dir():
            continue
        mfile, cfile = sub / "metrics.jsonl", sub / "config.json"
        if not (mfile.exists() and cfile.exists()):
            continue
        with open(mfile) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        with open(cfile) as f:
            cfg = json.load(f)
        if rows:
            runs.append((sub.name, cfg, rows))
    return runs


def plot_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)
    runs = _load_runs(experiment_dir)
    if not runs:
        raise ValueError(f"no runs with metrics.jsonl found under {experiment_dir}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle(f"Runs comparison — {experiment_dir.name}", fontsize=14)

    # Panel 1: val IoU (raw, noisy) — shows late-epoch behavior including dips
    ax = axes[0]
    for name, cfg, rows in runs:
        ax.plot(
            [r["epoch"] for r in rows],
            [r["iou"] for r in rows],
            label=_short_label(name, cfg),
            alpha=0.7,
        )
    ax.set_title("Val IoU (per-epoch, raw)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: running-best IoU — monotone, much easier to compare peaks
    ax = axes[1]
    for name, cfg, rows in runs:
        best_so_far, m = [], 0.0
        for r in rows:
            m = max(m, r["iou"])
            best_so_far.append(m)
        ax.plot(
            [r["epoch"] for r in rows],
            best_so_far,
            label=_short_label(name, cfg),
            alpha=0.8,
        )
    ax.set_title("Running-best val IoU")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3: val loss — check for overfitting and compare loss surfaces
    ax = axes[2]
    for name, cfg, rows in runs:
        ax.plot(
            [r["epoch"] for r in rows],
            [r["val_loss"] for r in rows],
            label=_short_label(name, cfg),
            alpha=0.7,
        )
    ax.set_title("Val loss")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    out_path = experiment_dir / "comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison of runs")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to outputs/metrics/<experiment_id>/",
    )
    args = parser.parse_args()
    out = plot_experiment(Path(args.experiment_dir))
    print(f"Wrote {out}")
