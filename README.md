![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)
![Field](https://img.shields.io/badge/Field-Remote%20Sensing-blue)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-8E44AD)

![Domain](https://img.shields.io/badge/Domain-Deforestation%20Detection-2E7D32)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Architecture](https://img.shields.io/badge/Model-U--Net%20%2B%20ResNet34-D84315)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=googleearth&logoColor=white)
![Sentinel-2](https://img.shields.io/badge/Imagery-Sentinel--2%20SR-0B5394)
![Labels](https://img.shields.io/badge/Labels-Hansen%20GFC%202025%20v1.13-1B5E20)
![AOI](https://img.shields.io/badge/AOI-Rond%C3%B4nia%2C%20Brazil-009739)
![Input](https://img.shields.io/badge/Input-18ch%20256%C3%97256-455A64)
![Conda](https://img.shields.io/badge/conda-44A833?logo=anaconda&logoColor=white)
![Hardware](https://img.shields.io/badge/Backend-Apple%20Silicon%20MPS-000000?logo=apple&logoColor=white)
![Code only](https://img.shields.io/badge/Repo-Code%20Only%20(no%20data%2Fweights)-lightgrey)

# Deforest

Deforestation detection over Rondônia, Brazil, from Sentinel-2 imagery. A U-Net (ResNet34 encoder) predicts pixel-level forest loss from an 18-channel pre/post dry-season composite, using annual loss labels from the Hansen Global Forest Change dataset.

This repo keeps **code only** — no data or trained weights are committed. See `docs/model-card.md` for full model details, intended use, and known limitations, and `docs/model-input-specification.md` for the exact tensor contract.

## Architecture

- U-Net, ResNet34 encoder, 18 input channels, 1 output channel (binary logits, sigmoid at inference).
- Input: pre/post Sentinel-2 dry-season composites (`COPERNICUS/S2_SR_HARMONIZED`) plus derived NDVI/NBR and their pre→post deltas — 18 channels total, channel-first `float32`, `256 x 256` tiles. Full channel table in `docs/model-input-specification.md`.
- Loss: Focal loss + Dice loss. Optimizer: AdamW.
- Full details, intended use, and limitations: `docs/model-card.md`.

## Repository layout

```
configs/
  datasets/       dataset contracts (label definition, raw/processed paths, GFC params)
  experiments/     training run configs (which dataset, checkpoint dir, hyperparameters)
scripts/
  gee_export_chips.py       submit a GEE batch export for a dataset contract
  gee_check_tasks.py        check status of submitted GEE export tasks
  setup_gdrive_rclone.sh    configure rclone + sync export shards from Google Drive
  build_download_manifest.py build download_manifest.tsv from downloaded shards; SHA-256s each file
  verify_gate_c.py          Gate C: export -> download chain is complete + consistent
  qc_report.py              automated QC report + visual panels for a processed dataset
  smoke_test_training.py    Gate A check: can training load the processed output?
  inspect_model_input.py    reconstruction/inspection tooling for a trained checkpoint
src/
  dataset_contract.py        dataset contract loading + validation
  GFC_process_tfrecords4.py  TFRecord -> chip/mask processor for Hansen-labeled datasets
  change_based_processor.py  TFRecord -> chip/mask processor for legacy change-based labels
  spatial_blocks.py          block-based split geometry for train/val/test (leakage-safe)
  split_data.py              train/val/test split from processed metadata
  dataset.py                 PyTorch Dataset over processed chips/masks
  train.py                   training loop, driven by an experiment config
  test.py                    evaluate a checkpoint on the held-out test split (run once)
  threshold_sweep.py         sweep decision thresholds on val for the IoU-optimal cut
  plot_run.py                render per-run training curves (curves.png)
  plot_experiment.py         cross-run comparison plot (comparison.png)
  band_names.py              single source of truth for the 18 canonical channel names
  gee_task_registry.py       persistent registry of submitted GEE export tasks
docs/
  model-card.md                   intended use, training data, limitations
  model-input-specification.md    full tensor/channel contract
data/                (git-ignored) raw TFRecords and processed .npy chips/masks
outputs/             (git-ignored) checkpoints, metrics, figures
```

## Setup

```bash
conda env create -f environment.yml
conda activate deforest
```

No CUDA assumed — this was built against an Apple Silicon Mac (torch uses MPS/CPU). `earthengine-api` is only needed for the export step (`scripts/gee_export_chips.py`); everything downstream just needs the rest of the env.

## Pipeline

Every dataset used anywhere below is defined by a contract at `configs/datasets/<dataset_id>.yaml` — the single source of truth for its label definition, raw/processed paths, and Hansen GFC parameters. Processors and the trainer validate their inputs against this contract and fail loudly on a mismatch rather than silently training on the wrong labels.

1. **Export chips from Earth Engine** (edit `DATASET_ID` in `scripts/gee_export_chips.py` to match a contract with `label_mode: hansen_loss`):
   ```bash
   python scripts/gee_export_chips.py
   python scripts/gee_check_tasks.py --dataset-id <dataset_id>   # poll until COMPLETED
   ```
   Lands sharded `.tfrecord` files in Google Drive.

   **Area of interest**: also set in `scripts/gee_export_chips.py`, via `AOI_CENTER_LON`/`AOI_CENTER_LAT`/`AOI_HALF_WIDTH_DEG` (a square AOI of side `2*AOI_HALF_WIDTH_DEG` degrees around that center). To resize, change `AOI_HALF_WIDTH_DEG` only — keeping the center fixed keeps you over the same validated Rondônia fishbone hotspot rather than drifting somewhere unchecked. Before committing to a new size, check its real Hansen GFC loss percentage rather than guessing — a much bigger box dilutes the positive rate (more intact/already-cleared land, less active frontier) and risks needing multiple GEE export tasks, which isn't currently supported. The in-script comment above `AOI_CENTER_LON` records the measured loss %/patch count/download size at a few sizes already checked this way.

2. **Pull the shards down** into `data/raw/<dataset_id>/` — either manually from Drive, or via:
   ```bash
   scripts/setup_gdrive_rclone.sh <dataset_id>
   ```
   Exports are gzip-compressed (`.tfrecord.gz`); decompress before processing:
   ```bash
   gunzip data/raw/<dataset_id>/*.tfrecord.gz
   ```

3. **Process TFRecords into chips/masks:**
   ```bash
   python src/GFC_process_tfrecords4.py --dataset-id <dataset_id>
   ```
   Writes `chips/*.npy`, `masks/*.npy`, `metadata.pkl`, and `normalization_stats.pkl` under `data/processed/<dataset_id>/`.

4. **QC the processed output:**
   ```bash
   python scripts/qc_report.py --dataset-id <dataset_id>
   ```
   Automated pass/fail on shape, channel order, mask dtype, duplicates, and NaN/Inf; renders visual inspection panels for the parts that need a human look (mask/imagery alignment).

5. **Split and smoke-test:**
   ```bash
   python src/split_data.py --dataset-id <dataset_id>
   python scripts/smoke_test_training.py --dataset-id <dataset_id>
   ```

6. **Train**, via an experiment config at `configs/experiments/<experiment_id>.yaml` (pins the dataset id, checkpoint dir, and hyperparameters):
   ```bash
   python src/train.py --experiment <experiment_id>
   ```
   Experiment configs support: `epochs`, `batch_size`, `learning_rate`, `num_workers`, `scheduler` (`cosine` or omit for constant LR), `eta_min` (cosine minimum LR), `alpha` and `gamma` (focal loss weighting). Each run writes to a **per-run stamped subdirectory** so successive runs never overwrite each other:
   ```
   outputs/metrics/<experiment_id>/<UTC stamp>/
     config.json      hyperparameters + git commit + start time
     metrics.jsonl    one JSON row per epoch (losses, IoU/F1/P/R, LR, epoch_seconds)
     train.log        full stdout capture
     curves.png       per-run curves (auto-rendered at end of run)
   outputs/metrics/<experiment_id>/comparison.png   cross-run overlay, refreshed after every run

   <checkpoint_dir>/<UTC stamp>/
     best_model.pth       best-val-IoU checkpoint from this run
     model_epoch_N.pth    every-10-epochs snapshot
   ```
   Metrics dir and checkpoint dir share the same `<UTC stamp>` so a run's curve and weights are trivially correlatable.

7. **Post-training analysis:**
   ```bash
   # Sweep decision thresholds on val to find the IoU-optimal cut (~10 min, no retraining)
   python src/threshold_sweep.py --experiment <experiment_id> --checkpoint <run>/best_model.pth

   # Regenerate a run's curves.png (auto-produced by train.py; also runnable standalone)
   python src/plot_run.py --run-dir outputs/metrics/<experiment_id>/<stamp>

   # Refresh the experiment-wide comparison.png overlay
   python src/plot_experiment.py --experiment-dir outputs/metrics/<experiment_id>

   # Final test-set evaluation — run ONCE, after all hyperparameter tuning is locked in
   python src/test.py --experiment <experiment_id> --checkpoint <run>/best_model.pth
   ```
   `src/test.py` deliberately isolates the test-split evaluation from val-time scoring; every look at the test set contaminates it as a generalization estimator, so it's a separate script.

## Data layout

- Dataset files live under `data/raw/` and `data/processed/` (git-ignored).
- Generic manifest-based downloads (S3/HF/HTTP) are supported via `data/manifest.tsv` + `scripts/download_data.sh`; Google Earth Engine exports use the Drive-based flow above instead.

## Status

- **Phase 0 (pipeline audit) — done.** Traced the original training pipeline end to end and documented it (`docs/model-input-specification.md`, `docs/model-card.md`). Found that the initial model was trained on change-index-derived labels (dNBR/dNDVI thresholds), not true annual forest-loss labels.
- **Phase 1 (Hansen GFC retrain) — baseline established.** Replaced the threshold-derived labels with the Hansen Global Forest Change dataset (`UMD/hansen/global_forest_change_2025_v1_13`, `treecover2000` + `lossyear`), which gives a real annual loss signal instead of a heuristic change threshold. Each dataset used anywhere in the pipeline is pinned to an explicit, versioned contract under `configs/datasets/`:
  - `existing_gfc_recovery_v0` — reprocessed legacy export used to shake out pipeline bugs. Not a results baseline.
  - `gee_canary_gfc_v1` — small fresh GEE export used to validate export → download → process → split → train end to end. Passed all gates.
  - `gee_full_gfc_v1` — the Phase 1 baseline. Exported, processed, and trained; Gate C (download integrity) verified. **Current best val IoU 0.5327** (60 epochs, cosine LR schedule, focal α=0.5 + γ=2.0, threshold 0.60). Held-out test set has not been touched.
  - `legacy_threshold_v1` — the original change-based labels, kept only as a comparison point against the Hansen-based results.
- **Phase 1 tuning experiments so far.** All runs on `gee_full_gfc_v1`, MPS-accelerated on Apple Silicon:
  - Enabled MPS by replacing `smp.losses.FocalLoss` with a local focal implementation that avoids the MPS-incompatible `.type()` call. ~30× speedup vs. CPU.
  - Cosine LR schedule vs. constant LR: comparable peak IoU, ~3× tighter late-stage IoU variance (much more trustworthy `best_model.pth`).
  - Focal alpha rebalance 0.75 → 0.5: **+0.014 IoU**. Diagnostic: threshold sweep on the α=0.75 checkpoint peaked at 0.70 (evidence the model was over-predicting positives); α=0.5 model is well-calibrated at ~0.55–0.60.
  - 30 → 60 epochs at α=0.5: +0.008 IoU (diminishing returns; model plateaued by epoch 49).

Next up: **positive-oversampled `WeightedRandomSampler`** — the highest expected-value remaining experiment (segmentation literature suggests +0.02 to +0.05 IoU on class-imbalanced tasks). Then final test-set evaluation via `src/test.py`.
