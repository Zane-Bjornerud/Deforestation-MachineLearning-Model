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
  gee_export_chips.py      submit a GEE batch export for a dataset contract
  gee_check_tasks.py       check status of submitted GEE export tasks
  setup_gdrive_rclone.sh   configure rclone + sync export shards from Google Drive
  qc_report.py             automated QC report + visual panels for a processed dataset
  smoke_test_training.py   Gate A check: can training load the processed output?
  inspect_model_input.py   reconstruction/inspection tooling for a trained checkpoint
src/
  dataset_contract.py       dataset contract loading + validation
  GFC_process_tfrecords4.py TFRecord -> chip/mask processor for Hansen-labeled datasets
  change_based_processor.py TFRecord -> chip/mask processor for legacy change-based labels
  split_data.py              train/val/test split from processed metadata
  dataset.py                 PyTorch Dataset over processed chips/masks
  train.py                   training loop, driven by an experiment config
  band_names.py              single source of truth for the 18 canonical channel names
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

1. **Export chips from Earth Engine** (edit `DATASET_ID` in the script to match a contract with `label_mode: hansen_loss`):
   ```bash
   python scripts/gee_export_chips.py
   python scripts/gee_check_tasks.py   # poll until COMPLETED
   ```
   Lands sharded `.tfrecord` files in Google Drive.

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

## Data layout

- Dataset files live under `data/raw/` and `data/processed/` (git-ignored).
- Generic manifest-based downloads (S3/HF/HTTP) are supported via `data/manifest.tsv` + `scripts/download_data.sh`; Google Earth Engine exports use the Drive-based flow above instead.

## Status

- **Phase 0 (pipeline audit) — done.** Traced the original training pipeline end to end and documented it (`docs/model-input-specification.md`, `docs/model-card.md`). Found that the initial model was trained on change-index-derived labels (dNBR/dNDVI thresholds), not true annual forest-loss labels.
- **Phase 1 (Hansen GFC retrain) — in progress.** Replacing the threshold-derived labels with the Hansen Global Forest Change dataset (`UMD/hansen/global_forest_change_2025_v1_13`, `treecover2000` + `lossyear`), which gives a real annual loss signal instead of a heuristic change threshold. Each dataset used anywhere in the pipeline is pinned to an explicit, versioned contract under `configs/datasets/`:
  - `existing_gfc_recovery_v0` — reprocessing an already-downloaded legacy export to shake out pipeline bugs. Not a results baseline.
  - `gee_canary_gfc_v1` — small fresh GEE export used to validate export → download → process → split → train end to end before spending quota on the full run. Currently working through this: export, processing, QC, and the split/smoke-test gates have passed; one-epoch training is the remaining canary test.
  - `gee_full_gfc_v1` — the actual Phase 1 baseline, to be exported and trained once the canary passes cleanly (Gate B).
  - `legacy_threshold_v1` — the original change-based labels, kept only as a comparison point against the Hansen-based results.

Next up: finish the canary's one-epoch training test, then launch the `gee_full_gfc_v1` export and train the Phase 1 baseline for comparison against the legacy change-based model.
