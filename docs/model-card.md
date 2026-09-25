# Model Card: UNet Rondônia v1

## Model summary

UNet Rondônia v1 is a binary semantic segmentation model for deforestation detection in Sentinel-2 imagery over Rondônia, Brazil. It uses a U-Net with a ResNet34 encoder and 18 input channels derived from pre/post-season optical bands and change indices. The current pipeline primarily uses Hansen Global Forest Change (GFC) loss labels (see `configs/datasets/gee_full_gfc_v1.yaml`); a legacy change-based labeling path (dNBR/dNDVI thresholding) is also supported via `legacy_threshold_v1`.

## Intended use

This model is intended for exploratory deforestation mapping and local analysis over the Rondônia AOI used during training. It is best suited for identifying pixels that exhibit a strong change signal between the pre and post compositing windows.

## Unsupported use

This model should not be used as a standalone operational monitoring system without further validation. It is not designed for:
- use outside the training geography without retraining or revalidation,
- direct interpretation as a legal or regulatory deforestation determination,
- high-stakes decision making without human review,
- scenes with materially different sensor characteristics, seasons, or land-cover distributions.

## Architecture

- Architecture: U-Net
- Encoder: ResNet34
- Input channels: 18
- Output channels: 1
- Output activation: sigmoid at inference time

## Training data

Training data are derived from Sentinel-2 surface reflectance composites exported from Google Earth Engine and stored as TFRecord shards, then processed into 256 x 256 numpy chips.

Current export and processing settings:
- Source imagery: COPERNICUS/S2_SR_HARMONIZED
- Cloud probability mask: COPERNICUS/S2_CLOUD_PROBABILITY
- Export CRS: EPSG:32720
- Export resolution: 10 m
- Patch size: 256 x 256
- Training region: Rondônia, Brazil
- Temporal windows:
  - Pre: 2020-06-01 to 2020-09-30
  - Post: 2021-06-01 to 2021-09-30

Labels come from one of two sources, chosen per dataset contract:
- Hansen (current, `label_mode: hansen_loss`): positive pixels are those with `treecover2000 ≥ forest_cover_threshold` AND recorded forest loss in the dataset's `target_year` (see `configs/datasets/gee_full_gfc_v1.yaml`).
- Change-based (legacy, `label_mode: change_based`): positive pixels are those satisfying both dNBR and dNDVI disturbance thresholds. Retained only for comparison against the Hansen pipeline.

## Input data

Model inputs are channel-first float32 tensors with shape 18 x 256 x 256.

Channel order:
1. B2_pre
2. B3_pre
3. B4_pre
4. B8_pre
5. B11_pre
6. B12_pre
7. NDVI_pre
8. NBR_pre
9. B2_post
10. B3_post
11. B4_post
12. B8_post
13. B11_post
14. B12_post
15. NDVI_post
16. NBR_post
17. dNDVI
18. dNBR

Normalization is per-channel z-score normalization using statistics computed from the training split only, then applied to val and test.

## Prediction target

The prediction target is binary deforestation segmentation.

- Positive-pixel definition:
  - Hansen (current): `treecover2000 ≥ forest_cover_threshold` AND Hansen `lossyear` matches the dataset's `target_year`.
  - Change-based (legacy): pixel satisfies both dNBR and dNDVI disturbance thresholds.
- Negative-pixel definition: all remaining pixels.
- Ignore-pixel definition: none is explicitly defined in the current pipeline.
- Label source: `hansen_gfc` for the active datasets; `change_index_threshold` for `legacy_threshold_v1`.
- Label version: pinned per dataset contract in `configs/datasets/<id>.yaml` (Hansen asset id + forest-cover threshold, or the `sensitive` threshold profile for change-based).
- Label time period: comparison between the pre and post compositing windows.

## Training procedure

Training uses the processed metadata splits produced by `src/split_data.py` and the dataset loader in `src/dataset.py`.

Current procedure:
1. Load processed chip and mask metadata.
2. Split metadata into train/validation/test as contiguous geographic bands of spatial blocks (`create_block_splits` in `src/split_data.py`) when the metadata carries `block_id`; fall back to a class-stratified random chip split otherwise.
3. Compute per-channel normalization statistics from the training split only.
4. Load chips as float32 tensors.
5. Normalize each channel using the precomputed mean/std and apply the same stats to train, val, and test.
6. Train a U-Net with focal loss plus Dice loss.
7. Save best checkpoint to `<experiment.checkpoint_dir>/<run_stamp>/best_model.pth` (per-experiment, per-run — e.g. `outputs/checkpoints/gee_full_gfc_v1/<stamp>/best_model.pth`, or the external-drive path pinned in that experiment's yaml).

Known properties of the split:
- Train, val, and test are contiguous geographic bands of spatial blocks along one grid axis, with a buffer strip of blocks dropped at each band boundary. Chips from the same immediate area cannot cross splits.
- Because the bands are contiguous rather than scattered, each split's composition (positive rate, landscape mix) depends on where deforestation activity falls within the AOI. `create_block_splits` prints and sanity-checks the per-split composition at split time to catch degenerate bands.
- Normalization statistics come from the training split only; val and test are z-scored using stats they did not contribute to.

## Evaluation status

Training reports per-epoch validation IoU, F1, precision, and recall, and keeps the checkpoint with the best validation IoU. Held-out test scoring runs separately via `src/test.py`, which additionally reports the full 2x2 confusion matrix (TP/FP/FN/TN plus specificity) and can score a dNBR/dNDVI threshold-rule baseline against the same split for comparison.

Because train/val/test are contiguous geographic bands, reported metrics reflect generalization to unseen terrain rather than interpolation within already-seen scenes. The tradeoff is that each split's composition depends on where deforestation activity falls within the AOI, so a single split's metrics can differ from what a scattered random split would produce.

## Geographic scope

The current model is trained on a Rondônia, Brazil AOI defined in the export script. It should be treated as region-specific unless retrained and revalidated on broader coverage.

## Temporal scope

The model is trained on a fixed dry-season comparison:
- Pre window: 2020-06-01 to 2020-09-30
- Post window: 2021-06-01 to 2021-09-30

The temporal design assumes that the change signal is meaningfully captured by this pre/post pairing.

## Known limitations

- Split composition depends on where deforestation activity falls within the AOI, since the bands are contiguous rather than scattered. `create_block_splits` sanity-checks this at split time.
- Channel order depends on stored metadata and should be verified against `band_names.py`.
- Hansen labels inherit Hansen GFC's own limitations (annual granularity, sub-canopy or partial-canopy loss missed). Change-based (legacy) labels are threshold-derived and may miss weak or ambiguous signals.
- The model has not been shown to generalize beyond the Rondônia training region.

## Ethical and operational considerations

This model should be used as an assistive tool, not as an automated decision authority. False positives and false negatives can both have operational consequences. Users should validate predictions against expert review and local context before acting on results.

If deployed outside the original training area or time period, the model should be revalidated and likely retrained. Any operational system should include uncertainty handling and clear escalation paths for ambiguous predictions.

## Version history

- v1: UNet Rondônia trained on 18-channel Sentinel-2 pre/post composites with Hansen GFC loss labels, block-level spatial split, and train-only normalization stats.