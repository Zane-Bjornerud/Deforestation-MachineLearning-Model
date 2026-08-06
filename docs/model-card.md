# Model Card: UNet Rondônia v1

## Model summary

UNet Rondônia v1 is a binary semantic segmentation model for deforestation detection in Sentinel-2 imagery over Rondônia, Brazil. It uses a U-Net with a ResNet34 encoder and 18 input channels derived from pre/post-season optical bands and change indices. The current training pipeline uses threshold-based change labels built from dNBR and dNDVI.

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

The current labels are derived from change indices, not directly from the original label band. Positive pixels are defined by thresholding both dNBR and dNDVI.

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

Normalization is per-channel z-score normalization using statistics computed from the processed dataset.

## Prediction target

The prediction target is binary deforestation segmentation.

- Positive-pixel definition: pixel satisfies both dNBR and dNDVI disturbance thresholds.
- Negative-pixel definition: all remaining pixels.
- Ignore-pixel definition: none is explicitly defined in the current pipeline.
- Label source: derived change-based labels from processed channels.
- Label version: sensitive threshold profile in the current processor.
- Label time period: comparison between the pre and post compositing windows.

## Training procedure

Training uses the processed metadata splits produced by `src/split_data.py` and the dataset loader in `src/dataset.py`.

Current procedure:
1. Load processed chip and mask metadata.
2. Split metadata into train/validation/test using class-stratified random splitting.
3. Compute normalization statistics from the full processed dataset before splitting.
4. Load chips as float32 tensors.
5. Normalize each channel using precomputed mean and standard deviation.
6. Train a U-Net with focal loss plus Dice loss.
7. Save best checkpoint to `outputs/checkpoints/best_model.pth`.

Known properties of the split (as used for this checkpoint):
- The split is stratified only by deforestation presence.
- Spatial grouping is not used.
- Scene grouping is not used.
- Geographic independence is not guaranteed.
- Normalization is computed before splitting, so the split is not strictly leakage-free.

This applies to the v1 checkpoint above, which predates block-level spatial
splitting. `src/split_data.py` now also supports a block-level spatial split
(`create_block_splits`, see `src/spatial_blocks.py`) for datasets processed
with a GEE `mixer.json` sidecar present -- chips are grouped into spatial
blocks and whole contiguous bands of blocks (not individual chips) are
assigned to train/val/test, with a buffer strip dropped at each band
boundary. `src/split_data.py --dataset-id <id>` picks this automatically
whenever the processed metadata carries `block_id`. It has not yet been used
for a trained checkpoint -- `gee_full_gfc_v1` (Phase 1's actual baseline,
not yet exported) is the first dataset expected to use it.

## Evaluation status

Formal independent evaluation has not yet been established in the current repository. The training script reports validation IoU, F1, precision, and recall during training, and saves the best checkpoint based on validation IoU.

Current evaluation caveats (for this checkpoint):
- Validation and test data may not be geographically independent.
- Neighboring chips and chips from the same scene may cross splits.
- Reported metrics should be interpreted as random-chip split performance, not as fully independent geographic generalization.

A future checkpoint trained on a block-split dataset (see above) will not
carry this caveat in the same form -- its val/test bands are contiguous
geographic regions disjoint from train, so its metrics should reflect
generalization to unseen terrain rather than interpolation within
already-seen scenes. That checkpoint's own caveat will instead be that
train/val/test are large contiguous regions rather than a representative
random sample, so any one split's composition (positive rate, landscape mix)
can differ from the others depending on where deforestation activity falls
within the AOI -- this is checked and printed by `create_block_splits` at
split time.

## Geographic scope

The current model is trained on a Rondônia, Brazil AOI defined in the export script. It should be treated as region-specific unless retrained and revalidated on broader coverage.

## Temporal scope

The model is trained on a fixed dry-season comparison:
- Pre window: 2020-06-01 to 2020-09-30
- Post window: 2021-06-01 to 2021-09-30

The temporal design assumes that the change signal is meaningfully captured by this pre/post pairing.

## Known limitations

- This v1 checkpoint has split leakage risk because its splitting was random and class-stratified only. `src/split_data.py` now also supports a block-level spatial split for GFC-labeled datasets with a `mixer.json` sidecar (see "Training procedure" above); this checkpoint predates that and was not retrained with it.
- Spatial and scene-level grouping were not enforced for this checkpoint.
- Normalization statistics are computed before splitting (still true for both split strategies -- normalization stats are computed from the full processed dataset in `GFC_process_tfrecords4.py`/legacy processors, not per-split. This is a separate leakage source from the train/val/test split itself and neither split strategy addresses it).
- Per-chip georeferencing metadata (tile row/col, centroid, bounding box) is now preserved for chips processed with a `mixer.json` present (`src/spatial_blocks.py`), but is not present in this checkpoint's training data, which predates that field.
- Channel order depends on stored metadata and should be verified against band_names.
- Labels are threshold-derived and may miss weak or ambiguous deforestation signals.
- The model has not yet been shown to generalize beyond the Rondônia training region.

## Ethical and operational considerations

This model should be used as an assistive tool, not as an automated decision authority. False positives and false negatives can both have operational consequences. Users should validate predictions against expert review and local context before acting on results.

If deployed outside the original training area or time period, the model should be revalidated and likely retrained. Any operational system should include uncertainty handling and clear escalation paths for ambiguous predictions.

## Version history

- v1: Initial UNet Rondônia model using 18-channel change-based Sentinel-2 inputs and threshold-derived labels.