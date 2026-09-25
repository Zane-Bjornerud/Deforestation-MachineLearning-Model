# Model Input Specification

## 1. Model identity

- Model name: Deforestation segmentation model (Hansen GFC labels; legacy change-index labels also supported)
- Model version: Current main branch — Hansen-loss pipeline primary, change-based pipeline retained as legacy
- Architecture: U-Net with ResNet34 encoder
- Checkpoint: `<experiment.checkpoint_dir>/<run_stamp>/best_model.pth` (per-experiment, per-run; see `configs/experiments/<id>.yaml`)
- Number of input channels: 18
- Output channels: 1 (binary mask logits)

## 2. Intended prediction target

- Positive-pixel definition:
  - Hansen (current): `treecover2000 ≥ forest_cover_threshold` AND Hansen `lossyear` matches the dataset's `target_year`.
  - Change-based (legacy): pixel satisfies both dNBR and dNDVI disturbance thresholds.
- Negative-pixel definition: All other pixels.
- Ignore-pixel definition: None explicitly defined.
- Label source: `hansen_gfc` for active datasets (`gee_full_gfc_v1`, `gee_canary_gfc_v1`, `existing_gfc_recovery_v0`); `change_index_threshold` for `legacy_threshold_v1`.
- Label version: pinned per dataset contract in `configs/datasets/<id>.yaml`.
- Label time period: Change between pre window 2020-06-01 to 2020-09-30 and post window 2021-06-01 to 2021-09-30

## 3. Tensor contract

- Expected shape: C x H x W = 18 x 256 x 256
- Expected dtype: float32
- Channel dimension position: First dimension (channel-first)
- Accepted image dimensions: Pipeline standardizes to 256 x 256 before training
- Default tile dimensions: 256 x 256
- No-data representation: No explicit sentinel value documented; invalid handling is not explicitly enforced in dataset loader

## 4. Channel order

| Index | Channel | Source | Time period | Native resolution | Served resolution | Raw range | Preprocessing |
|------:|---------|--------|-------------|-------------------|-------------------|-----------|---------------|
| 0 | B2_pre | Sentinel-2 SR Harmonized | Pre | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 1 | B3_pre | Sentinel-2 SR Harmonized | Pre | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 2 | B4_pre | Sentinel-2 SR Harmonized | Pre | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 3 | B8_pre | Sentinel-2 SR Harmonized | Pre | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 4 | B11_pre | Sentinel-2 SR Harmonized | Pre | 20 m | 10 m | DN-scale | Upsampled by EE export grid |
| 5 | B12_pre | Sentinel-2 SR Harmonized | Pre | 20 m | 10 m | DN-scale | Upsampled by EE export grid |
| 6 | NDVI_pre | Derived from B8,B4 | Pre | Derived | 10 m | approx −1 to 1 | Computed index |
| 7 | NBR_pre | Derived from B8,B12 | Pre | Derived | 10 m | approx −1 to 1 | Computed index |
| 8 | B2_post | Sentinel-2 SR Harmonized | Post | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 9 | B3_post | Sentinel-2 SR Harmonized | Post | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 10 | B4_post | Sentinel-2 SR Harmonized | Post | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 11 | B8_post | Sentinel-2 SR Harmonized | Post | 10 m | 10 m | DN-scale | Cloud-mask + median composite |
| 12 | B11_post | Sentinel-2 SR Harmonized | Post | 20 m | 10 m | DN-scale | Upsampled by EE export grid |
| 13 | B12_post | Sentinel-2 SR Harmonized | Post | 20 m | 10 m | DN-scale | Upsampled by EE export grid |
| 14 | NDVI_post | Derived from B8,B4 | Post | Derived | 10 m | approx −1 to 1 | Computed index |
| 15 | NBR_post | Derived from B8,B12 | Post | Derived | 10 m | approx −1 to 1 | Computed index |
| 16 | dNDVI | Post minus Pre NDVI | Change | Derived | 10 m | centered around 0 | Difference index |
| 17 | dNBR | Pre minus Post NBR | Change | Derived | 10 m | centered around 0 | Difference index |

## 5. Source imagery

- Dataset identifier: COPERNICUS/S2_SR_HARMONIZED and COPERNICUS/S2_CLOUD_PROBABILITY
- Processing level: Sentinel-2 surface reflectance harmonized
- Spatial resolution: Exported at 10 m pixel size
- CRS: EPSG:32720
- Cloud filtering: Cloud probability threshold less than 40
- Cloud masking: Per-scene probability mask join and threshold
- Temporal composite: Median composite over configured pre and post windows
- Date-selection rules: Fixed pre and post ranges in script constants

## 6. Preprocessing

1. Read TFRecord examples and parse feature bands.
2. Read the target label band: Hansen pipeline uses `lossyear` + `treecover2000`; change-based (legacy) pipeline ignores any stored label field and derives from indices instead.
3. Reshape each band to square and resize to 256 x 256 if needed.
4. Stack channels as channel-first tensor.
5. Generate binary target mask: Hansen pipeline uses `treecover2000 ≥ forest_cover_threshold` AND `lossyear == target_year`; change-based (legacy) pipeline uses dNBR AND dNDVI disturbance thresholds.
6. Save chip and mask as float32 numpy arrays.
7. Save metadata including band names and source provenance.

## 7. Normalization

- Method: Per-channel z-score normalization
- Statistics source: Sampled processed chips from train metadata file
- Training-only statistics: Yes
- Mean per channel: Stored in normalization_stats.pkl
- Standard deviation per channel: Stored in normalization_stats.pkl
- Clipping: None explicitly applied
- Invalid-pixel behavior: No explicit nan or nodata filtering in dataset normalization

## 8. Spatial assumptions

- Training region: AOI rectangle in Rondônia defined in export script
- Chip size: 256 x 256
- Pixel size: 10 m served grid
- Projection: EPSG:32720
- Resampling: Resize to 256 x 256 in processor when needed; export itself uses fixed patch dimensions
- Tile overlap: Not explicitly configured in current export script
- Padding: Not explicitly configured

## 9. Temporal assumptions

- Baseline period: 2020-06-01 to 2020-09-30
- Comparison period: 2021-06-01 to 2021-09-30
- Label period: Thresholded change between baseline and comparison composites
- Seasonal matching rules: Dry-season window matching across years

## 10. Validation rules

An input must be rejected when:

- Channel count is not exactly 18.
- Channel order does not match expected canonical order or recorded metadata order
- Spatial dimensions are not 256 x 256 and no approved resize policy is applied.
- Input contains non-finite values and no cleaning policy is applied.
- Required change channels are missing for change-based target generation
- CRS or pixel size metadata is missing or incompatible with training contract during upstream data creation.

## 11. Known uncertainties

- Split composition depends on where deforestation activity falls within the AOI. Because train/val/test are contiguous geographic bands rather than scattered blocks, any one split can be non-representative of the whole AOI. `create_block_splits` prints per-split positive rate and positive-fraction range at split time so this is visible up front.
- Channel order is enforced against `band_names.CANONICAL_BAND_ORDER` at dataset-load time (`_validate_band_names` in `src/dataset.py`) — any chip whose stored `band_names` do not match the canonical 18 in canonical order raises immediately. Metadata `band_names` remains the local source of truth per chip.
- Hansen labels inherit Hansen GFC's own limitations (annual `lossyear` granularity, sub-canopy or partial-canopy loss missed).