# Inference run: {{run_name}}

Copy this file to `notes/phase2/runs/<run_name>.md` and fill in the `{{placeholders}}` before starting. One filled-in copy per inference run so the pipeline provenance stays paired with the outputs.

**Purpose:** {{one line — e.g. "run gee_full_gfc_v1 checkpoint on 2022 vs 2023 dry season over same Rondônia AOI"}}
**Model:** experiment `{{experiment_id}}`, checkpoint `{{stamp}}`
**Target AOI:** {{short description or geometry-file reference}}
**Prepared:** {{YYYY-MM-DD}}
**Author:** {{name}}

---

## 0. Preflight

Do not proceed until each of these is a yes. Any "no" is either a stop or a domain-shift caveat that must be recorded and accepted.

- [ ] Trained checkpoint exists at `{{checkpoint_path}}`
- [ ] Training normalization stats exist at `{{training_data_dir}}/normalization_stats.pkl`
- [ ] Target AOI overlaps the training biome (Amazon / same forest type)
- [ ] Target time window matches the training season pairing (dry-season pre/post)
- [ ] Same S2 collection available for target window (`COPERNICUS/S2_SR_HARMONIZED`)
- [ ] Same cloud probability collection available (`COPERNICUS/S2_CLOUD_PROBABILITY`)

Domain-shift caveats to record if any preflight is "no":

- {{example: pre window is wet season not dry — expect degraded performance}}

---

## 1. Target definition

Every value below must match training unless there's a deliberate reason not to. Divergences here are where "the model silently returns garbage" comes from.

| Field | Training value | Inference value | Match? |
|---|---|---|---|
| S2 collection | `COPERNICUS/S2_SR_HARMONIZED` | {{...}} | {{y/n}} |
| Cloud collection | `COPERNICUS/S2_CLOUD_PROBABILITY` | {{...}} | {{y/n}} |
| Cloud probability cutoff | 40 | {{...}} | {{y/n}} |
| CRS | EPSG:32720 | {{...}} | {{y/n}} |
| Resolution | 10 m | {{...}} | {{y/n}} |
| Patch size | 256 x 256 | {{...}} | {{y/n}} |
| AOI center (lon, lat) | `-63.75, -9.75` | {{...}} | {{y/n}} |
| AOI half-width (deg) | 1.50 | {{...}} | {{y/n}} |
| Pre window | 2020-06-01 to 2020-09-30 | {{YYYY-MM-DD to YYYY-MM-DD}} | {{y/n}} |
| Post window | 2021-06-01 to 2021-09-30 | {{YYYY-MM-DD to YYYY-MM-DD}} | {{y/n}} |
| Canonical channel order | see `band_names.py` | (produced by fresh GEE export) | y |

---

## 2. Export chips from GEE

Reuse the training-time export path so the input contract can't drift.

```bash
# Confirm DATASET_ID + AOI + windows in scripts/gee_export_chips.py match section 1.
python scripts/gee_export_chips.py
python scripts/gee_check_tasks.py --dataset-id {{inference_dataset_id}}
```

**Expected:** sharded `.tfrecord.gz` files in Drive + one `mixer.json` sidecar.

- [ ] All tasks COMPLETED
- [ ] `verify_gate_c.py` passes for `{{inference_dataset_id}}`

---

## 3. Download and process

```bash
scripts/setup_gdrive_rclone.sh {{inference_dataset_id}}
gunzip data/raw/{{inference_dataset_id}}/*.tfrecord.gz
python src/GFC_process_tfrecords4.py --dataset-id {{inference_dataset_id}}
```

**Expected:** `data/processed/{{inference_dataset_id}}/` with `chips/*.npy`, `masks/*.npy`, `metadata.pkl`.

**Do NOT** run `src/split_data.py`. There is no train/val/test partition for an inference dataset, and you must NOT recompute normalization stats — inference reuses the training checkpoint's `normalization_stats.pkl`, unchanged.

---

## 4. QC the processed output

```bash
python scripts/qc_report.py --dataset-id {{inference_dataset_id}}
```

- [ ] Channel order matches canonical band order
- [ ] Chip count matches expected patch count for the AOI
- [ ] No NaN/Inf reported
- [ ] Visual panels look sensible (cloud coverage, terrain matches expectations)

---

## 5. Run inference

*Inference script not yet written — decide the following before implementing.*

**Design decisions**

- [ ] Output format: float32 probability raster (recommended — lets you re-threshold downstream) OR uint8 binary at fixed threshold
- [ ] TTA: on (4× compute, small accuracy bump) or off
- [ ] Chip-boundary handling: no overlap (accept seam artifacts) OR overlap + center-crop stitching
- [ ] Batch size for throughput on `{{DEVICE}}`

**Skeleton**

```
Load model from {{checkpoint_path}}.
Load normalization stats from {{training_data_dir}}/normalization_stats.pkl.
For each chip in data/processed/{{inference_dataset_id}}/chips/:
    Load raw chip.
    Normalize using TRAINING means/stds (do not recompute).
    Convert to torch tensor (1, 18, 256, 256) float32.
    Forward pass → logits → sigmoid → probability (256, 256).
    Optionally TTA: average sigmoid over identity + H-flip + V-flip + HV-flip.
    Save float32 probability array to data/processed/{{inference_dataset_id}}/predictions/<chip_id>.npy.
```

---

## 6. Stitch chips into a georeferenced raster

*Stitching not yet written — decide the following before implementing.*

**Design decisions**

- [ ] Stitching strategy: no-overlap direct placement OR overlap with center-crop OR overlap with probability averaging
- [ ] Output geometry: bounded by AOI, matching training grid (EPSG:32720, 10 m)
- [ ] Missing/masked pixels: how represented (NaN, 0, sentinel)

**Skeleton**

```
Read data/processed/{{inference_dataset_id}}/metadata.pkl.
For each predicted chip:
    Look up (block_row, block_col) or centroid + bbox from metadata.
    Place chip's probability array into the correct spot in the AOI-wide grid.
Reconcile overlaps if applicable.
Write full-AOI probability GeoTIFF with CRS EPSG:32720 and correct geotransform.
Write full-AOI binary GeoTIFF at threshold t = {{threshold_value}}.
```

**Output paths**

- `{{output_dir}}/prob_map.tif` — float32 probabilities
- `{{output_dir}}/binary_map_t{{threshold_value}}.tif` — uint8 binary mask

---

## 7. Verification gates

Before trusting outputs, run each check and record the result.

- [ ] Predicted positive-pixel share vs. Hansen reference rate for this AOI/year: predicted `{{X%}}` vs. reference `{{Y%}}`. Order-of-magnitude divergence = pipeline broken.
- [ ] Overlay against Hansen `lossyear == {{target_year}}` for the same AOI. Broad agreement in obvious clearing regions?
- [ ] Overlay against dNBR/dNDVI threshold-rule baseline (`evaluate_baseline` from `src/test.py`, adapted for this dataset). Broad agreement = sanity; sharp disagreement in one area = investigate.
- [ ] Visual spot-check on 5 random chips: RGB + predicted mask + Hansen mask side by side. No systematic misalignment?
- [ ] Prediction rasters open in QGIS at the correct geographic location with correct CRS.

---

## 8. Deliverables

- [ ] `prob_map.tif`
- [ ] `binary_map_t{{threshold_value}}.tif`
- [ ] Filled copy of this template stored alongside the outputs
- [ ] Verification PNGs from section 7

Storage location: `{{output_dir}}`

---

## Post-run notes

{{Surprises, warnings, followups discovered during the run. Update the model-card "Known limitations" section if a new failure mode was discovered.}}
