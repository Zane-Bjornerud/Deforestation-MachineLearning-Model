# Inference run

Copy this file per run, rename it, fill it in as you go.

Model:
AOI:
Pre window:
Post window:
Date started:

## Before starting

- Checkpoint file exists
- Training normalization_stats.pkl exists (reuse it, don't recompute)
- AOI is in the same biome as training
- Windows are dry-season like training

## Export

Edit AOI + dates in `scripts/gee_export_chips.py`, then:

```
python scripts/gee_export_chips.py
python scripts/gee_check_tasks.py --dataset-id <id>
```

Wait for COMPLETED.

## Download and process

```
scripts/setup_gdrive_rclone.sh <id>
gunzip data/raw/<id>/*.tfrecord.gz
python src/GFC_process_tfrecords4.py --dataset-id <id>
```

Skip `split_data.py` — there's no train/val split for inference, and the training stats stay as-is.

## QC

```
python scripts/qc_report.py --dataset-id <id>
```

Check the visual panels look right.

## Inference (script TBD)

Load the model. Load the training normalization stats. For each chip: normalize with the training stats, forward pass, save the sigmoid output as float32.

Decide before writing:
- probabilities or binary output
- TTA yes/no
- overlap between chips yes/no

## Stitch

Reassemble the per-chip predictions into one GeoTIFF at 10 m, EPSG:32720.

## Check before trusting

- Predicted positive % is in the same ballpark as Hansen for this year and area
- Visual overlay against Hansen looks reasonable
- Visual overlay against dNBR/dNDVI rule looks reasonable
- Rasters land in the right spot in QGIS

## Outputs

- `prob_map.tif`
- `binary_map.tif`
- This file, filled in

## Notes

