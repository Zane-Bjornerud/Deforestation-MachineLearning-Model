#!/usr/bin/env python
"""
Build a Sentinel-2 (pre/post) + Hansen Global Forest Change deforestation
chip dataset and submit it as a Google Earth Engine batch export to Drive.

Run in the `deforest` conda env (has earthengine-api installed):
    conda activate deforest
    python scripts/gee_export_chips.py

All label parameters (Hansen asset version, forest cover threshold, target
year, pre/post composite windows) come from the dataset contract at
configs/datasets/<DATASET_ID>.yaml -- that file is the single source of
truth, not this script. Edit the contract, not the constants here, to change
the label definition.

Output lands in Google Drive under DRIVE_FOLDER as sharded .tfrecord files,
matching the format src/GFC_process_tfrecords4.py already knows how to parse
(named float bands + a single byte-valued "label" band).

After the task completes (check with scripts/gee_check_tasks.py), download
the files from Drive into data/raw/<DATASET_ID>/ (the contract's raw_path).
Then run:
    python src/GFC_process_tfrecords4.py --dataset-id <DATASET_ID>
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import ee

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from band_names import CANONICAL_BAND_ORDER
from dataset_contract import LABEL_MODE_HANSEN, load_contract
from gee_task_registry import add_task_submission

# --- Config -----------------------------------------------------------------

PROJECT = "decent-being-438620-b7"

# EDIT ME: which dataset contract this run produces. Must have a matching
# configs/datasets/<DATASET_ID>.yaml with label_mode: hansen_loss.
# "gee_canary_gfc_v1" for a small test export, "gee_full_gfc_v1" for the real
# Phase 1 export.
DATASET_ID = "gee_full_gfc_v1"

CONTRACT = load_contract(DATASET_ID)
if CONTRACT.label_mode != LABEL_MODE_HANSEN:
    raise ValueError(
        f"{DATASET_ID}: this script only produces hansen_loss datasets, "
        f"contract says label_mode={CONTRACT.label_mode!r}"
    )

# EDIT ME: area of interest, centered on a Rondonia deforestation-frontier
# "fishbone" hotspot along BR-364 (-63.75, -9.75) -- the same center the
# canary export used. The AOI is a square of side 2*AOI_HALF_WIDTH_DEG
# degrees around that center; to resize, change AOI_HALF_WIDTH_DEG only
# and keep the center fixed, so you stay over the same known-active region
# rather than drifting into unvalidated ground.
#
# Before picking a size, each was checked against the actual Hansen GFC
# 2021 loss layer (real loss %, not a guess) to confirm it isn't diluted
# into mostly-empty forest or already-cleared land:
#   half_width=0.75 (the original canary box)  -- 1.80% loss,  4,184 patches, ~20 GB
#   half_width=1.50 (this box, "2x" -- current) -- 1.44% loss, 16,736 patches, ~80 GB
#   half_width=2.25 ("3x")                      -- 1.02% loss, 37,661 patches, ~180 GB
# Loss % drops as the box grows (you're pulling in more interior forest /
# already-cleared land beyond the active edge), and a much bigger box risks
# needing multiple GEE export tasks, which this script does not currently
# support (see scripts/verify_gate_c.py's docstring). Re-check a new
# candidate's loss % before committing to it -- see README.md "Area of
# interest" for how.
AOI_CENTER_LON, AOI_CENTER_LAT = -63.75, -9.75
AOI_HALF_WIDTH_DEG = 1.50  # "2x" box, locked in -- see comment above for other sizes
AOI_COORDS = [
    [AOI_CENTER_LON - AOI_HALF_WIDTH_DEG, AOI_CENTER_LAT - AOI_HALF_WIDTH_DEG],
    [AOI_CENTER_LON + AOI_HALF_WIDTH_DEG, AOI_CENTER_LAT - AOI_HALF_WIDTH_DEG],
    [AOI_CENTER_LON + AOI_HALF_WIDTH_DEG, AOI_CENTER_LAT + AOI_HALF_WIDTH_DEG],
    [AOI_CENTER_LON - AOI_HALF_WIDTH_DEG, AOI_CENTER_LAT + AOI_HALF_WIDTH_DEG],
]
CRS = "EPSG:32720"  # UTM 20S, covers most of Rondonia; adjust if AOI moves

# Pre/post imagery windows and Hansen label parameters, from the contract.
PRE_START, PRE_END = CONTRACT.extra["pre_start"], CONTRACT.extra["pre_end"]
POST_START, POST_END = CONTRACT.extra["post_start"], CONTRACT.extra["post_end"]
GFC_ASSET = CONTRACT.extra["gfc_asset"]
TARGET_LOSS_YEAR = CONTRACT.target_year
FOREST_COVER_THRESHOLD = CONTRACT.extra["forest_cover_threshold"]

MAX_CLOUD_PROB = 40  # from COPERNICUS/S2_CLOUD_PROBABILITY, 0-100

PATCH_SIZE = 256
SCALE = 10  # meters/pixel

DRIVE_FOLDER = "deforest_export"
FILE_PREFIX = f"{DATASET_ID}_{TARGET_LOSS_YEAR}"

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

IMAGERY_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"

# --- Imagery ------------------------------------------------------------


def build_composite(aoi, start, end, max_cloud_prob=MAX_CLOUD_PROB):
    """Cloud-masked median Sentinel-2 SR composite, scaled to reflectance."""
    s2 = ee.ImageCollection(IMAGERY_COLLECTION).filterBounds(aoi).filterDate(start, end)
    s2_clouds = (
        ee.ImageCollection(CLOUD_COLLECTION).filterBounds(aoi).filterDate(start, end)
    )

    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=s2_clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def mask_clouds(img):
        clouds = ee.Image(img.get("cloud_mask")).select("probability")
        return img.updateMask(clouds.lt(max_cloud_prob))

    masked = ee.ImageCollection(joined).map(mask_clouds)
    # NOTE: kept as raw digital numbers (no *0.0001 reflectance scaling) to match
    # the existing chips in data/*.tfrecord, which are unscaled (e.g. B2 up to ~3900).
    return masked.select(S2_BANDS).median().clip(aoi)


def add_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    return img.addBands([ndvi, nbr])


def build_label(aoi, target_year=TARGET_LOSS_YEAR, gfc_asset=GFC_ASSET, forest_thresh=FOREST_COVER_THRESHOLD):
    """Binary byte mask: forest in 2000 that Hansen GFC says was lost in target_year."""
    gfc = ee.Image(gfc_asset)
    loss_year_code = target_year - 2000
    was_forest = gfc.select("treecover2000").gte(forest_thresh)
    lost_this_year = gfc.select("lossyear").eq(loss_year_code)
    return was_forest.And(lost_this_year).toByte().rename("label").clip(aoi)


def build_export_image(aoi):
    renamed = [f"{b}_pre" for b in S2_BANDS] + ["NDVI_pre", "NBR_pre"]
    pre = add_indices(build_composite(aoi, PRE_START, PRE_END)).select(
        S2_BANDS + ["NDVI", "NBR"], renamed
    )

    renamed_post = [f"{b}_post" for b in S2_BANDS] + ["NDVI_post", "NBR_post"]
    post = add_indices(build_composite(aoi, POST_START, POST_END)).select(
        S2_BANDS + ["NDVI", "NBR"], renamed_post
    )

    dndvi = post.select("NDVI_post").subtract(pre.select("NDVI_pre")).rename("dNDVI")
    dnbr = pre.select("NBR_pre").subtract(post.select("NBR_post")).rename("dNBR")

    imagery = pre.addBands(post).addBands(dndvi).addBands(dnbr).toFloat()
    label = build_label(aoi)

    return imagery.addBands(label)  # label stays byte -> exports as bytes_list


# --- Export ------------------------------------------------------------


def submit_export(image, aoi):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=FILE_PREFIX,
        folder=DRIVE_FOLDER,
        fileNamePrefix=FILE_PREFIX,
        region=aoi,
        scale=SCALE,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="TFRecord",
        formatOptions={
            "patchDimensions": [PATCH_SIZE, PATCH_SIZE],
            "compressed": True,
        },
    )
    task.start()
    return task


def _git_commit_info():
    """Best-effort git commit + dirty-tree flag, for export provenance."""
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", None


def write_export_manifest(raw_dir, task_id):
    """Write the minimum metadata needed to reproduce/audit this export,
    next to where the downloaded TFRecords will land."""
    git_commit, git_dirty = _git_commit_info()

    manifest = {
        "dataset_id": CONTRACT.dataset_id,
        "label_contract_version": CONTRACT.label_contract_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gee_project": PROJECT,
        "imagery_collection": IMAGERY_COLLECTION,
        "cloud_probability_collection": CLOUD_COLLECTION,
        "hansen_asset": GFC_ASSET,
        "forest_cover_threshold": FOREST_COVER_THRESHOLD,
        "target_year": TARGET_LOSS_YEAR,
        "pre_start": PRE_START,
        "pre_end": PRE_END,
        "post_start": POST_START,
        "post_end": POST_END,
        "aoi_coords": AOI_COORDS,
        "crs": CRS,
        "scale_m_per_px": SCALE,
        "chip_dimensions": [PATCH_SIZE, PATCH_SIZE],
        "channel_names_and_order": CANONICAL_BAND_ORDER,
        "label_construction": CONTRACT.label_semantics,
        "gee_task_id": task_id,
        "export_prefix": FILE_PREFIX,
        "submission_time_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = os.path.join(raw_dir, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def _expected_patch_count(aoi):
    """Rough expected patch count from the AOI's true area (via EE, which
    already accounts for polygon shape) divided by one patch's ground
    footprint. Used by scripts/verify_gate_c.py to sanity-check that a
    downloaded mixer.json's totalPatches roughly matches what was asked
    for -- catching a silently truncated/partial export."""
    area_m2 = aoi.area(maxError=1).getInfo()
    patch_area_m2 = (SCALE * PATCH_SIZE) ** 2
    return area_m2 / patch_area_m2


def main():
    ee.Initialize(project=PROJECT)
    aoi = ee.Geometry.Polygon([AOI_COORDS])
    image = build_export_image(aoi)
    task = submit_export(image, aoi)

    raw_dir = CONTRACT.raw_path
    os.makedirs(raw_dir, exist_ok=True)
    manifest_path = write_export_manifest(raw_dir, task.id)

    task_record = add_task_submission(
        dataset_id=DATASET_ID,
        task_id=task.id,
        expected_file_prefix=FILE_PREFIX,
        expected_geographic_extent={
            "aoi_coords": AOI_COORDS,
            "crs": CRS,
            "expected_patch_count_estimate": _expected_patch_count(aoi),
        },
    )

    print(f"Submitted export task: {task.id}")
    print(f"Dataset contract: configs/datasets/{DATASET_ID}.yaml")
    print(f"Export manifest (provenance): {manifest_path}")
    print(f"Task registry: artifacts/exports/{DATASET_ID}/export_manifest.json "
          f"(retry_count={task_record['retry_count']})")
    print(f"Drive folder: {DRIVE_FOLDER} (prefix: {FILE_PREFIX})")
    print(f"Check progress with: python scripts/gee_check_tasks.py --dataset-id {DATASET_ID}")
    print(f"After download, place TFRecords in {raw_dir}/")
    print(f"Then run: python scripts/build_download_manifest.py --dataset-id {DATASET_ID}")
    print(f"Then run: python scripts/verify_gate_c.py --dataset-id {DATASET_ID}")
    print(f"Then run: python src/GFC_process_tfrecords4.py --dataset-id {DATASET_ID}")


if __name__ == "__main__":
    main()
