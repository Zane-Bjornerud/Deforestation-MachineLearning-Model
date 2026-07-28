#!/usr/bin/env python
"""
Build a Sentinel-2 (pre/post) + Hansen Global Forest Change deforestation
chip dataset and submit it as a Google Earth Engine batch export to Drive.

Run in the `deforest` conda env (has earthengine-api installed):
    conda activate deforest
    python scripts/gee_export_chips.py

Output lands in Google Drive under DRIVE_FOLDER as sharded .tfrecord files,
matching the format src/GFC_process_tfrecords4.py already knows how to parse
(named float bands + a single byte-valued "label" band).

After the task completes (check with scripts/gee_check_tasks.py), download
the files from Drive into data/.
"""

import ee
import json
import os

# --- Config -----------------------------------------------------------------

PROJECT = "decent-being-438620-b7"

# EDIT ME: rough Rondonia deforestation-frontier bounding box. Narrow this to
# your actual area of interest before running a real export.
AOI_COORDS = [
    [-64.5, -10.5],
    [-63.0, -10.5],
    [-63.0, -9.0],
    [-64.5, -9.0],
]
CRS = "EPSG:32720"  # UTM 20S, covers most of Rondonia; adjust if AOI moves

# Pre/post imagery windows (dry season = fewer clouds in the Amazon)
PRE_START, PRE_END = "2020-06-01", "2020-09-30"
POST_START, POST_END = "2021-06-01", "2021-09-30"

# Hansen Global Forest Change: label = forest in 2000 that was lost in TARGET_LOSS_YEAR
GFC_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"  # latest as of 2026; bump the year/version as Hansen releases new ones
TARGET_LOSS_YEAR = 2021
FOREST_COVER_THRESHOLD = 30  # % tree cover in 2000 required to count as forest

MAX_CLOUD_PROB = 40  # from COPERNICUS/S2_CLOUD_PROBABILITY, 0-100

PATCH_SIZE = 256
SCALE = 10  # meters/pixel

DRIVE_FOLDER = "deforest_export"
FILE_PREFIX = f"rondonia_deforest_chips_{TARGET_LOSS_YEAR}"

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

EXPORT_METADATA = {
    "project": PROJECT,
    "file_prefix": FILE_PREFIX,
    "aoi_coords": AOI_COORDS,
    "crs": CRS,
    "scale_m_per_px": SCALE,
    "patch_size": PATCH_SIZE,
    "s2_bands": S2_BANDS,
    "pre_start": PRE_START,
    "pre_end": PRE_END,
    "post_start": POST_START,
    "post_end": POST_END,
    "target_loss_year": TARGET_LOSS_YEAR,
    "forest_cover_threshold": FOREST_COVER_THRESHOLD,
}

# --- Imagery ------------------------------------------------------------


def build_composite(aoi, start, end, max_cloud_prob=MAX_CLOUD_PROB):
    """Cloud-masked median Sentinel-2 SR composite, scaled to reflectance."""
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
    s2_clouds = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(start, end)
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


def main():
    ee.Initialize(project=PROJECT)
    aoi = ee.Geometry.Polygon([AOI_COORDS])
    image = build_export_image(aoi)
    task = submit_export(image, aoi)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/export_metadata.json", "w") as f:
        json.dump(EXPORT_METADATA, f, indent=2)

    print(f"Submitted export task: {task.id}")
    print(f"Drive folder: {DRIVE_FOLDER} (prefix: {FILE_PREFIX})")
    print("Check progress with: python scripts/gee_check_tasks.py")


if __name__ == "__main__":
    main()
