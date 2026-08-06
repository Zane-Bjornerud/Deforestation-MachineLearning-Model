"""Derive per-chip geographic identity from a GEE export's mixer.json sidecar.

GEE's batch TFRecord export writes one <prefix>-mixer.json file per export
alongside the .tfrecord shards. It records the patch grid's affine transform
and dimensions -- enough to compute every chip's tile row/column, centroid,
and bounding box purely from its sequential position in the patch grid,
without any change to the export image itself.

This is the basis for the block-level spatial split in split_data.py: chips
are grouped into fixed-size blocks (block_size_tiles x block_size_tiles
patches) so that whole blocks, not individual chips, get assigned to
train/val/test -- avoiding the leakage that comes from two neighboring
chips (near-identical imagery/context) landing on opposite sides of a split.

Load-bearing assumption: GEE fills TFRecord shards with contiguous,
sequential chunks of the patch grid (row-major, following patchesPerRow),
not round-robin. Callers must process shards in sorted filename order and
maintain one running counter across all of them for the derived tile
row/column to be correct.
"""

import json
from pathlib import Path


def load_mixer(raw_path: str) -> dict:
    """Load the mixer.json sidecar for a raw export directory.

    Raises FileNotFoundError if none or more than one is present -- callers
    should treat this as "no spatial metadata available" and degrade
    gracefully rather than crash the whole processing run.
    """
    matches = list(Path(raw_path).glob("*mixer.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *mixer.json in {raw_path}, found "
            f"{len(matches)}"
        )
    return json.loads(matches[0].read_text())


def load_export_task_id(raw_path: str) -> str | None:
    """Best-effort lookup of the GEE task id from export_manifest.json."""
    manifest_path = Path(raw_path) / "export_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text()).get("gee_task_id")
    except (json.JSONDecodeError, OSError):
        return None


def patch_geometry(mixer: dict, global_patch_index: int, block_size_tiles: int) -> dict:
    """Derive tile row/col, centroid, bbox, and block_id for one patch.

    global_patch_index is this chip's 0-based position in the sequential,
    row-major patch grid across all shards (see module docstring).
    """
    patches_per_row = mixer["patchesPerRow"]
    patch_w, patch_h = mixer["patchDimensions"]
    a, b, c, d, e, f = mixer["projection"]["affine"]["doubleMatrix"]

    tile_row = global_patch_index // patches_per_row
    tile_col = global_patch_index % patches_per_row

    def to_map(col_px, row_px):
        x = a * col_px + b * row_px + c
        y = d * col_px + e * row_px + f
        return x, y

    col0, row0 = tile_col * patch_w, tile_row * patch_h
    x0, y0 = to_map(col0, row0)
    x1, y1 = to_map(col0 + patch_w, row0 + patch_h)
    cx, cy = to_map(col0 + patch_w / 2, row0 + patch_h / 2)

    block_row = tile_row // block_size_tiles
    block_col = tile_col // block_size_tiles

    return {
        "tile_row": tile_row,
        "tile_col": tile_col,
        "centroid_x": cx,
        "centroid_y": cy,
        "bounding_box": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
        "block_id": f"{block_row}_{block_col}",
        "block_row": block_row,
        "block_col": block_col,
    }
