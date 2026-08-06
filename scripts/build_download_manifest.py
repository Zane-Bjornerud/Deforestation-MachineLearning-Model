#!/usr/bin/env python
"""
Build data/raw/<dataset_id>/download_manifest.tsv from the shard files
already downloaded into a dataset's raw_path, and backfill the task
registry (artifacts/exports/<dataset_id>/export_manifest.json) with each
task's output files and, if a mixer.json is present, the spatial blocks it
covers.

Run after pulling files down from Drive (manually or via
scripts/setup_gdrive_rclone.sh), before processing:

    conda activate deforest
    python scripts/build_download_manifest.py --dataset-id gee_full_gfc_v1

Columns: filename, file_size_bytes, checksum, task_id, block_id,
download_status, verified_at.

block_id is intentionally left blank: block IDs are chip-level (see
src/spatial_blocks.py), but this manifest tracks whole shard *files*, and a
single shard spans ~hundreds of chips across many blocks. The task
registry's per-task block_ids field (backfilled below from mixer.json) is
the block-coverage record; this manifest is a file-integrity record.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_contract import load_contract
from gee_task_registry import load_registry, registry_path, save_registry, update_task_output
from spatial_blocks import DEFAULT_BLOCK_SIZE_TILES, load_mixer, patch_geometry

MANIFEST_COLUMNS = [
    "filename",
    "file_size_bytes",
    "checksum",
    "task_id",
    "block_id",
    "download_status",
    "verified_at",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def match_task(filename: str, tasks: list[dict]) -> str | None:
    """A shard belongs to whichever task's expected_file_prefix it starts
    with. Ambiguous if the registry has two tasks with one prefix a strict
    prefix of the other -- not handled specially, first match wins."""
    for t in tasks:
        if filename.startswith(t["expected_file_prefix"]):
            return t["task_id"]
    return None


def block_ids_for_task(mixer: dict, block_size_tiles: int) -> list[str]:
    total_patches = mixer["totalPatches"]
    ids = {
        patch_geometry(mixer, i, block_size_tiles)["block_id"] for i in range(total_patches)
    }
    return sorted(ids)


def build_manifest(dataset_id: str, block_size_tiles: int = DEFAULT_BLOCK_SIZE_TILES) -> Path:
    contract = load_contract(dataset_id)
    raw_path = Path(contract.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} does not exist -- nothing downloaded yet.")

    registry = load_registry(dataset_id)
    tasks = registry["tasks"]
    if not tasks:
        print(
            f"WARNING: no tasks registered for {dataset_id!r} in "
            f"{registry_path(dataset_id)} -- task_id column will be blank for "
            "every file. This is expected for datasets exported before the "
            "task registry existed (e.g. the canary)."
        )

    shard_files = sorted(
        p for p in raw_path.iterdir() if p.name.endswith(".tfrecord") or p.name.endswith(".tfrecord.gz")
    )
    if not shard_files:
        raise FileNotFoundError(f"No *.tfrecord or *.tfrecord.gz files found in {raw_path}")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for path in shard_files:
        task_id = match_task(path.name, tasks)
        rows.append(
            {
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
                "checksum": f"sha256:{sha256_of(path)}",
                "task_id": task_id or "",
                "block_id": "",  # see module docstring
                "download_status": "present",
                "verified_at": now,
            }
        )
        print(f"  checksummed {path.name} ({rows[-1]['file_size_bytes']:,} bytes)")

    manifest_path = raw_path / "download_manifest.tsv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Backfill task registry: which files came from which task, and (if a
    # mixer.json is available) which spatial blocks that task's export
    # actually covers.
    unmatched = [r["filename"] for r in rows if not r["task_id"]]
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} file(s) didn't match any registered "
            f"task's expected_file_prefix, e.g. {unmatched[0]!r} -- their "
            "task_id is blank in the manifest and they won't be backfilled "
            "into the registry."
        )

    if tasks:
        try:
            mixer = load_mixer(str(raw_path))
            block_ids = block_ids_for_task(mixer, block_size_tiles)
            print(f"mixer.json found -- {len(block_ids)} blocks covered (block_size_tiles={block_size_tiles})")
        except FileNotFoundError:
            mixer = None
            block_ids = None
            print("No mixer.json in raw_path yet -- block_ids left null in the registry.")

        for t in tasks:
            output_files = [r["filename"] for r in rows if r["task_id"] == t["task_id"]]
            if output_files:
                update_task_output(dataset_id, t["task_id"], output_files, block_ids)

    # "Export metadata is stored alongside raw data" -- copy the task
    # registry next to the raw shards, not just in artifacts/exports/.
    registry_copy_path = raw_path / "export_task_registry.json"
    with open(registry_copy_path, "w") as f:
        json.dump(load_registry(dataset_id), f, indent=2)

    print(f"\nDownload manifest: {manifest_path} ({len(rows)} files)")
    print(f"Registry copy: {registry_copy_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build download_manifest.tsv for a dataset's downloaded GEE export shards"
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--block-size-tiles",
        type=int,
        default=DEFAULT_BLOCK_SIZE_TILES,
        help="Defaults to src/spatial_blocks.py's DEFAULT_BLOCK_SIZE_TILES, the same "
        "value src/GFC_process_tfrecords4.py uses. Only override this if you've "
        "also changed BLOCK_SIZE_TILES there -- otherwise the block_ids backfilled "
        "into the task registry here won't match what processing produces later.",
    )
    args = parser.parse_args()
    build_manifest(args.dataset_id, args.block_size_tiles)
