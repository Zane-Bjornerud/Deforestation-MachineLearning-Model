"""Task-level registry for GEE batch export tasks.

This is a different thing from the per-dataset provenance manifest
(export_manifest.json written by scripts/gee_export_chips.py into
raw_path -- git commit, imagery collection, Hansen params, etc.). This
registry tracks the *lifecycle* of each individual Earth Engine export
task for a dataset: submission, completion, retries, and eventually which
downloaded files and spatial blocks came out of it. A dataset backed by an
AOI too large for one export would have multiple task records here.

Written at artifacts/exports/<dataset_id>/export_manifest.json:
  - scripts/gee_export_chips.py appends a record at submission time.
  - scripts/gee_check_tasks.py updates completion_status/failure_message
    when it polls Earth Engine.
  - scripts/build_download_manifest.py backfills output_files and
    block_ids once shards are downloaded and a mixer.json is available.
  - scripts/verify_gate_c.py reads it to check the full export -> download
    chain before processing is allowed to start.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_DIR = Path("artifacts/exports")


def registry_path(dataset_id: str) -> Path:
    return REGISTRY_DIR / dataset_id / "export_manifest.json"


def load_registry(dataset_id: str) -> dict:
    path = registry_path(dataset_id)
    if not path.exists():
        return {"dataset_id": dataset_id, "tasks": []}
    return json.loads(path.read_text())


def save_registry(dataset_id: str, registry: dict) -> Path:
    path = registry_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2))
    return path


def add_task_submission(
    dataset_id: str,
    task_id: str,
    expected_file_prefix: str,
    expected_geographic_extent: dict,
) -> dict:
    """Append a new task record at submission time.

    retry_count is the number of prior task records already in this
    dataset's registry -- 0 for a first submission, 1+ if this is a
    resubmission after an earlier task failed or was cancelled.
    """
    registry = load_registry(dataset_id)
    record = {
        "task_id": task_id,
        "block_ids": None,  # backfilled post-download from mixer.json, see build_download_manifest.py
        "expected_file_prefix": expected_file_prefix,
        "expected_geographic_extent": expected_geographic_extent,
        "submission_status": "submitted",
        "submission_time_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "PENDING",
        "failure_message": None,
        "retry_count": len(registry["tasks"]),
        "excluded": False,
        "output_files": [],
    }
    registry["tasks"].append(record)
    save_registry(dataset_id, registry)
    return record


def update_task_completion(
    dataset_id: str, task_id: str, completion_status: str, failure_message: str | None = None
) -> bool:
    """Update a task's completion_status/failure_message. Returns whether a
    matching task record was found."""
    registry = load_registry(dataset_id)
    found = False
    for t in registry["tasks"]:
        if t["task_id"] == task_id:
            t["completion_status"] = completion_status
            t["failure_message"] = failure_message
            found = True
    if found:
        save_registry(dataset_id, registry)
    return found


def update_task_output(
    dataset_id: str, task_id: str, output_files: list[str], block_ids: list[str] | None
) -> bool:
    """Backfill output_files and block_ids once shards are downloaded and
    (if available) a mixer.json has been parsed. Returns whether a matching
    task record was found."""
    registry = load_registry(dataset_id)
    found = False
    for t in registry["tasks"]:
        if t["task_id"] == task_id:
            t["output_files"] = output_files
            t["block_ids"] = block_ids
            found = True
    if found:
        save_registry(dataset_id, registry)
    return found
