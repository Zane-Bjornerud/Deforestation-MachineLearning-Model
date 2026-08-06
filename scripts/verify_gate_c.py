#!/usr/bin/env python
"""
Gate C: verify a dataset's export -> download chain is complete and
consistent before processing is allowed to start.

    conda activate deforest
    python scripts/verify_gate_c.py --dataset-id gee_full_gfc_v1

Checks (see docs/... plan step "Submit and verify the full GEE export"):
  1. Every COMPLETED task has at least one corresponding downloaded file.
  2. Every FAILED/CANCELLED task is either retried (a later task record
     exists) or explicitly marked excluded in the registry.
  3. No duplicated shards (by filename or by content checksum).
  4. Every downloaded file has a checksum, and the manifest isn't stale
     relative to what's actually in raw_path.
  5. The downloaded mixer.json's patch count roughly matches what the AOI
     was expected to produce.
  6. Export metadata (provenance manifest + task registry copy) is stored
     alongside the raw data.

Exits 0 if every check passes, 1 otherwise. Run this before
src/GFC_process_tfrecords4.py, the same way scripts/qc_report.py gates
training.

Known limitation: checks 1, 2, and 5 assume one export task per dataset,
matching the current single-task submission in scripts/gee_export_chips.py.
If that script is later changed to split a large AOI across multiple
tasks, the mixer.json handling here (one mixer.json assumed per raw_path)
and the patch-count check will need to become per-task-prefix instead of
dataset-wide.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_contract import load_contract
from gee_task_registry import load_registry, registry_path
from spatial_blocks import load_mixer

PATCH_COUNT_TOLERANCE = 0.10  # allow +/-10% vs the AOI-area estimate


def _read_download_manifest(raw_path: Path):
    manifest_path = raw_path / "download_manifest.tsv"
    if not manifest_path.exists():
        return None
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def check_completed_tasks_have_files(tasks, rows):
    completed = [t for t in tasks if t["completion_status"] == "COMPLETED"]
    if not completed:
        return None, "no COMPLETED tasks in registry yet -- nothing to check"
    missing = []
    for t in completed:
        matched = [r for r in rows if r["task_id"] == t["task_id"]]
        if not matched:
            missing.append(t["task_id"])
    if missing:
        return False, f"{len(missing)} COMPLETED task(s) have no downloaded files: {missing}"
    return True, f"all {len(completed)} COMPLETED task(s) have >=1 downloaded file"


def check_failed_tasks_handled(tasks):
    failed = [t for t in tasks if t["completion_status"] in ("FAILED", "CANCELLED")]
    if not failed:
        return None, "no FAILED/CANCELLED tasks in registry"
    unhandled = []
    for t in failed:
        if t.get("excluded"):
            continue
        retried = any(other["retry_count"] > t["retry_count"] for other in tasks)
        if not retried:
            unhandled.append(t["task_id"])
    if unhandled:
        return False, (
            f"{len(unhandled)} failed/cancelled task(s) neither retried nor "
            f"excluded: {unhandled} -- resubmit via gee_export_chips.py, or "
            "set \"excluded\": true on the task record if intentionally skipped"
        )
    return True, f"all {len(failed)} failed/cancelled task(s) retried or excluded"


def check_no_duplicate_shards(rows):
    filenames = [r["filename"] for r in rows]
    dup_names = {n for n in filenames if filenames.count(n) > 1}
    if dup_names:
        return False, f"duplicate filenames in manifest: {sorted(dup_names)}"

    checksums = [r["checksum"] for r in rows]
    dup_checksum_files = []
    seen = {}
    for r in rows:
        if r["checksum"] in seen:
            dup_checksum_files.append((seen[r["checksum"]], r["filename"]))
        else:
            seen[r["checksum"]] = r["filename"]
    if dup_checksum_files:
        return False, f"different filenames with identical content checksum: {dup_checksum_files}"
    return True, f"{len(rows)} files, no duplicate filenames or content"


def check_all_checksummed_and_fresh(raw_path, rows):
    missing_checksum = [r["filename"] for r in rows if not r["checksum"]]
    if missing_checksum:
        return False, f"{len(missing_checksum)} manifest row(s) missing a checksum: {missing_checksum}"

    on_disk = {
        p.name for p in raw_path.iterdir() if p.name.endswith(".tfrecord") or p.name.endswith(".tfrecord.gz")
    }
    in_manifest = {r["filename"] for r in rows}
    stale = on_disk - in_manifest
    if stale:
        return False, (
            f"{len(stale)} file(s) on disk not in download_manifest.tsv (manifest is "
            f"stale -- rerun build_download_manifest.py): {sorted(stale)}"
        )
    return True, f"all {len(rows)} files checksummed, manifest matches raw_path contents"


def check_blocks_match_expected(raw_path, tasks):
    try:
        mixer = load_mixer(str(raw_path))
    except FileNotFoundError:
        return False, "no mixer.json in raw_path -- cannot verify block coverage"

    actual_patches = mixer["totalPatches"]
    if not tasks:
        return None, f"mixer.json has {actual_patches} patches but no registered task to compare against"

    expected = tasks[0].get("expected_geographic_extent", {}).get("expected_patch_count_estimate")
    if expected is None:
        return None, f"mixer.json has {actual_patches} patches; no expected_patch_count_estimate recorded to compare"

    ratio = actual_patches / expected if expected else float("inf")
    if abs(ratio - 1) > PATCH_COUNT_TOLERANCE:
        return False, (
            f"downloaded {actual_patches} patches vs expected ~{expected:.0f} "
            f"({ratio:.1%} of expected) -- export may be truncated or AOI mismatched"
        )
    return True, f"downloaded {actual_patches} patches vs expected ~{expected:.0f} ({ratio:.1%})"


def check_metadata_alongside_raw(raw_path):
    missing = []
    if not (raw_path / "export_manifest.json").exists():
        missing.append("export_manifest.json")
    if not (raw_path / "export_task_registry.json").exists():
        missing.append("export_task_registry.json")
    if missing:
        return False, f"missing in {raw_path}: {missing}"
    return True, "export_manifest.json and export_task_registry.json both present in raw_path"


def run(dataset_id: str) -> bool:
    contract = load_contract(dataset_id)
    raw_path = Path(contract.raw_path)
    registry = load_registry(dataset_id)
    tasks = registry["tasks"]
    rows = _read_download_manifest(raw_path) or []

    checks = [
        ("Every successful task has a downloaded file", check_completed_tasks_have_files(tasks, rows)),
        ("Failed tasks retried or excluded", check_failed_tasks_handled(tasks)),
        ("No duplicated shards", check_no_duplicate_shards(rows) if rows else (False, "no download_manifest.tsv -- run build_download_manifest.py")),
        ("Every file has a checksum, manifest is fresh", check_all_checksummed_and_fresh(raw_path, rows) if rows else (False, "no download_manifest.tsv -- run build_download_manifest.py")),
        ("Expected blocks match downloaded blocks", check_blocks_match_expected(raw_path, tasks)),
        ("Export metadata stored alongside raw data", check_metadata_alongside_raw(raw_path)),
    ]

    print(f"=== Gate C: {dataset_id} ===")
    print(f"Task registry: {registry_path(dataset_id)} ({len(tasks)} task(s))")
    print(f"Raw path: {raw_path}\n")

    hard_fail = False
    for name, (ok, msg) in checks:
        symbol = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"[{symbol}] {name}: {msg}")
        if ok is False:
            hard_fail = True

    print()
    if hard_fail:
        print("Gate C: FAILED -- do not proceed to processing.")
    else:
        print("Gate C: no hard failures (some checks may be SKIPped -- read above).")
    return not hard_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify export/download integrity before processing")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    passed = run(args.dataset_id)
    sys.exit(0 if passed else 1)
