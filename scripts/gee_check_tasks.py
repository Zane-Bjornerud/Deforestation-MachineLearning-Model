#!/usr/bin/env python
"""
List recent Earth Engine batch export tasks and their status.

    conda activate deforest
    python scripts/gee_check_tasks.py
    python scripts/gee_check_tasks.py --dataset-id gee_full_gfc_v1

With --dataset-id, also writes each matching task's state back into that
dataset's task registry (artifacts/exports/<dataset_id>/export_manifest.json,
see src/gee_task_registry.py) -- this is what lets
scripts/verify_gate_c.py check completion/failure status offline, without
re-polling Earth Engine itself.
"""

import os
import sys

import ee

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gee_task_registry import load_registry, update_task_completion

PROJECT = "decent-being-438620-b7"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="List/refresh Earth Engine export task status")
    parser.add_argument(
        "--dataset-id",
        help="If given, also write matching tasks' status into "
        "artifacts/exports/<dataset_id>/export_manifest.json.",
    )
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    tasks = ee.batch.Task.list()
    if not tasks:
        print("No tasks found.")
        return

    registry_task_ids = set()
    if args.dataset_id:
        registry = load_registry(args.dataset_id)
        registry_task_ids = {t["task_id"] for t in registry["tasks"]}
        if not registry_task_ids:
            print(
                f"WARNING: no tasks registered yet for {args.dataset_id!r} in "
                f"artifacts/exports/{args.dataset_id}/export_manifest.json -- "
                "nothing to update. Submit via scripts/gee_export_chips.py first."
            )

    for t in tasks[:20]:
        status = t.status()
        state = status.get("state")
        desc = status.get("description")
        err = status.get("error_message", "")
        line = f"{status['id']}  {state:10s}  {desc}"
        if err:
            line += f"  ERROR: {err}"

        if status["id"] in registry_task_ids:
            update_task_completion(args.dataset_id, status["id"], state, err or None)
            line += "  [registry updated]"

        print(line)


if __name__ == "__main__":
    main()
