#!/usr/bin/env python
"""
List recent Earth Engine batch export tasks and their status.

    conda activate deforest
    python scripts/gee_check_tasks.py
"""

import ee

PROJECT = "decent-being-438620-b7"


def main():
    ee.Initialize(project=PROJECT)
    tasks = ee.batch.Task.list()
    if not tasks:
        print("No tasks found.")
        return

    for t in tasks[:20]:
        status = t.status()
        state = status.get("state")
        desc = status.get("description")
        err = status.get("error_message", "")
        line = f"{status['id']}  {state:10s}  {desc}"
        if err:
            line += f"  ERROR: {err}"
        print(line)


if __name__ == "__main__":
    main()
