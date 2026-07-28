"""Dataset contracts: the explicit, versioned specification of the label
definition and processing pipeline for a given dataset.

Two label modes exist in this repo and must never be mixed within one
processed dataset directory:

  hansen_loss   -- Hansen GFC treecover2000/lossyear based. Produced only by
                   src/GFC_process_tfrecords4.py. The active Phase 1 label
                   definition.
  change_based  -- dNBR/dNDVI threshold based. Produced only by
                   src/change_based_processor.py. Legacy-only.

Every dataset used anywhere in the pipeline (export, processing, splitting,
training) has a contract file at configs/datasets/<dataset_id>.yaml. Each
processor validates its own identity against the contract before writing
output; the dataset loader and trainer validate that metadata already on
disk matches the contract before use. A mismatch fails loudly instead of
silently training on the wrong labels.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import pickle

import yaml

LABEL_MODE_HANSEN = "hansen_loss"
LABEL_MODE_CHANGE = "change_based"
VALID_LABEL_MODES = {LABEL_MODE_HANSEN, LABEL_MODE_CHANGE}

# Which processor script + which label_source tag each label_mode requires.
MODE_REQUIREMENTS = {
    LABEL_MODE_HANSEN: {
        "processor": "GFC_process_tfrecords4",
        "label_source": "hansen_gfc",
    },
    LABEL_MODE_CHANGE: {
        "processor": "change_based_processor",
        "label_source": "change_index_threshold",
    },
}

CONTRACT_DIR = Path("configs/datasets")

_REQUIRED_FIELDS = {
    "dataset_id",
    "label_mode",
    "target_year",
    "processor",
    "raw_path",
    "processed_path",
    "label_semantics",
    "label_source",
    "label_contract_version",
}


@dataclass
class DatasetContract:
    dataset_id: str
    label_mode: str
    target_year: Optional[int]
    processor: str
    raw_path: str
    processed_path: str
    label_semantics: str
    label_source: str
    label_contract_version: int
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.label_mode not in VALID_LABEL_MODES:
            raise ValueError(
                f"{self.dataset_id}: label_mode={self.label_mode!r} must be "
                f"one of {sorted(VALID_LABEL_MODES)}"
            )
        expected = MODE_REQUIREMENTS[self.label_mode]
        if self.processor != expected["processor"]:
            raise ValueError(
                f"{self.dataset_id}: label_mode={self.label_mode!r} must be "
                f"produced by processor={expected['processor']!r}, contract "
                f"says processor={self.processor!r}"
            )
        if self.label_source != expected["label_source"]:
            raise ValueError(
                f"{self.dataset_id}: label_mode={self.label_mode!r} must use "
                f"label_source={expected['label_source']!r}, contract says "
                f"label_source={self.label_source!r}"
            )
        if self.label_mode == LABEL_MODE_HANSEN and self.target_year is None:
            raise ValueError(
                f"{self.dataset_id}: label_mode=hansen_loss requires an "
                "explicit target_year"
            )


def load_contract(dataset_id_or_path) -> DatasetContract:
    """Load a dataset contract by dataset_id (resolved under configs/datasets/)
    or by an explicit path to a contract YAML file."""
    path = Path(dataset_id_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONTRACT_DIR / f"{dataset_id_or_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"No dataset contract found at {path}. Every dataset must have "
            "an explicit contract under configs/datasets/ before it can be "
            "exported, processed, split, or trained on."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValueError(
            f"Contract {path} is missing required fields: {sorted(missing)}"
        )

    known = {k: raw[k] for k in _REQUIRED_FIELDS}
    extra = {k: v for k, v in raw.items() if k not in _REQUIRED_FIELDS}
    return DatasetContract(**known, extra=extra)


def validate_processor_identity(contract: DatasetContract, processor_module_name: str):
    """Raise if the running processor doesn't match what the contract requires.

    Call this at the top of each processor's main entry point, e.g.
    validate_processor_identity(contract, "GFC_process_tfrecords4").
    """
    if contract.processor != processor_module_name:
        raise ValueError(
            f"{contract.dataset_id}: contract requires processor "
            f"{contract.processor!r} but {processor_module_name!r} is "
            f"running. label_mode={contract.label_mode!r} must only be "
            f"produced by {contract.processor!r}."
        )


def assert_no_label_source_conflict(output_dir, new_label_source):
    """Refuse to add records with a different label_source to an existing
    processed dataset folder. This is the enforcement mechanism for the rule
    that hansen_loss and change_based labels must never share a folder."""
    metadata_path = Path(output_dir) / "metadata.pkl"
    if not metadata_path.exists():
        return

    with open(metadata_path, "rb") as f:
        existing_metadata = pickle.load(f)

    existing_sources = {
        item.get("label_source", "unknown_legacy") for item in existing_metadata
    }
    existing_sources.discard(new_label_source)

    if existing_sources:
        raise ValueError(
            f"{output_dir} already contains chips with label_source="
            f"{sorted(existing_sources)}, which conflicts with "
            f"{new_label_source!r}. hansen_loss and change_based labels must "
            "never be written to the same processed dataset folder -- pick a "
            "different processed_path."
        )


def validate_metadata_matches_contract(metadata: list, contract: DatasetContract):
    """Validate that processed metadata on disk actually matches its contract.

    Called by the dataset loader (dataset.py) and by train.py/split_data.py
    before use. Fails early and loudly if metadata.pkl was produced by the
    wrong processor / label_source / dataset_id for the label_mode the
    contract declares -- e.g. change_based_processor output loaded against a
    hansen_loss contract, or metadata that predates the contract system.
    """
    if not metadata:
        raise ValueError(
            f"{contract.dataset_id}: metadata is empty, nothing to validate."
        )

    for item in metadata:
        for check_field in ("dataset_id", "label_mode", "label_source"):
            actual = item.get(check_field)
            expected = getattr(contract, check_field)
            if actual != expected:
                raise ValueError(
                    f"Dataset contract violation for {contract.dataset_id!r} "
                    f"(label_mode={contract.label_mode!r}): chip "
                    f"{item.get('chip_id')!r} has {check_field}={actual!r}, "
                    f"expected {expected!r}. This metadata was likely "
                    f"produced by the wrong processor, or predates the "
                    f"contract system -- re-run "
                    f"'python src/{contract.processor}.py --dataset-id "
                    f"{contract.dataset_id}' to regenerate it."
                )


def write_dataset_manifest(
    output_dir, contract: DatasetContract, metadata: list, extra_manifest_fields=None
):
    """Write a dataset-level manifest.json alongside metadata.pkl.

    Chip-level metadata already carries band_names per record; this is the
    dataset-level summary (one file, not one per chip) recording the
    canonical band order plus the contract fields that describe how the
    whole dataset was produced, for a quick sanity check without unpickling
    metadata.pkl.

    extra_manifest_fields: optional dict merged in as-is, e.g. processor-side
    QC counters (shards discovered/processed, records read/skipped/failed)
    that scripts/qc_report.py reads instead of re-parsing raw TFRecords.
    """
    band_names = metadata[0]["band_names"] if metadata else []

    manifest = {
        "dataset_id": contract.dataset_id,
        "label_mode": contract.label_mode,
        "label_source": contract.label_source,
        "label_contract_version": contract.label_contract_version,
        "target_year": contract.target_year,
        "processor": contract.processor,
        "raw_path": contract.raw_path,
        "processed_path": contract.processed_path,
        "label_semantics": contract.label_semantics,
        "chip_count": len(metadata),
        "patch_size": metadata[0]["patch_size"] if metadata else None,
        "band_names": band_names,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **(extra_manifest_fields or {}),
    }

    manifest_path = Path(output_dir) / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
