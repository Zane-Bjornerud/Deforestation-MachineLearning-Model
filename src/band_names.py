"""Single source of truth for the 18 model input channel names.

Sentinel-2 exports pair a "pre" and "post" composite. When both composites
keep the same band names (B2, B3, ...) and get merged with .addBands(),
Earth Engine's default collision handling silently suffixes the second
occurrence with "_1" instead of raising an error. That's how the legacy
TFRecords in data/*.tfrecord ended up with bare names (B2, B3, ...) for the
pre period and "_1"-suffixed names (B2_1, B3_1, ...) for the post period,
while scripts/gee_export_chips.py explicitly renames to the "_pre"/"_post"
convention used everywhere else in this repo (configs, docs, model card).

legacy_to_canonical() maps the accidental legacy names to the canonical ones.
"""

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_BANDS = ["NDVI", "NBR"]

CANONICAL_BAND_ORDER = (
    [f"{b}_pre" for b in S2_BANDS + INDEX_BANDS]
    + [f"{b}_post" for b in S2_BANDS + INDEX_BANDS]
    + ["dNDVI", "dNBR"]
)

CANONICAL_BAND_SET = set(CANONICAL_BAND_ORDER)

# Bands with no pre/post variant: computed once from the pre/post pair itself.
_CHANGE_BANDS = {"dNDVI", "dNBR"}


def legacy_to_canonical(name: str) -> str:
    """Map a raw legacy TFRecord band name to its canonical _pre/_post name.

    Legacy pre-period bands were exported unsuffixed (e.g. "B3"); legacy
    post-period bands picked up Earth Engine's automatic "_1" collision
    suffix (e.g. "B3_1"). Change bands (dNDVI, dNBR) were never duplicated
    and keep their name as-is.
    """
    if name in _CHANGE_BANDS:
        return name
    if name.endswith("_1"):
        return name[: -len("_1")] + "_post"
    return name + "_pre"


def canonicalize_band_names(names: list[str]) -> list[str]:
    """Rename a list of legacy band names to canonical names, preserving order.

    Raises ValueError if the renamed result isn't exactly the expected set
    of 18 canonical bands (e.g. a band is missing or duplicated).
    """
    renamed = [legacy_to_canonical(name) for name in names]
    if set(renamed) != CANONICAL_BAND_SET or len(renamed) != len(CANONICAL_BAND_ORDER):
        raise ValueError(
            f"Band names {names} did not canonicalize to the expected 18 "
            f"canonical bands. Got: {renamed}"
        )
    return renamed


def to_canonical_band_names(names: list[str]) -> list[str]:
    """Map a list of raw TFRecord band names to canonical names, whether the
    raw names are already canonical (current scripts/gee_export_chips.py
    exports use _pre/_post directly) or legacy bare/_1-suffixed (old raw
    exports under data/raw/existing_gfc_export). Does not reorder -- callers
    that need a fixed channel order must still index CANONICAL_BAND_ORDER
    explicitly; see raw_bands_to_canonical_order.
    """
    if set(names) == CANONICAL_BAND_SET and len(names) == len(CANONICAL_BAND_ORDER):
        return list(names)
    return canonicalize_band_names(names)


def raw_bands_to_canonical_order(raw_band_dict: dict) -> tuple[list, list[str]]:
    """Reindex a {raw_band_name: array} dict into the fixed CANONICAL_BAND_ORDER.

    This is the single point where channel order is decided. Callers must
    build their per-chip array from the returned (arrays, names) rather than
    from whatever order dict/TFRecord iteration produced -- protobuf map and
    Python dict iteration order are not a channel-order guarantee.

    Returns (arrays_in_canonical_order, CANONICAL_BAND_ORDER).
    """
    canonical_keys = to_canonical_band_names(list(raw_band_dict.keys()))
    by_canonical_name = dict(zip(canonical_keys, raw_band_dict.values()))
    ordered_arrays = [by_canonical_name[name] for name in CANONICAL_BAND_ORDER]
    return ordered_arrays, list(CANONICAL_BAND_ORDER)
