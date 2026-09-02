"""BI-RADS label normalization and project class mapping."""

from __future__ import annotations

import re
from pathlib import Path

CLASS_NAMES = ("SINIF1", "SINIF2", "SINIF3")

BIRADS_TO_CLASS = {
    "1": "SINIF1",
    "2": "SINIF1",
    "3": "SINIF2",
    "4a": "SINIF2",
    "4b": "SINIF3",
    "4c": "SINIF3",
    "5": "SINIF3",
    "6": "SINIF3",
}


def normalize_birads(value: object) -> str:
    """Return a canonical BI-RADS token such as ``4a``."""

    token = str(value).strip().lower().replace("bi-rads", "").replace("birads", "")
    token = re.sub(r"[^0-9abc]", "", token)
    if token.endswith("0") and token[:-1] in {"1", "2", "3", "5", "6"}:
        token = token[:-1]
    if token not in BIRADS_TO_CLASS:
        raise ValueError(f"Unsupported BI-RADS value: {value!r}")
    return token


def birads_to_class(value: object) -> str:
    """Map a BI-RADS value to the project's three-class target."""

    return BIRADS_TO_CLASS[normalize_birads(value)]


def infer_birads(path: str | Path) -> str:
    """Infer a BI-RADS category from a filename or one of its parent folders."""

    candidate = str(path).replace("\\", "/")
    matches = re.findall(r"birads[\s_-]*([1-6](?:[abc])?)", candidate, flags=re.I)
    if not matches:
        raise ValueError(
            f"Could not infer BI-RADS from {path!s}. Include BIRADS1, BIRADS4a, etc. "
            "in the path."
        )
    return normalize_birads(matches[-1])


def infer_patient_id(path: str | Path) -> str:
    """Infer the INbreast patient identifier from a source filename."""

    stem = Path(path).stem
    match = re.search(r"(?<!\d)(\d{6,})(?!\d)", stem)
    if match:
        return match.group(1)
    token = re.split(r"[_\s-]+", stem, maxsplit=1)[0]
    if not token:
        raise ValueError(f"Could not infer patient ID from {path!s}")
    return token

