from pathlib import Path

import pytest

from mammography.labels import birads_to_class, infer_birads, infer_patient_id


@pytest.mark.parametrize(
    ("birads", "expected"),
    [
        ("1", "SINIF1"),
        (2, "SINIF1"),
        ("4A", "SINIF2"),
        ("BI-RADS 4b", "SINIF3"),
        ("4c", "SINIF3"),
        (6, "SINIF3"),
    ],
)
def test_birads_to_class(birads, expected):
    assert birads_to_class(birads) == expected


def test_infer_birads_from_parent_folder():
    path = Path("data/raw/BIRADS4a/SYNTHETIC_SAMPLE.png")
    assert infer_birads(path) == "4a"


def test_infer_patient_id_from_inbreast_filename():
    path = Path("SYNTHETIC_PATIENT_MG_L_CC_ANON.png")
    assert infer_patient_id(path) == "SYNTHETIC"
