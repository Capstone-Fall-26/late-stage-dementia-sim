from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.consistency_checker import (
    build_family_lookup,
    compute_consistency_report,
    is_contradictory,
)

from core.models import PatientProfile


def test_build_family_lookup(profile: PatientProfile) -> None:
    lookup = build_family_lookup(profile)
    assert lookup["walter"] == "husband"
    assert lookup["susan"] == "daughter"
    assert lookup["ellen"] == "sister"


def test_is_contradictory_detects_wrong_husband(profile: PatientProfile) -> None:
    lookup = build_family_lookup(profile)
    assert is_contradictory("My husband Ellen is visiting today.", lookup) is True


def test_is_contradictory_allows_correct_attribution(profile: PatientProfile) -> None:
    lookup = build_family_lookup(profile)
    assert is_contradictory("My husband Walter is visiting today.", lookup) is False


def test_is_contradictory_no_relation_word_is_safe(profile: PatientProfile) -> None:
    lookup = build_family_lookup(profile)
    assert is_contradictory("The roses are blooming in the garden.", lookup) is False


def test_compute_consistency_report_structure(profile: PatientProfile) -> None:
    records = [
        {
            "style": "validation",
            "stage": 7,
            "factual_inversion": False,
            "patient_response": "My husband Walter will be home soon.",
        },
        {
            "style": "pressuring",
            "stage": 7,
            "factual_inversion": True,
            "patient_response": "My husband Ellen is here.",
        },
    ]
    report = compute_consistency_report(records, profile, threshold=0.9)
    assert "overall_retention_rate" in report
    assert report["non_inversion_turns"] == 1
    assert report["inversion_turns"] == 1
    assert report["overall_retention_rate"] == 1.0
