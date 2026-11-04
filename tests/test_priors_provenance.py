"""Every numeric prior must state where its number came from.

The `source` field of each prior starts with CITED, ASSUMPTION or PLACEHOLDER
(see the header of config/statistical_priors.yaml). A CITED prior must name a
study whose extraction notes exist in research/extracted/, so a number can
never claim a citation that was not actually extracted and recorded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.models import Stage, StatisticalPriors

_EXTRACTED = Path(__file__).resolve().parents[1] / "research" / "extracted"
_LABELS = ("CITED", "ASSUMPTION", "PLACEHOLDER")
_FIELDS = (
    "verbal_fluency_mean_wpm",
    "non_verbal_cue_probability",
    "factual_inversion_rate",
    "anomia_rate",
    "distress_escalation_prob",
)


def _all_sources(priors: StatisticalPriors) -> list[tuple[str, str, str]]:
    rows = []
    for stage in (Stage.FIVE, Stage.SIX, Stage.SEVEN):
        profile = priors.get_stage_profile(stage)
        for field in _FIELDS:
            rows.append((stage.yaml_key, field, getattr(profile, field).source))
    return rows


def test_every_source_has_a_provenance_label(priors: StatisticalPriors) -> None:
    for stage, field, source in _all_sources(priors):
        assert source.startswith(tuple(f"{label}:" for label in _LABELS)), (
            f"{stage}.{field} source must start with one of {_LABELS}, got: {source!r}"
        )


def test_cited_sources_have_extraction_notes(priors: StatisticalPriors) -> None:
    cited = [row for row in _all_sources(priors) if row[2].startswith("CITED:")]
    assert cited, "expected at least one CITED prior"
    for stage, field, source in cited:
        match = re.match(r"CITED: ([A-Za-z]+_\d{4})\b", source)
        assert match, f"{stage}.{field}: CITED source must start with <Author>_<year>"
        notes = _EXTRACTED / f"{match.group(1).lower()}.md"
        assert notes.exists(), f"{stage}.{field}: missing extraction notes {notes}"


@pytest.mark.parametrize("stage", [Stage.FIVE, Stage.SIX, Stage.SEVEN])
def test_no_stale_pending_extraction_marker(priors: StatisticalPriors, stage: Stage) -> None:
    profile = priors.get_stage_profile(stage)
    for field in _FIELDS:
        assert "pending literature extraction" not in getattr(profile, field).source
