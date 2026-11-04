"""Tests for the prior provenance guard and the `priors-status` CLI command."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from behavior.priors_check import (
    ASSUMPTION,
    CITED,
    PLACEHOLDER,
    UNLABELED,
    assert_no_placeholders,
    prior_sources,
    provenance_label,
)
from cli.app import app
from core.models import StatisticalPriors

_ROOT = Path(__file__).resolve().parents[1]
_PRIORS = _ROOT / "config" / "statistical_priors.yaml"
_LOG = _ROOT / "research" / "extraction_log.csv"
runner = CliRunner()


def _with_sources(priors: StatisticalPriors, **overrides: str) -> StatisticalPriors:
    """Deep copy of `priors` with `stage_N.field` sources replaced."""
    copy = priors.model_copy(deep=True)
    for key, source in overrides.items():
        stage_key, field = key.split("__")
        getattr(copy.stages[stage_key], field).source = source
    return copy


def test_provenance_label() -> None:
    assert provenance_label("CITED: Smith_2020 - x") == CITED
    assert provenance_label("ASSUMPTION: modeling choice") == ASSUMPTION
    assert provenance_label("PLACEHOLDER: none found") == PLACEHOLDER
    assert provenance_label("Smith_2020") == UNLABELED
    assert provenance_label("") == UNLABELED


def test_assert_no_placeholders_flags_unsourced_keys(priors: StatisticalPriors) -> None:
    every_key = set(prior_sources(priors))
    cited = _with_sources(
        priors, **{f"{k.split('.')[0]}__{k.split('.')[1]}": "CITED: A_2020 - x" for k in every_key}
    )
    assert assert_no_placeholders(cited) == []

    mixed = _with_sources(
        cited,
        stage_5__anomia_rate="PLACEHOLDER: none",
        stage_6__anomia_rate="ASSUMPTION: guess",
        stage_7__anomia_rate="",
    )
    assert assert_no_placeholders(mixed) == [
        "stage_5.anomia_rate",
        "stage_6.anomia_rate",
        "stage_7.anomia_rate",
    ]


def test_cli_priors_status_exits_nonzero_while_unsourced() -> None:
    result = runner.invoke(app, ["priors-status", "--priors", str(_PRIORS)])
    assert result.exit_code == 1
    assert "stage_5.anomia_rate" in result.output
    assert "CITED" in result.output


def test_cli_priors_status_exits_zero_when_all_cited(tmp_path: Path) -> None:
    text = _PRIORS.read_text()
    text = re.sub(r'source: "(ASSUMPTION|PLACEHOLDER):', 'source: "CITED:', text)
    path = tmp_path / "priors.yaml"
    path.write_text(text)
    result = runner.invoke(app, ["priors-status", "--priors", str(path)])
    assert result.exit_code == 0
    assert "All priors are literature-cited" in result.output


def test_extraction_log_rows_have_citation_and_link() -> None:
    with _LOG.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for row in rows:
        assert row["source_citation"].strip(), row
        assert row["doi_or_url"].startswith("https://doi.org/"), row


def test_every_cited_key_has_a_value_row_in_the_log(priors: StatisticalPriors) -> None:
    with _LOG.open(newline="") as fh:
        value_keys = {r["key"] for r in csv.DictReader(fh) if r["role"] == "value"}
    for key, source in prior_sources(priors).items():
        if provenance_label(source) == CITED:
            assert key in value_keys, f"{key} is CITED but has no value row in {_LOG.name}"


@pytest.mark.research_gate
def test_research_gate_all_priors_cited(priors: StatisticalPriors) -> None:
    """Fails until every prior is literature-cited. Run with `pytest -m research_gate`."""
    remaining = assert_no_placeholders(priors)
    assert not remaining, f"{len(remaining)} prior(s) not literature-cited: {remaining}"
