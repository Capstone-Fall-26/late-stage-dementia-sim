"""Persona/factual consistency checker for benchmark transcripts.

Validates that turns *not* flagged with `factual_inversion` retain the
patient's core family-relationship facts (e.g. never calling anyone other
than Walter "husband"), and reports the resulting retention rate against a
90% target -- the threshold from the project's evaluation goals.

The contradiction heuristic is deliberately narrow and explainable: it only
flags a turn when the response uses a family-relation word (husband,
daughter, son, sister, ...) alongside a *different* family member's name
than the one the profile assigns to that relation. This avoids false
positives from open-ended confabulation (which is expected and desired in
`factual_inversion` turns) while still catching concrete, checkable
identity-attribution errors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import typer
from rich.console import Console
from rich.table import Table

from core.models import PatientProfile, load_patient_profile

console = Console()

_TOKEN_RE = re.compile(r"[a-z']+")

# Canonical relation words -> the PatientProfile.family_members `relation`
# value they correspond to. A response using one of these words alongside a
# wrong family member's first name is flagged as a contradiction.
_RELATION_WORDS: dict[str, str] = {
    "husband": "husband",
    "wife": "husband",
    "daughter": "daughter",
    "son": "son",
    "sister": "sister",
    "brother": "sister",
}

_DEFAULT_RESULTS = Path("evaluation/results/benchmark_results.json")
_DEFAULT_PROFILE = Path("config/patient_profiles/stage7_late.json")
_DEFAULT_REPORT = Path("evaluation/results/consistency_report.json")


def build_family_lookup(profile: PatientProfile) -> dict[str, str]:
    """Map each family member's lowercase first name to their `relation`."""
    return {member.name.split()[0].lower(): member.relation for member in profile.family_members}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def is_contradictory(response_text: str, family_lookup: dict[str, str]) -> bool:
    """Return True if the response misattributes a family relation."""
    tokens = _tokenize(response_text)
    mentioned_names = {name for name in family_lookup if name in tokens}
    if not mentioned_names:
        return False

    for relation_word, canonical_relation in _RELATION_WORDS.items():
        if relation_word not in tokens:
            continue
        correct_names = {name for name, rel in family_lookup.items() if rel == canonical_relation}
        wrong_names = mentioned_names - correct_names
        if wrong_names:
            return True
    return False


def load_records(results_path: Path) -> list[dict]:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {results_path}. Run `python evaluation/run_benchmarks.py` first."
        )
    return json.loads(results_path.read_text(encoding="utf-8"))


def compute_consistency_report(
    records: list[dict], profile: PatientProfile, threshold: float = 0.90
) -> dict:
    """Compute overall and per-style/per-stage factual retention rates."""
    family_lookup = build_family_lookup(profile)

    for record in records:
        record["_contradictory"] = is_contradictory(record["patient_response"], family_lookup)

    non_inversion = [r for r in records if not r.get("factual_inversion")]
    inversion = [r for r in records if r.get("factual_inversion")]

    def _rate(rows: list[dict]) -> float:
        if not rows:
            return 1.0
        return 1.0 - (sum(1 for r in rows if r["_contradictory"]) / len(rows))

    overall_retention = _rate(non_inversion)
    drift_rate = 1.0 - _rate(inversion) if inversion else 0.0

    by_style: dict[str, float] = {}
    by_stage: dict[str, float] = {}
    for style in {r["style"] for r in records}:
        rows = [r for r in non_inversion if r["style"] == style]
        by_style[style] = _rate(rows)
    for stage in {r["stage"] for r in records}:
        rows = [r for r in non_inversion if r["stage"] == stage]
        by_stage[str(stage)] = _rate(rows)

    return {
        "overall_retention_rate": overall_retention,
        "non_inversion_turns": len(non_inversion),
        "inversion_turns": len(inversion),
        "inversion_drift_rate": drift_rate,
        "retention_by_style": by_style,
        "retention_by_stage": by_stage,
        "threshold": threshold,
        "passed": overall_retention >= threshold,
    }


def main(
    results_path: Path = typer.Option(_DEFAULT_RESULTS, "--results"),
    profile_path: Path = typer.Option(_DEFAULT_PROFILE, "--profile"),
    threshold: float = typer.Option(0.90, "--threshold"),
    report_path: Path = typer.Option(_DEFAULT_REPORT, "--report"),
) -> None:
    """Compute and print the factual consistency report; exit 1 if below threshold."""
    profile = load_patient_profile(profile_path)
    records = load_records(results_path)
    report = compute_consistency_report(records, profile, threshold=threshold)

    table = Table(title="Factual Consistency Report")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Overall retention rate", f"{report['overall_retention_rate']:.1%}")
    table.add_row("Threshold", f"{threshold:.1%}")
    table.add_row("Passed", "[green]YES[/green]" if report["passed"] else "[red]NO[/red]")
    table.add_row("Non-inversion turns", str(report["non_inversion_turns"]))
    table.add_row("Inversion turns (expected drift)", str(report["inversion_turns"]))
    table.add_row("Inversion drift rate", f"{report['inversion_drift_rate']:.1%}")
    for style, rate in report["retention_by_style"].items():
        table.add_row(f"Retention ({style})", f"{rate:.1%}")
    for stage, rate in report["retention_by_stage"].items():
        table.add_row(f"Retention (stage {stage})", f"{rate:.1%}")
    console.print(table)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]Report written to {report_path}[/green]")

    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
