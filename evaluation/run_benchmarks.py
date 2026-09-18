"""Automated benchmarking suite.

Runs all 20 fixed trainee dialogues (10 validation-style + 10
pressuring-style, see `trainee_prompts.py`) against the patient engine, once
per simulated GDS/FAST stage (5, 6, 7) -- 60 dialogue runs in total, each
starting from a fresh distress state (calm) with an independent RNG stream
for reproducibility.

Per-turn metrics (word count, distress state, sampled constraints, latency)
are written to `evaluation/results/benchmark_results.{json,csv}` for
consumption by `consistency_checker.py` and `generate_charts.py`.

By default this uses the network-free `mock` LLM backend so it runs
identically in CI and in local development without a live Ollama daemon or
Gemini API key. Pass `--backend ollama` or `--backend gemini` to benchmark a
real model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console

from core.llm_providers import get_llm_provider
from core.models import Stage, load_patient_profile, load_statistical_priors
from core.patient_engine import PatientEngine
from evaluation.trainee_prompts import build_dialogue_sets
from memory.client import PatientMemoryClient
from memory.embeddings import get_embedder

load_dotenv()
console = Console()

_DEFAULT_PROFILE = Path("config/patient_profiles/stage7_late.json")
_DEFAULT_PRIORS = Path("config/statistical_priors.yaml")
_DEFAULT_OUTPUT_DIR = Path("evaluation/results")


def run_dialogue(
    engine: PatientEngine, dialogue_id: str, style: str, turns: list[str]
) -> list[dict]:
    """Run one fixed dialogue script against a fresh PatientEngine, recording metrics."""
    records: list[dict] = []
    for turn_index, utterance in enumerate(turns):
        result = engine.respond(utterance)
        records.append(
            {
                "dialogue_id": dialogue_id,
                "style": style,
                "stage": result.constraints.stage.value,
                "turn_index": turn_index,
                "trainee_utterance": utterance,
                "patient_response": result.response_text,
                "distress_state": result.distress_state.value,
                "word_count": len(result.response_text.split()),
                "nonverbal_used": bool(result.constraints.nonverbal_cue),
                "memory_fog": result.constraints.memory_fog,
                "factual_inversion": result.constraints.factual_inversion,
                "anomia": result.constraints.anomia,
                "distress_escalation": result.constraints.distress_escalation,
                "memory_hits_count": len(result.memory_hits),
                "latency_ms": result.latency_ms,
            }
        )
    return records


def run_benchmarks(
    profile_path: Path = _DEFAULT_PROFILE,
    priors_path: Path = _DEFAULT_PRIORS,
    backend: str = "mock",
    model: str = "mock",
    stages: tuple[int, ...] = (5, 6, 7),
    seed: int = 42,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the full 20-dialogue x N-stage benchmark suite and persist results."""
    base_profile = load_patient_profile(profile_path)
    priors = load_statistical_priors(priors_path)
    embedder = get_embedder("hashing")
    dialogue_sets = build_dialogue_sets()

    llm = get_llm_provider(
        backend=backend,
        model=model,
        api_key=os.environ.get("GEMINI_API_KEY"),
        host=os.environ.get("OLLAMA_HOST"),
    )

    all_records: list[dict] = []
    run_counter = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark_memory.db"
        memory_client = PatientMemoryClient(db_path=db_path)
        memory_client.seed_from_profile(base_profile, embedder)

        for stage_value in stages:
            stage_profile = base_profile.model_copy(update={"stage": Stage(stage_value)})
            for style, dialogues in dialogue_sets.items():
                for idx, turns in enumerate(dialogues):
                    dialogue_id = f"stage{stage_value}_{style}_{idx + 1:02d}"
                    rng = np.random.default_rng(seed + run_counter)
                    engine = PatientEngine(
                        stage_profile, priors, llm, memory_client, embedder, rng=rng
                    )
                    console.print(f"[dim]Running {dialogue_id}...[/dim]")
                    all_records.extend(run_dialogue(engine, dialogue_id, style, turns))
                    run_counter += 1

        memory_client.close()

    df = pd.DataFrame(all_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8"
    )

    console.print(f"[green]Completed {run_counter} dialogue runs, {len(df)} total turns.[/green]")
    console.print(f"[green]Results written to {output_dir}/benchmark_results.{{json,csv}}[/green]")
    return df


def main(
    backend: str = typer.Option("mock", help="ollama | gemini | mock"),
    model: str = typer.Option("mock", help="Model name for the chosen backend."),
    seed: int = typer.Option(42, help="RNG seed for reproducibility."),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT_DIR, help="Directory to write results into."),
) -> None:
    run_benchmarks(backend=backend, model=model, seed=seed, output_dir=output_dir)


if __name__ == "__main__":
    typer.run(main)
