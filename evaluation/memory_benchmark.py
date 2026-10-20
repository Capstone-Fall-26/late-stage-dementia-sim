"""Memory-layer benchmark: how well does degraded recall retrieve ground truth?

For each GDS/FAST stage, every biographical fact in the patient profile is
queried with a paraphrase-free cue (the fact's own text, as a trainee might
ask about it) many times. We record, per stage:

- ``top1_faithful``: fraction of queries whose top hit is the intended fact
  and was neither swapped nor altered at retrieval time.
- ``noise_rate``: fraction of queries whose top hit carries any noise flag.
- ``mean_top1_score``: mean attenuated similarity of the top hit.

Results go to ``evaluation/results/memory_benchmark.json``. Uses a temporary
Milvus Lite database and the offline hashing embedder, so it is
deterministic for a given ``--seed`` and needs no network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from core.models import Stage, load_patient_profile, load_statistical_priors
from memory.client import FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import HashingEmbedder
from memory.noise import MemoryNoiseModel, build_confusion_map
from memory.recall import MemoryRecall

console = Console()
app = typer.Typer(add_completion=False)


def run_memory_benchmark(
    profile_path: Path, priors_path: Path, trials: int, seed: int
) -> dict[str, dict[str, float]]:
    profile = load_patient_profile(profile_path)
    priors = load_statistical_priors(priors_path)
    embedder = HashingEmbedder()
    results: dict[str, dict[str, float]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        client = PatientMemoryClient(db_path=Path(tmp_dir) / "memory_benchmark.db")
        client.seed_from_profile(profile, embedder)
        for stage in Stage:
            rng = np.random.default_rng(seed + stage.value)
            noise = MemoryNoiseModel(rng=rng, confusion_map=build_confusion_map(profile))
            recall = MemoryRecall(client, embedder, priors, rng=rng, noise=noise)
            faithful = noisy = 0
            scores: list[float] = []
            total = 0
            for fact in profile.biographical_facts:
                for _ in range(trials):
                    top = recall.recall(
                        fact.fact, FACTS_COLLECTION, stage, top_k=3, reinforce=False
                    )[0]
                    total += 1
                    scores.append(top.score)
                    noisy += top.noise_type is not None
                    faithful += top.noise_type is None and top.text == fact.fact
            results[f"stage_{stage.value}"] = {
                "queries": float(total),
                "top1_faithful": faithful / total,
                "noise_rate": noisy / total,
                "mean_top1_score": float(np.mean(scores)),
            }
        client.close()
    return results


@app.command()
def main(
    profile: Path = typer.Option(Path("config/patient_profiles/stage7_late.json")),
    priors: Path = typer.Option(Path("config/statistical_priors.yaml")),
    trials: int = typer.Option(50, help="Queries per fact per stage."),
    seed: int = typer.Option(0),
    output: Path = typer.Option(Path("evaluation/results/memory_benchmark.json")),
) -> None:
    results = run_memory_benchmark(profile, priors, trials, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table = Table(title="Memory benchmark")
    for col in ("Stage", "Queries", "Top-1 faithful", "Noise rate", "Mean top-1 score"):
        table.add_column(col)
    for stage, row in results.items():
        table.add_row(
            stage,
            str(int(row["queries"])),
            f"{row['top1_faithful']:.2f}",
            f"{row['noise_rate']:.2f}",
            f"{row['mean_top1_score']:.3f}",
        )
    console.print(table)
    console.print(f"Wrote {output}")


if __name__ == "__main__":
    app()
