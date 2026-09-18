"""Publication-ready chart generator for the research paper.

Reads `evaluation/results/benchmark_results.json` (produced by
`run_benchmarks.py`) and writes three PNG figures to `evaluation/plots/`:

1. `word_count_by_stage.png` -- word count distribution per GDS/FAST stage
   (violin + boxplot overlay), split by trainee style.
2. `distress_transition_heatmap.png` -- empirical distress-state transition
   frequencies observed across all benchmark dialogues.
3. `persona_consistency_vs_turn_depth.png` -- factual/persona consistency
   (1 - contradiction rate) as a function of conversation turn depth, with
   a bootstrapped confidence band, split by trainee style.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

from core.models import load_patient_profile
from evaluation.consistency_checker import build_family_lookup, is_contradictory

console = Console()

_DEFAULT_RESULTS = Path("evaluation/results/benchmark_results.json")
_DEFAULT_PROFILE = Path("config/patient_profiles/stage7_late.json")
_DEFAULT_PLOTS_DIR = Path("evaluation/plots")

_DISTRESS_ORDER = ["calm", "confused", "agitated", "withdrawn"]
_PALETTE = "colorblind"


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", palette=_PALETTE, font_scale=1.05)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"


def load_dataframe(results_path: Path = _DEFAULT_RESULTS) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {results_path}. Run `python evaluation/run_benchmarks.py` first."
        )
    records = json.loads(results_path.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def plot_word_count_by_stage(df: pd.DataFrame, output_dir: Path) -> Path:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.violinplot(
        data=df, x="stage", y="word_count", hue="style", split=True, inner=None, ax=ax, alpha=0.6
    )
    sns.boxplot(
        data=df,
        x="stage",
        y="word_count",
        hue="style",
        ax=ax,
        width=0.15,
        showcaps=True,
        boxprops={"zorder": 3},
        whiskerprops={"zorder": 3},
        medianprops={"color": "black", "zorder": 4},
        fliersize=2,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Trainee style", loc="upper right")
    ax.set_title("Word Count Distribution by GDS/FAST Stage")
    ax.set_xlabel("GDS/FAST Stage")
    ax.set_ylabel("Words per Turn")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "word_count_by_stage.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_distress_transition_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    _apply_style()
    sorted_df = df.sort_values(["dialogue_id", "turn_index"]).copy()
    sorted_df["next_state"] = sorted_df.groupby("dialogue_id")["distress_state"].shift(-1)
    transitions = sorted_df.dropna(subset=["next_state"])

    matrix = pd.crosstab(
        transitions["distress_state"], transitions["next_state"], normalize="index"
    )
    matrix = matrix.reindex(index=_DISTRESS_ORDER, columns=_DISTRESS_ORDER, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "P(next state)"},
    )
    ax.set_title("Empirical Distress-State Transition Frequencies")
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "distress_transition_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_persona_consistency_vs_turn_depth(
    df: pd.DataFrame, profile_path: Path, output_dir: Path
) -> Path:
    _apply_style()
    profile = load_patient_profile(profile_path)
    family_lookup = build_family_lookup(profile)

    working = df[~df["factual_inversion"]].copy()
    working["consistent"] = ~working["patient_response"].apply(
        lambda text: is_contradictory(text, family_lookup)
    )
    working["consistent"] = working["consistent"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.lineplot(
        data=working,
        x="turn_index",
        y="consistent",
        hue="style",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Persona Consistency vs. Conversation Turn Depth")
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Fraction Consistent (non-inversion turns)")
    ax.legend(title="Trainee style")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "persona_consistency_vs_turn_depth.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_charts(
    results_path: Path = _DEFAULT_RESULTS,
    profile_path: Path = _DEFAULT_PROFILE,
    output_dir: Path = _DEFAULT_PLOTS_DIR,
) -> list[Path]:
    df = load_dataframe(results_path)
    paths = [
        plot_word_count_by_stage(df, output_dir),
        plot_distress_transition_heatmap(df, output_dir),
        plot_persona_consistency_vs_turn_depth(df, profile_path, output_dir),
    ]
    for path in paths:
        console.print(f"[green]Wrote {path}[/green]")
    return paths


def main(
    results_path: Path = typer.Option(_DEFAULT_RESULTS, "--results"),
    profile_path: Path = typer.Option(_DEFAULT_PROFILE, "--profile"),
    output_dir: Path = typer.Option(_DEFAULT_PLOTS_DIR, "--output-dir"),
) -> None:
    generate_all_charts(results_path=results_path, profile_path=profile_path, output_dir=output_dir)


if __name__ == "__main__":
    typer.run(main)
