"""Interactive CLI for the late-stage dementia patient simulation.

Usage:

    dementia-sim chat --backend mock
    dementia-sim chat --backend ollama --model llama3.2:3b
    dementia-sim chat --backend gemini --model gemini-flash-lite-latest
    dementia-sim chat --backend azure --model my-gpt4o-mini-deployment
    dementia-sim test-connection --backend ollama
    dementia-sim seed-memory
    dementia-sim recall "tell me about your garden" --stage 7
    dementia-sim add-memory "Susan brought lilies today" --stage 6
    dementia-sim age-memory --stage 7
    dementia-sim memory-stats
    dementia-sim memory-map --stage 7 --query "tell me about your garden"
    dementia-sim reset-memory
    dementia-sim list-profiles
"""

from __future__ import annotations

import json
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from behavior.priors_check import (
    ASSUMPTION,
    CITED,
    PLACEHOLDER,
    UNLABELED,
    assert_no_placeholders,
    prior_sources,
    provenance_label,
)
from core.llm_config import resolve_llm_config
from core.llm_providers import get_llm_provider
from core.models import (
    EngineTurnResult,
    Stage,
    load_patient_profile,
    load_statistical_priors,
)
from core.patient_engine import PatientEngine
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import get_embedder
from memory.encoding import MemoryEncoder
from memory.noise import MemoryNoiseModel, build_confusion_map
from memory.recall import MemoryRecall
from memory.visualize import (
    build_memory_map,
    load_memory_records,
    load_transcript,
    write_memory_map,
)

load_dotenv()

app = typer.Typer(help="Interactive late-stage dementia patient simulation CLI.")
console = Console()

_DEFAULT_PROFILE = Path("config/patient_profiles/stage7_late.json")
_DEFAULT_PRIORS = Path("config/statistical_priors.yaml")
_DEFAULT_DB_PATH = Path("./milvus_patient_memory.db")


def _print_debug_panel(result: EngineTurnResult) -> None:
    table = Table(title="Debug", show_header=False, title_style="bold yellow")
    memory_status = (
        "Fog" if result.constraints.memory_fog else ("Pass" if result.memory_hits else "No hits")
    )
    table.add_row("State", f"[bold]{result.distress_state.value.capitalize()}[/bold]")
    table.add_row("Memory Retrieval", memory_status)
    noisy = sum(1 for h in result.memory_hits if h.noise_type)
    table.add_row("Noisy hits", f"{noisy}/{len(result.memory_hits)}")
    table.add_row("Memory encoded", str(result.memory_encoded))
    table.add_row("Non-verbal sampled", str(bool(result.constraints.nonverbal_cue)))
    table.add_row("Max words", str(result.constraints.max_words))
    table.add_row("Factual inversion", str(result.constraints.factual_inversion))
    table.add_row("Anomia", str(result.constraints.anomia))
    table.add_row("Distress escalation", str(result.constraints.distress_escalation))
    table.add_row("Latency (ms)", f"{result.latency_ms:.1f}")
    console.print(table)


def _save_transcript(transcript: list[dict], patient_id: str, session_start: datetime) -> Path:
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{patient_id}_{session_start.strftime('%Y%m%dT%H%M%S')}.json"
    path = transcripts_dir / filename
    payload = {
        "patient_id": patient_id,
        "session_start": session_start.isoformat(),
        "turns": transcript,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]Transcript saved to {path}[/green]")
    return path


@app.command()
def chat(
    profile: Path = typer.Option(
        _DEFAULT_PROFILE, "--profile", "-p", help="Path to patient profile JSON."
    ),
    priors: Path = typer.Option(
        _DEFAULT_PRIORS, "--priors", help="Path to statistical priors YAML."
    ),
    backend: str = typer.Option("mock", "--backend", "-b", help="ollama | gemini | azure | mock"),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model/deployment name for the chosen backend. Falls back to the "
        "backend's env var (OLLAMA_MODEL, GEMINI_MODEL, AZURE_OPENAI_DEPLOYMENT) if omitted.",
    ),
    db_path: Path = typer.Option(
        _DEFAULT_DB_PATH, "--db-path", help="Milvus Lite database file path."
    ),
    embedding_backend: str = typer.Option(
        "hashing", "--embedding-backend", help="hashing | sentence-transformer"
    ),
    seed: int | None = typer.Option(None, "--seed", help="RNG seed for reproducibility."),
    debug: bool = typer.Option(
        True, "--debug/--no-debug", help="Show live debug panel after each turn."
    ),
    remember: bool = typer.Option(
        True,
        "--remember/--no-remember",
        help="Encode what the trainee says as new (noisy, possibly failed) patient memories.",
    ),
) -> None:
    """Run an interactive trainee <-> simulated-patient chat session."""
    cfg = resolve_llm_config(backend, model)

    try:
        engine = PatientEngine.from_config(
            profile_path=profile,
            priors_path=priors,
            backend=cfg.backend,
            model=cfg.model,
            api_key=cfg.api_key,
            host=cfg.host,
            endpoint=cfg.endpoint,
            api_version=cfg.api_version,
            db_path=db_path,
            embedding_backend=embedding_backend,
            seed=seed,
            remember_conversation=remember,
        )
    except Exception as exc:
        console.print(f"[red]Failed to initialize patient engine: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"Simulating {engine.profile.preferred_name} -- Stage {engine.profile.stage.value} dementia\n"
            "Type 'quit' or press Ctrl+C to end the session and save the transcript.",
            title="Dementia Simulation",
            style="bold blue",
        )
    )

    transcript: list[dict] = []
    session_start = datetime.now()

    try:
        while True:
            user_input = Prompt.ask("[bold cyan]Trainee[/bold cyan]")
            if user_input.strip().lower() in {"quit", "exit"}:
                break

            result = engine.respond(user_input)
            console.print(
                Panel(result.response_text, title=engine.profile.preferred_name, style="magenta")
            )
            if debug:
                _print_debug_panel(result)

            transcript.append(
                {
                    "turn": len(transcript) + 1,
                    "trainee": user_input,
                    "patient": result.response_text,
                    "distress_state": result.distress_state.value,
                    "constraints": result.constraints.model_dump(),
                    "memory_hits": [hit.model_dump() for hit in result.memory_hits],
                    "memory_encoded": result.memory_encoded,
                    "latency_ms": result.latency_ms,
                    "timestamp": result.timestamp,
                }
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")
    finally:
        if transcript:
            _save_transcript(transcript, engine.profile.patient_id, session_start)
        engine.close()


@app.command("test-connection")
def test_connection(
    backend: str = typer.Option("mock", "--backend", "-b", help="ollama | gemini | azure | mock"),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the backend's default model/deployment."
    ),
) -> None:
    """Send a single test prompt to the configured backend and report the result.

    A quick way to confirm credentials and connectivity (Ollama daemon
    reachable, Gemini/Azure API key and endpoint valid) before starting a
    full chat session.
    """
    cfg = resolve_llm_config(backend, model)
    console.print(f"Testing [bold]{cfg.backend}[/bold] backend (model: {cfg.model!r})...")

    try:
        provider = get_llm_provider(
            backend=cfg.backend,
            model=cfg.model,
            api_key=cfg.api_key,
            host=cfg.host,
            endpoint=cfg.endpoint,
            api_version=cfg.api_version,
        )
        api_mode = getattr(provider, "api_mode", None)
        if api_mode:
            console.print(f"Detected Azure API shape: [bold]{api_mode}[/bold]")
        start = time.perf_counter()
        reply = provider.generate(
            system_prompt="You are a connectivity check. Reply with only the word OK.",
            messages=[{"role": "user", "content": "Reply with only the word OK."}],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Connected[/green] in {elapsed_ms:.0f}ms")
    console.print(f"Response: {reply.strip()!r}")


@app.command("seed-memory")
def seed_memory(
    profile: Path = typer.Option(_DEFAULT_PROFILE, "--profile", "-p"),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
) -> None:
    """Seed episodic/biographical memory collections from a patient profile."""
    profile_obj = load_patient_profile(profile)
    embedder = get_embedder(embedding_backend)
    client = PatientMemoryClient(db_path=db_path)
    counts = client.seed_from_profile(profile_obj, embedder)
    client.close()
    console.print(f"[green]Seeded memory for {profile_obj.name}: {counts}[/green]")


@app.command("priors-status")
def priors_status(
    priors: Path = typer.Option(_DEFAULT_PRIORS, "--priors", help="Statistical priors YAML."),
) -> None:
    """Show which behavior priors are literature-cited and which are not.

    Exits 1 while any prior is still an ASSUMPTION or PLACEHOLDER, so it can gate
    runs that are meant to produce research output.
    """
    priors_obj = load_statistical_priors(priors)
    sources = prior_sources(priors_obj)
    unsourced = set(assert_no_placeholders(priors_obj))

    table = Table(title="Statistical prior provenance")
    table.add_column("Key")
    table.add_column("Status")
    table.add_column("Source", overflow="fold")
    colors = {CITED: "green", ASSUMPTION: "yellow", PLACEHOLDER: "red", UNLABELED: "red"}
    counts = {CITED: 0, ASSUMPTION: 0, PLACEHOLDER: 0, UNLABELED: 0}
    for key, source in sources.items():
        label = provenance_label(source)
        counts[label] += 1
        table.add_row(key, f"[{colors[label]}]{label}[/{colors[label]}]", source)
    console.print(table)
    console.print(
        f"{len(sources)} priors: {counts[CITED]} cited, {counts[ASSUMPTION]} assumption, "
        f"{counts[PLACEHOLDER] + counts[UNLABELED]} placeholder."
    )
    console.print("[dim]distress_transition_matrix has no source field and is an assumption.[/dim]")
    if unsourced:
        console.print(f"[red]{len(unsourced)} prior(s) are not literature-cited.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All priors are literature-cited.[/green]")


def _parse_stage(stage: int) -> Stage:
    try:
        return Stage(stage)
    except ValueError as exc:
        console.print("[red]--stage must be 5, 6, or 7[/red]")
        raise typer.Exit(code=1) from exc


def _open_memory(
    profile: Path, db_path: Path, embedding_backend: str, seed: int | None
) -> tuple[PatientMemoryClient, MemoryRecall, MemoryEncoder]:
    """Open + seed the store and build recall/encoder sharing one noise model."""
    profile_obj = load_patient_profile(profile)
    priors = load_statistical_priors(_DEFAULT_PRIORS)
    embedder = get_embedder(embedding_backend)
    client = PatientMemoryClient(db_path=db_path, dim=embedder.dim)
    client.seed_from_profile(profile_obj, embedder)
    rng = np.random.default_rng(seed)
    noise = MemoryNoiseModel(rng=rng, confusion_map=build_confusion_map(profile_obj))
    recall = MemoryRecall(client, embedder, priors, rng=rng, noise=noise)
    return client, recall, MemoryEncoder(client, embedder, noise)


@app.command("recall")
def recall_cmd(
    query: str = typer.Argument(..., help="Natural-language cue, as a trainee might say it."),
    stage: int = typer.Option(7, "--stage", help="GDS/FAST stage: 5, 6 or 7."),
    collection: str = typer.Option("episodic", "--collection", "-c", help="episodic | facts"),
    top_k: int = typer.Option(3, "--top-k", "-k"),
    profile: Path = typer.Option(_DEFAULT_PROFILE, "--profile", "-p"),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
    seed: int | None = typer.Option(None, "--seed"),
    reinforce: bool = typer.Option(
        False, "--reinforce/--no-reinforce", help="Count this lookup as rehearsal."
    ),
) -> None:
    """Try a degraded memory lookup and show what surfaced (and what was corrupted)."""
    stage_enum = _parse_stage(stage)
    name = FACTS_COLLECTION if collection.startswith("fact") else EPISODIC_COLLECTION
    client, recall, _ = _open_memory(profile, db_path, embedding_backend, seed)
    try:
        hits = recall.recall(query, name, stage_enum, top_k=top_k, reinforce=reinforce)
    finally:
        client.close()

    table = Table(title=f"Recall (stage {stage_enum.value}): {query!r}")
    table.add_column("Score", justify="right")
    table.add_column("Noise")
    table.add_column("Surfaced")
    table.add_column("Original (if altered)")
    for hit in hits:
        table.add_row(
            f"{hit.score:.3f}",
            hit.noise_type or "-",
            hit.text,
            hit.original_text if hit.original_text and hit.original_text != hit.text else "",
        )
    console.print(table)


@app.command("add-memory")
def add_memory(
    text: str = typer.Argument(..., help="Experience to encode into episodic memory."),
    stage: int = typer.Option(7, "--stage", help="GDS/FAST stage: 5, 6 or 7."),
    valence: str = typer.Option("neutral", "--valence", help="positive | neutral | negative"),
    profile: Path = typer.Option(_DEFAULT_PROFILE, "--profile", "-p"),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Encode a new experience; at later stages it may fail to stick or be stored garbled."""
    stage_enum = _parse_stage(stage)
    client, _, encoder = _open_memory(profile, db_path, embedding_backend, seed)
    try:
        result = encoder.encode_episode(text, stage_enum, emotional_valence=valence)
    finally:
        client.close()
    if result is None:
        console.print("[yellow]Encoding failed -- the patient did not retain this.[/yellow]")
    elif result.corrupted:
        console.print(f"[yellow]Stored with corruption:[/yellow] {result.stored_text!r}")
        console.print(f"Original: {result.original_text!r}")
    else:
        console.print(f"[green]Stored intact:[/green] {result.stored_text!r}")


@app.command("age-memory")
def age_memory(
    stage: int = typer.Option(7, "--stage", help="GDS/FAST stage: 5, 6 or 7."),
    passes: int = typer.Option(1, "--passes", min=1, help="Number of decay passes to run."),
    profile: Path = typer.Option(_DEFAULT_PROFILE, "--profile", "-p"),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Permanently corrupt a stage-scaled fraction of stored memories (simulates progression)."""
    stage_enum = _parse_stage(stage)
    client, _, encoder = _open_memory(profile, db_path, embedding_backend, seed)
    try:
        for i in range(passes):
            altered = encoder.decay_pass(stage_enum)
            console.print(f"Pass {i + 1}: altered {altered}")
    finally:
        client.close()
    console.print(
        "[yellow]Memories are now permanently degraded; run `reset-memory` to restore.[/yellow]"
    )


@app.command("memory-stats")
def memory_stats(
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
) -> None:
    """Show row, conversation-sourced and corrupted counts per collection."""
    client = PatientMemoryClient(db_path=db_path, dim=get_embedder(embedding_backend).dim)
    try:
        stats = client.stats()
    finally:
        client.close()
    table = Table(title="Memory stats")
    for col in ("Collection", "Rows", "From conversation", "Corrupted"):
        table.add_column(col)
    for name, row in stats.items():
        table.add_row(name, str(row["rows"]), str(row["from_conversation"]), str(row["corrupted"]))
    console.print(table)


@app.command("memory-map")
def memory_map(
    stage: int = typer.Option(7, "--stage", help="GDS/FAST stage label: 5, 6 or 7."),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    out: Path | None = typer.Option(
        None, "--out", help="Output HTML (default evaluation/plots/memory_map_stage{N}.html)."
    ),
    query: str | None = typer.Option(None, "--query", help="Cue to place on the map."),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
    projection: str = typer.Option("tsne", "--projection", help="tsne | pca"),
    transcript: Path | None = typer.Option(
        None,
        "--transcript",
        help="Saved chat transcript JSON to replay as a step-through communication sequence.",
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the HTML afterwards."),
) -> None:
    """Render a read-only interactive 2D map of memory vectors with degradation drift."""
    stage_enum = _parse_stage(stage)
    if not db_path.exists():
        console.print(f"[red]No memory database at {db_path}; run `seed-memory` first.[/red]")
        raise typer.Exit(code=1)
    if projection not in {"tsne", "pca"}:
        console.print("[red]--projection must be tsne or pca[/red]")
        raise typer.Exit(code=1)
    embedder = get_embedder(embedding_backend)
    client = PatientMemoryClient(db_path=db_path, dim=embedder.dim)
    try:
        df = load_memory_records(client)
    finally:
        client.close()

    if transcript is not None and not transcript.exists():
        console.print(f"[red]Transcript not found: {transcript}[/red]")
        raise typer.Exit(code=1)
    turns = load_transcript(transcript) if transcript else None
    query_vector = embedder.embed(query) if query else None
    fig = build_memory_map(
        df,
        embedder,
        query_vector=query_vector,
        query_label=query,
        stage=stage_enum.value,
        projection=projection,
        turns=turns,
    )
    path = write_memory_map(
        fig,
        out or Path(f"evaluation/plots/memory_map_stage{stage_enum.value}.html"),
        df=df,
        stage=stage_enum.value,
        query_label=query,
    )
    corrupted = int(df["corrupted"].sum()) if len(df) else 0
    facts = int((df["kind"] == "fact").sum()) if len(df) else 0
    console.print(f"[green]Wrote {path}[/green]")
    console.print(f"Points: {len(df)}  corrupted: {corrupted}  facts: {facts}")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


@app.command("reset-memory")
def reset_memory(
    profile: Path = typer.Option(_DEFAULT_PROFILE, "--profile", "-p"),
    db_path: Path = typer.Option(_DEFAULT_DB_PATH, "--db-path"),
    embedding_backend: str = typer.Option("hashing", "--embedding-backend"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Wipe all memories (including conversation memories) and re-seed from the profile."""
    if not yes and not typer.confirm(f"Delete all memories in {db_path} and re-seed?"):
        raise typer.Exit()
    profile_obj = load_patient_profile(profile)
    embedder = get_embedder(embedding_backend)
    client = PatientMemoryClient(db_path=db_path, dim=embedder.dim)
    try:
        client.reset()
        counts = client.seed_from_profile(profile_obj, embedder)
    finally:
        client.close()
    console.print(f"[green]Memory reset and re-seeded: {counts}[/green]")


@app.command("list-profiles")
def list_profiles(
    profiles_dir: Path = typer.Option(Path("config/patient_profiles"), "--dir"),
) -> None:
    """List available patient profiles."""
    table = Table(title="Patient Profiles")
    table.add_column("File")
    table.add_column("Patient ID")
    table.add_column("Name")
    table.add_column("Stage")

    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profile_obj = load_patient_profile(path)
            table.add_row(
                path.name, profile_obj.patient_id, profile_obj.name, str(profile_obj.stage.value)
            )
        except Exception as exc:
            table.add_row(path.name, "-", f"[red]invalid: {exc}[/red]", "-")

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
