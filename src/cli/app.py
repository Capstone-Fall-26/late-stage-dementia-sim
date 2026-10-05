"""Interactive CLI for the late-stage dementia patient simulation.

Usage:

    dementia-sim chat --backend mock
    dementia-sim chat --backend ollama --model llama3.2:3b
    dementia-sim chat --backend gemini --model gemini-flash-lite-latest
    dementia-sim chat --backend azure --model my-gpt4o-mini-deployment
    dementia-sim test-connection --backend ollama
    dementia-sim seed-memory
    dementia-sim list-profiles
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core.llm_config import resolve_llm_config
from core.llm_providers import get_llm_provider
from core.models import EngineTurnResult, load_patient_profile
from core.patient_engine import PatientEngine
from memory.client import PatientMemoryClient
from memory.embeddings import get_embedder

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
