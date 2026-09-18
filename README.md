# Late-Stage Dementia Simulation

An open-source research framework exploring whether an LLM can generate
clinically-plausible synthetic dementia behaviors (GDS/FAST stages 5-7) for
caregiver and clinician training, grounded in statistical priors extracted
from academic literature.

## What this is

- A **patient simulation engine** (`src/core/patient_engine.py`) that drives
  an LLM (local Ollama or Google Gemini) to role-play a dementia patient,
  constrained turn-by-turn by literature-grounded statistical priors
  (`config/statistical_priors.yaml`).
- A **vector memory system** (`src/memory/`) using Milvus Lite to model
  episodic and biographical patient memory, with simulated retrieval decay
  and confabulation (noise injection).
- A **behavioral state machine** (`src/behavior/`) sampling per-turn
  symptom constraints (word limits, non-verbal cues, anomia, factual
  inversion) and tracking a coarse emotional/distress state.
- An **MCP tool server** (`src/behavior/mcp_server.py`) exposing all of the
  above as tools for any MCP-compatible agent framework.
- An **interactive CLI** (`src/cli/app.py`) for live trainee <-> patient
  chat sessions with a live debug panel and JSON transcript logging.
- An **evaluation suite** (`evaluation/`) that runs 20 fixed multi-turn
  benchmark dialogues, checks factual consistency, and generates
  publication-ready charts.

**Every numeric prior is currently a placeholder.** See
`research/README.md` for the literature-extraction workflow used to
replace placeholders with real, cited values before any research claim is
made from this framework's output.

## Quickstart

### 1. Install Miniforge3

If you don't already have a conda distribution, install
[Miniforge3](https://github.com/conda-forge/miniforge) (a minimal,
community-maintained conda installer that defaults to the conda-forge
channel):

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Restart your shell after installation so `conda` is on your `PATH`.

### 2. Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate dementia-sim
```

If you already have an environment named `dementia-sim`, update it instead:

```bash
conda env update -n dementia-sim -f environment.yml
conda activate dementia-sim
```

### 3. Install pre-commit hooks

Immediately after activating the environment, install the git hooks so
`detect-secrets`, `ruff`, and `mypy` run automatically before every commit:

```bash
pre-commit install
```

The repository already ships a `.secrets.baseline` (see
`.pre-commit-config.yaml`). If `detect-secrets` flags a new finding you've
verified is safe, update the baseline with:

```bash
detect-secrets scan --baseline .secrets.baseline
```

You can run all hooks manually at any time with:

```bash
pre-commit run --all-files
```

This project intentionally does not use a cloud CI service. `pre-commit`
running locally (detect-secrets + ruff + mypy) plus `pytest` locally is the
whole "CI" story -- see [Running checks locally](#running-checks-locally)
below.

### 4. Initialize Milvus Lite memory

Milvus Lite needs no separate server -- it's a local embedded database file
created on first use. Seed it from the default patient profile:

```bash
python -m cli.app seed-memory
```

This creates `./milvus_patient_memory.db` and populates the
`episodic_memory` and `biographical_facts` collections from
`config/patient_profiles/stage7_late.json`.

### 5. (Optional) Pull an Ollama model for local inference

Install [Ollama](https://ollama.com), then pull a small local model:

```bash
ollama pull llama3.2:3b
# or
ollama pull mistral
```

### 6. (Optional) Configure Gemini

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (get one at
https://aistudio.google.com/apikey). Never commit `.env`.

```bash
cp .env.example .env
```

### 7. Run the interactive CLI

```bash
# Fully offline, deterministic mock backend -- no model required:
python -m cli.app chat --backend mock

# Local Ollama:
python -m cli.app chat --backend ollama --model llama3.2:3b

# Gemini:
python -m cli.app chat --backend gemini --model gemini-flash-lite-latest
```

Type `quit` or press `Ctrl+C` to end a session; the full transcript is
saved as JSON under `transcripts/`.

Other CLI commands:

```bash
python -m cli.app list-profiles
python -m cli.app seed-memory --profile config/patient_profiles/stage7_late.json
```

### 8. Run the MCP tool server

```bash
python -m behavior.mcp_server
```

Exposes `sample_symptom_constraints`, `get_distress_state`,
`transition_distress_state`, `recall_patient_memory`,
`get_patient_biographical_fact`, and `get_patient_profile_summary` as MCP
tools.

## Running checks locally

Run the full local check suite (this is the project's "CI"):

```bash
pre-commit run --all-files   # detect-secrets, ruff, mypy
pytest                        # unit + integration tests, with coverage
```

## Running the evaluation suite

```bash
# 1. Run 20 fixed benchmark dialogues (10 validation-style, 10
#    pressuring-style) across stages 5, 6, and 7:
python evaluation/run_benchmarks.py --backend mock --model mock

# 2. Check factual/persona consistency against a 90% retention target:
python evaluation/consistency_checker.py

# 3. Generate publication-ready charts into evaluation/plots/:
python evaluation/generate_charts.py
```

Results land in `evaluation/results/` (`benchmark_results.{json,csv}`,
`consistency_report.json`) and `evaluation/plots/`
(`word_count_by_stage.png`, `distress_transition_heatmap.png`,
`persona_consistency_vs_turn_depth.png`).

## Project layout

```
config/                      Statistical priors + patient profiles
research/                    Literature extraction template/workflow
src/
  core/                       Domain models, LLM providers, patient engine, logging
  memory/                     Milvus Lite client, embeddings, degraded recall
  behavior/                   Symptom sampler, distress state machine, MCP server
  cli/                        Typer/rich interactive CLI
evaluation/                   Benchmark runner, consistency checker, chart generator
tests/                        pytest suite
transcripts/                  Saved CLI session transcripts (gitignored)
logs/                         Rotating loguru logs + prompt traces (gitignored)
```

## Contributing literature-grounded priors

All numeric priors in `config/statistical_priors.yaml` are placeholders.
See `research/README.md` for the extraction template and workflow to
replace them with values sourced from real published studies.
