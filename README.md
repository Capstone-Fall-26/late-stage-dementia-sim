# Late-Stage Dementia Simulation

An open-source research framework exploring whether an LLM can generate
clinically-plausible synthetic dementia behaviors (GDS/FAST stages 5-7) for
caregiver and clinician training, grounded in statistical priors extracted
from academic literature.

## What this is

- A **patient simulation engine** (`src/core/patient_engine.py`) that drives
  an LLM (local Ollama, Google Gemini, or Azure OpenAI) to role-play a
  dementia patient, constrained turn-by-turn by literature-grounded
  statistical priors (`config/statistical_priors.yaml`).
- A **vector memory system** (`src/memory/`) using Milvus Lite to model
  episodic and biographical patient memory, with simulated retrieval decay,
  confabulation, persistent corruption of stored memories, failed/noisy
  encoding of new experiences, and rehearsal. See
  [Patient memory model](#patient-memory-model).
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

**Most numeric priors are still assumptions or placeholders.** Each prior's
`source` field in `config/statistical_priors.yaml` is labelled `CITED`,
`ASSUMPTION` or `PLACEHOLDER`; only one value (stage 5 `anomia_rate`) is
currently `CITED`. See `research/README.md` for the extraction workflow and
`research/extracted/CHANGELOG.md` for what has been done and what is still
missing, before any research claim is made from this framework's output.

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

Try the memory layer directly (no LLM needed):

```bash
python -m cli.app recall "tell me about your garden" --stage 7   # degraded lookup
python -m cli.app add-memory "Susan brought lilies" --stage 6    # may fail / be garbled
python -m cli.app age-memory --stage 7 --passes 2                # permanent decay
python -m cli.app memory-stats
python -m cli.app reset-memory --yes                             # wipe + re-seed
```

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

### 6b. (Optional) Configure Azure OpenAI (Azure AI Foundry)

Create an Azure OpenAI resource and deploy a chat model (e.g.
`gpt-4.1-mini`) in [Azure AI Foundry](https://ai.azure.com), then set
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
(the deployment name, not the base model name), and
`AZURE_OPENAI_API_VERSION` in `.env`. Never commit `.env`.

Azure currently issues two different endpoint shapes depending on how the
resource was created -- a classic per-resource endpoint
(`https://<resource>.openai.azure.com`) and a newer Azure AI Foundry
project endpoint (containing `/openai/v1`, e.g. `.../openai/v1/responses`).
Paste whichever one your resource gives you into `AZURE_OPENAI_ENDPOINT`
as-is; the backend detects which API shape it is and talks to it
correctly either way.

Whichever backend you configure, verify credentials and connectivity
before starting a full session:

```bash
python -m cli.app test-connection --backend ollama
python -m cli.app test-connection --backend gemini
python -m cli.app test-connection --backend azure
```

Each sends one minimal prompt and reports success/failure with latency,
without touching memory or the patient profile.

### 7. Run the interactive CLI

```bash
# Fully offline, deterministic mock backend -- no model required:
python -m cli.app chat --backend mock

# Local Ollama:
python -m cli.app chat --backend ollama --model llama3.2:3b

# Gemini:
python -m cli.app chat --backend gemini --model gemini-flash-lite-latest

# Azure OpenAI (deployment name from .env, e.g. gpt-4.1-mini, or pass --model explicitly):
python -m cli.app chat --backend azure
```

Type `quit` or press `Ctrl+C` to end a session; the full transcript is
saved as JSON under `transcripts/`.

By default `chat` also *encodes what the trainee says as new patient
memories* (`--remember`, on by default). At later stages that encoding often
fails or stores a garbled version, and those memories can surface in later
turns. Pass `--no-remember` for a read-only session. Note that memories
persist in the database file across sessions; use `reset-memory` to start
clean.

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
`get_patient_biographical_fact`, `store_patient_memory`,
`age_patient_memory`, `get_memory_stats`, and `get_patient_profile_summary`
as MCP tools.

## Patient memory model

Two Milvus collections hold the patient's memory (`src/memory/client.py`):
`episodic_memory` (autobiographical episodes with an era and emotional
valence) and `biographical_facts` (discrete facts). The memory layer
simulates impairment through several independent mechanisms:

| Mechanism | Where | Effect |
|---|---|---|
| Era retention (Ribot's law) | `recall.py` | Remote memories (childhood) keep their similarity score; recent ones are attenuated, more steeply at later stages. |
| Time decay + rehearsal | `recall.py` | Conversation-formed memories decay with real elapsed time; successful retrievals increment `access_count` and give a small, capped boost. |
| Cue noise | `recall.py` | The query vector is perturbed, so a good cue can still miss. |
| Retrieval swap | `recall.py` | With stage-dependent probability the top hit is replaced by a neighboring memory (`noise_type="swap"`). |
| Retrieval substitution | `recall.py` / `noise.py` | Names, relations and numbers in the surfaced text are altered (e.g. Susan -> Ellen, 28 -> 31). |
| Encoding failure | `encoding.py` | A new experience is not stored at all, with probability rising from stage 5 to 7. |
| Encoding corruption | `encoding.py` | What is stored is already garbled; the clean text is kept in `original_text`. |
| Decay pass | `encoding.py` | `age-memory` permanently corrupts a stage-scaled fraction of stored records and re-embeds them, so the vectors drift with the content. |

Each returned `MemoryHit` carries `confused`, `noise_type`
(`swap`, `substitution`, `swap+substitution`, `stored_corruption`, or
`None`), and `original_text` when the ground truth is known, so evaluation
can tell what the patient *should* have recalled.

**LLM integration.** `PatientEngine` recalls episodic and factual memories
for every trainee utterance (skipped on a "memory fog" turn) and injects them
into the system prompt, tagging uncertain or hazy ones so the model conveys
them loosely or incorrectly instead of quoting them. The model sees the
*degraded* text, never the clean original. Short-term conversational context
comes from the engine's message history window, independent of the vector
store.

**Reproducibility.** All noise draws from the injected `numpy` RNG
(`--seed`). Seeded profile memories are stored clean; persistent corruption
only occurs through conversation encoding and explicit `age-memory` runs.

**Tuning.** Stage magnitudes live in `NoiseConfig` (`src/memory/noise.py`)
and the retrieval constants at the top of `recall.py`. Like the statistical
priors, they are simulation placeholders, not clinical measurements.

**Embedding dimension.** The store is created with the embedder's dimension.
Opening an existing database with a different embedder raises a clear error;
run `reset-memory` (or use a new `--db-path`) after switching embedders.

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

# 4. Benchmark the memory layer: per-stage top-1 faithful recall and noise rate
#    over every biographical fact (offline, deterministic per --seed):
python evaluation/memory_benchmark.py
```

Results land in `evaluation/results/` (`benchmark_results.{json,csv}`,
`consistency_report.json`, `memory_benchmark.json`) and `evaluation/plots/`
(`word_count_by_stage.png`, `distress_transition_heatmap.png`,
`persona_consistency_vs_turn_depth.png`).

## Project layout

```
config/                      Statistical priors + patient profiles
research/                    Literature extraction template/workflow
src/
  core/                       Domain models, LLM providers, patient engine, logging
  memory/                     Milvus Lite client, embeddings, degraded recall,
                              noise model, encoding/decay
  behavior/                   Symptom sampler, distress state machine, MCP server
  cli/                        Typer/rich interactive CLI
evaluation/                   Benchmark runner, consistency checker, chart generator
tests/                        pytest suite
transcripts/                  Saved CLI session transcripts (gitignored)
logs/                         Rotating loguru logs + prompt traces (gitignored)
```

## Contributing literature-grounded priors

Each numeric prior in `config/statistical_priors.yaml` carries a
`CITED`, `ASSUMPTION` or `PLACEHOLDER` label in its `source` field. See
`research/README.md` for the extraction template and workflow to promote
assumptions and placeholders to cited values; `tests/test_priors_provenance.py`
enforces that a `CITED` value has extraction notes in `research/extracted/`.
