"""MCP server exposing behavioral/statistical simulation tools to LLM agents.

Run standalone with:

    python -m behavior.mcp_server

This exposes symptom sampling, distress-state tracking, and memory recall
as MCP tools so that any MCP-compatible agent framework (not just this
repo's own `PatientEngine`) can drive the patient simulation while staying
governed by the same statistical priors and patient profile.

Session state (the per-conversation `DistressTracker`) is kept in an
in-process dict keyed by `session_id`, since Milvus Lite and the priors are
process-wide singletons but distress state must be conversation-scoped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastmcp import FastMCP

from behavior.distress import DistressTracker, evaluate_trigger
from behavior.symptom_sampler import sample_constraints
from core.models import DistressState, Stage, load_patient_profile, load_statistical_priors
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import get_embedder
from memory.encoding import MemoryEncoder
from memory.noise import MemoryNoiseModel, build_confusion_map
from memory.recall import MemoryRecall

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PRIORS_PATH = Path(
    os.environ.get("STATISTICAL_PRIORS_PATH", _REPO_ROOT / "config" / "statistical_priors.yaml")
)
_PROFILE_PATH = Path(
    os.environ.get(
        "DEFAULT_PATIENT_PROFILE", _REPO_ROOT / "config" / "patient_profiles" / "stage7_late.json"
    )
)
_MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", str(_REPO_ROOT / "milvus_patient_memory.db"))
_EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "hashing")

mcp = FastMCP(name="dementia-sim-behavior")

_priors = load_statistical_priors(_PRIORS_PATH)
_profile = load_patient_profile(_PROFILE_PATH)
_embedder = get_embedder(_EMBEDDING_BACKEND)
_memory_client = PatientMemoryClient(db_path=_MILVUS_DB_PATH)
_memory_client.seed_from_profile(_profile, _embedder)
_noise = MemoryNoiseModel(confusion_map=build_confusion_map(_profile))
_recall = MemoryRecall(_memory_client, _embedder, _priors, noise=_noise)
_encoder = MemoryEncoder(_memory_client, _embedder, _noise)

_sessions: dict[str, DistressTracker] = {}


def _get_tracker(session_id: str) -> DistressTracker:
    if session_id not in _sessions:
        _sessions[session_id] = DistressTracker(_priors, rng=np.random.default_rng())
    return _sessions[session_id]


@mcp.tool
def sample_symptom_constraints(stage: int, distress_state: str) -> dict:
    """Sample behavioral constraints for the next patient turn.

    Args:
        stage: GDS/FAST stage, one of 5, 6, 7.
        distress_state: one of "calm", "confused", "agitated", "withdrawn".

    Returns:
        A dict with keys: max_words, nonverbal_cue, memory_fog,
        factual_inversion, anomia, distress_escalation.
    """
    constraints = sample_constraints(Stage(stage), DistressState(distress_state), _priors)
    return constraints.model_dump(mode="json")


@mcp.tool
def get_distress_state(session_id: str) -> dict:
    """Get the current distress state for a session, creating it (as calm) if new.

    Args:
        session_id: opaque identifier for the calling conversation/session.
    """
    tracker = _get_tracker(session_id)
    return {
        "session_id": session_id,
        "distress_state": tracker.state.value,
        "history": [s.value for s in tracker.history],
    }


@mcp.tool
def transition_distress_state(session_id: str, user_input: str) -> dict:
    """Evaluate a trainee utterance and transition the session's distress state.

    Args:
        session_id: opaque identifier for the calling conversation/session.
        user_input: the trainee's latest utterance, used to detect
            sensitive-topic hits, pressuring language, and redirection
            language via lexical heuristics.
    """
    tracker = _get_tracker(session_id)
    assessment = evaluate_trigger(user_input, _profile, _priors)
    new_state = tracker.transition(assessment)
    return {
        "session_id": session_id,
        "distress_state": new_state.value,
        "sensitive_topic_hit": assessment.sensitive_topic_hit,
        "matched_trigger": assessment.matched_trigger,
        "redirection_detected": assessment.redirection_detected,
        "pressuring_detected": assessment.pressuring_detected,
    }


@mcp.tool
def recall_patient_memory(
    query: str, stage: int, collection: str = EPISODIC_COLLECTION, top_k: int = 5
) -> dict:
    """Recall cognitively-degraded patient memories relevant to a query.

    Args:
        query: natural-language query, typically the trainee's utterance.
        stage: GDS/FAST stage, one of 5, 6, 7, controlling decay/noise severity.
        collection: "episodic_memory" or "biographical_facts".
        top_k: number of memory hits to return.
    """
    hits = _recall.recall(query, collection, Stage(stage), top_k=top_k)
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}


@mcp.tool
def get_patient_biographical_fact(topic: str, stage: int = 7) -> dict:
    """Recall biographical facts about the patient relevant to a topic.

    Args:
        topic: natural-language topic or question (e.g. "her career").
        stage: GDS/FAST stage controlling recall degradation severity.
    """
    hits = _recall.recall(topic, FACTS_COLLECTION, Stage(stage), top_k=3)
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}


@mcp.tool
def store_patient_memory(text: str, stage: int, emotional_valence: str = "neutral") -> dict:
    """Encode a new experience into the patient's episodic memory.

    At later stages encoding may fail outright (the patient does not retain
    it) or the stored text may be corrupted; the result says which happened.

    Args:
        text: the experience, written from the patient's perspective.
        stage: GDS/FAST stage, one of 5, 6, 7.
        emotional_valence: "positive", "neutral" or "negative".
    """
    result = _encoder.encode_episode(text, Stage(stage), emotional_valence=emotional_valence)
    if result is None:
        return {"stored": False}
    return {
        "stored": True,
        "corrupted": result.corrupted,
        "stored_text": result.stored_text,
        "original_text": result.original_text,
    }


@mcp.tool
def age_patient_memory(stage: int) -> dict:
    """Run one decay pass, permanently corrupting a stage-scaled fraction of memories.

    Args:
        stage: GDS/FAST stage, one of 5, 6, 7.
    """
    return {"altered": _encoder.decay_pass(Stage(stage))}


@mcp.tool
def get_memory_stats() -> dict:
    """Row, conversation-sourced and corrupted counts for each memory collection."""
    return _memory_client.stats()


@mcp.tool
def get_patient_profile_summary() -> dict:
    """Return a non-sensitive summary of the active patient profile."""
    return {
        "patient_id": _profile.patient_id,
        "name": _profile.name,
        "preferred_name": _profile.preferred_name,
        "stage": _profile.stage.value,
        "birth_year": _profile.birth_year,
        "biography": _profile.biography,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
