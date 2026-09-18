from __future__ import annotations

from pathlib import Path

import numpy as np

from core.llm_providers import MockProvider
from core.models import PatientProfile, StatisticalPriors
from core.patient_engine import PatientEngine
from memory.client import PatientMemoryClient
from memory.embeddings import HashingEmbedder


def _build_engine(
    tmp_path: Path, profile: PatientProfile, priors: StatisticalPriors
) -> PatientEngine:
    memory_client = PatientMemoryClient(db_path=tmp_path / "engine_test.db")
    embedder = HashingEmbedder()
    engine = PatientEngine(
        profile=profile,
        priors=priors,
        llm=MockProvider(seed=1),
        memory_client=memory_client,
        embedder=embedder,
        rng=np.random.default_rng(2024),
    )
    engine.seed_memory()
    return engine


def test_respond_produces_valid_turn_result(
    tmp_path: Path, profile: PatientProfile, priors: StatisticalPriors
) -> None:
    engine = _build_engine(tmp_path, profile, priors)
    result = engine.respond("Would you like to tell me about your garden?")

    assert isinstance(result.response_text, str)
    assert len(result.response_text) > 0
    assert result.constraints.stage == profile.stage
    assert result.latency_ms >= 0.0

    if result.constraints.max_words > 0:
        assert len(result.response_text.split()) <= result.constraints.max_words + 2
    engine.close()


def test_respond_zero_words_returns_nonverbal_only(
    tmp_path: Path, profile: PatientProfile, priors: StatisticalPriors
) -> None:
    engine = _build_engine(tmp_path, profile, priors)
    from core.models import SymptomConstraints

    constraints = SymptomConstraints(
        stage=profile.stage,
        distress_state=engine.distress.state,
        max_words=0,
        nonverbal_cue="*gazes blankly*",
        memory_fog=False,
        factual_inversion=False,
        anomia=False,
        distress_escalation=False,
    )
    processed = engine._postprocess("some words the model produced", constraints)
    assert processed == "*gazes blankly*"
    engine.close()


def test_multi_turn_session_accumulates_history(
    tmp_path: Path, profile: PatientProfile, priors: StatisticalPriors
) -> None:
    engine = _build_engine(tmp_path, profile, priors)
    engine.respond("Hello Margaret, how are you today?")
    engine.respond("Would you like to talk about your roses?")
    assert len(engine._conversation) == 4
    engine.close()
