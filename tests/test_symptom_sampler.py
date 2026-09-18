from __future__ import annotations

import numpy as np

from behavior.symptom_sampler import sample_constraints
from core.models import DistressState, Stage, StatisticalPriors


def test_sample_constraints_basic_invariants(
    priors: StatisticalPriors, rng: np.random.Generator
) -> None:
    constraints = sample_constraints(Stage.SEVEN, DistressState.CALM, priors, rng=rng)
    assert constraints.stage == Stage.SEVEN
    assert constraints.distress_state == DistressState.CALM
    assert constraints.max_words >= 0
    assert isinstance(constraints.factual_inversion, bool)
    assert isinstance(constraints.anomia, bool)
    assert isinstance(constraints.memory_fog, bool)
    if constraints.nonverbal_cue is not None:
        assert isinstance(constraints.nonverbal_cue, str)


def test_later_stage_has_lower_average_word_count(priors: StatisticalPriors) -> None:
    rng5 = np.random.default_rng(0)
    rng7 = np.random.default_rng(0)
    n = 300
    stage5_words = [
        sample_constraints(Stage.FIVE, DistressState.CALM, priors, rng=rng5).max_words
        for _ in range(n)
    ]
    stage7_words = [
        sample_constraints(Stage.SEVEN, DistressState.CALM, priors, rng=rng7).max_words
        for _ in range(n)
    ]
    assert np.mean(stage7_words) < np.mean(stage5_words)


def test_withdrawn_reduces_word_count_relative_to_calm(priors: StatisticalPriors) -> None:
    rng_calm = np.random.default_rng(5)
    rng_withdrawn = np.random.default_rng(5)
    n = 300
    calm_words = [
        sample_constraints(Stage.SIX, DistressState.CALM, priors, rng=rng_calm).max_words
        for _ in range(n)
    ]
    withdrawn_words = [
        sample_constraints(Stage.SIX, DistressState.WITHDRAWN, priors, rng=rng_withdrawn).max_words
        for _ in range(n)
    ]
    assert np.mean(withdrawn_words) < np.mean(calm_words)


def test_agitated_increases_distress_escalation_probability(priors: StatisticalPriors) -> None:
    rng_calm = np.random.default_rng(11)
    rng_agitated = np.random.default_rng(11)
    n = 500
    calm_escalations = sum(
        sample_constraints(Stage.SIX, DistressState.CALM, priors, rng=rng_calm).distress_escalation
        for _ in range(n)
    )
    agitated_escalations = sum(
        sample_constraints(
            Stage.SIX, DistressState.AGITATED, priors, rng=rng_agitated
        ).distress_escalation
        for _ in range(n)
    )
    assert agitated_escalations > calm_escalations
