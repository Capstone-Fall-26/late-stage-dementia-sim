from __future__ import annotations

import numpy as np

from behavior.distress import DistressTracker, evaluate_trigger
from core.models import DistressState, PatientProfile, StatisticalPriors


def test_evaluate_trigger_detects_sensitive_topic(
    profile: PatientProfile, priors: StatisticalPriors
) -> None:
    assessment = evaluate_trigger("Don't you remember Walter is deceased?", profile, priors)
    assert assessment.sensitive_topic_hit is True
    assert assessment.matched_trigger is not None


def test_evaluate_trigger_detects_redirection(
    profile: PatientProfile, priors: StatisticalPriors
) -> None:
    assessment = evaluate_trigger("Would you like to tell me about your garden?", profile, priors)
    assert assessment.redirection_detected is True
    assert assessment.pressuring_detected is False


def test_evaluate_trigger_detects_pressuring(
    profile: PatientProfile, priors: StatisticalPriors
) -> None:
    assessment = evaluate_trigger("Don't you remember me? Think harder.", profile, priors)
    assert assessment.pressuring_detected is True


def test_transition_matrix_rows_normalized(priors: StatisticalPriors) -> None:
    tracker = DistressTracker(priors, rng=np.random.default_rng(3))
    from behavior.distress import TriggerAssessment

    assessment = TriggerAssessment(
        sensitive_topic_hit=False,
        matched_trigger=None,
        redirection_detected=False,
        pressuring_detected=False,
    )
    row = priors.get_transition_row(tracker.state)
    probs = tracker._reweight(row, assessment)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_pressuring_biases_toward_agitated_or_withdrawn(priors: StatisticalPriors) -> None:
    from behavior.distress import TriggerAssessment

    pressuring = TriggerAssessment(
        sensitive_topic_hit=True,
        matched_trigger="test",
        redirection_detected=False,
        pressuring_detected=True,
    )
    calm_signal = TriggerAssessment(
        sensitive_topic_hit=False,
        matched_trigger=None,
        redirection_detected=False,
        pressuring_detected=False,
    )

    tracker_pressured = DistressTracker(
        priors, initial_state=DistressState.CALM, rng=np.random.default_rng(42)
    )
    tracker_baseline = DistressTracker(
        priors, initial_state=DistressState.CALM, rng=np.random.default_rng(42)
    )

    n = 400
    pressured_agitated_or_withdrawn = 0
    baseline_agitated_or_withdrawn = 0
    for _ in range(n):
        tracker_pressured.reset(DistressState.CALM)
        tracker_baseline.reset(DistressState.CALM)
        if tracker_pressured.transition(pressuring) in (
            DistressState.AGITATED,
            DistressState.WITHDRAWN,
        ):
            pressured_agitated_or_withdrawn += 1
        if tracker_baseline.transition(calm_signal) in (
            DistressState.AGITATED,
            DistressState.WITHDRAWN,
        ):
            baseline_agitated_or_withdrawn += 1

    assert pressured_agitated_or_withdrawn > baseline_agitated_or_withdrawn
