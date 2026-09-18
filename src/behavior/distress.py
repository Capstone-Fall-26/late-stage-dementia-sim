"""Patient distress/emotional state machine.

Tracks a coarse emotional state (`calm`, `confused`, `agitated`,
`withdrawn`) that evolves turn-by-turn based on:

1. A base row-stochastic transition matrix (from
   `config/statistical_priors.yaml`), representing the natural drift of
   emotional state independent of trainee behavior.
2. Lexical heuristics detecting whether the trainee's utterance touched a
   patient-specific sensitive topic/trigger, used *pressuring* language
   (quizzing, correcting, insisting), or used *redirection* language
   (validation, gentle topic shift) -- these heuristics reweight the
   transition probabilities before sampling the next state.

This is deliberately a simple lexical heuristic, not a clinical NLP model;
see the `redirection_keywords`/`pressuring_keywords` comment in
`statistical_priors.yaml`.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

from core.models import DistressState, PatientProfile, StatisticalPriors

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "that",
    "this",
    "it",
    "her",
    "his",
    "she",
    "he",
    "when",
    "asks",
    "asking",
    "being",
    "with",
    "about",
    "you",
    "your",
    "if",
    "as",
    "at",
    "by",
}
_TOKEN_RE = re.compile(r"[a-z']+")
_OVERLAP_THRESHOLD = 0.34


class TriggerAssessment(BaseModel):
    """Lexical assessment of a single trainee utterance."""

    sensitive_topic_hit: bool
    matched_trigger: str | None = None
    redirection_detected: bool
    pressuring_detected: bool


def _tokenize(text: str) -> set[str]:
    return {w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def evaluate_trigger(
    user_input: str, profile: PatientProfile, priors: StatisticalPriors
) -> TriggerAssessment:
    """Assess a trainee utterance for sensitive-topic hits and tone heuristics."""
    input_tokens = _tokenize(user_input)
    lowered = user_input.lower()

    matched_trigger: str | None = None
    sensitive_hit = False
    for trigger in [*profile.distress_triggers, *profile.sensitive_topics]:
        trigger_tokens = _tokenize(trigger)
        if not trigger_tokens:
            continue
        overlap = len(trigger_tokens & input_tokens) / len(trigger_tokens)
        if overlap >= _OVERLAP_THRESHOLD:
            sensitive_hit = True
            matched_trigger = trigger
            break

    redirection_detected = any(kw in lowered for kw in priors.redirection_keywords)
    pressuring_detected = any(kw in lowered for kw in priors.pressuring_keywords)

    return TriggerAssessment(
        sensitive_topic_hit=sensitive_hit,
        matched_trigger=matched_trigger,
        redirection_detected=redirection_detected,
        pressuring_detected=pressuring_detected,
    )


class DistressTracker:
    """Stateful distress/emotional state machine for a single patient session."""

    _STATES: list[DistressState] = [
        DistressState.CALM,
        DistressState.CONFUSED,
        DistressState.AGITATED,
        DistressState.WITHDRAWN,
    ]

    def __init__(
        self,
        priors: StatisticalPriors,
        initial_state: DistressState = DistressState.CALM,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.priors = priors
        self.state = initial_state
        self.rng = rng or np.random.default_rng()
        self.history: list[DistressState] = [initial_state]

    def reset(self, state: DistressState = DistressState.CALM) -> None:
        self.state = state
        self.history = [state]

    def _reweight(
        self, base_row: dict[DistressState, float], assessment: TriggerAssessment
    ) -> dict[DistressState, float]:
        weights = dict(base_row)

        if assessment.sensitive_topic_hit or assessment.pressuring_detected:
            weights[DistressState.AGITATED] *= 1.6
            weights[DistressState.WITHDRAWN] *= 1.3
            weights[DistressState.CALM] *= 0.5

        if assessment.redirection_detected and not assessment.pressuring_detected:
            weights[DistressState.CALM] *= 1.5
            weights[DistressState.AGITATED] *= 0.6
            weights[DistressState.WITHDRAWN] *= 0.7

        total = sum(weights.values())
        if total <= 0:
            return dict.fromkeys(self._STATES, 1.0 / len(self._STATES))
        return {state: w / total for state, w in weights.items()}

    def transition(self, assessment: TriggerAssessment) -> DistressState:
        """Sample and apply the next distress state given a trigger assessment."""
        base_row = self.priors.get_transition_row(self.state)
        probs = self._reweight(base_row, assessment)
        p_array = np.array([probs[s] for s in self._STATES])
        p_array = p_array / p_array.sum()
        next_idx = int(self.rng.choice(len(self._STATES), p=p_array))
        self.state = self._STATES[next_idx]
        self.history.append(self.state)
        return self.state
