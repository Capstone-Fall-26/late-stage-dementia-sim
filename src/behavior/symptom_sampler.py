"""Samples per-turn behavioral constraints from literature-derived priors.

`sample_constraints` is the single entry point: given the patient's current
disease stage and emotional/distress state, it draws from the statistical
priors in `config/statistical_priors.yaml` (loaded into a `StatisticalPriors`
object) to produce a `SymptomConstraints` instance describing exactly how
the *next* patient utterance should be shaped:

- `max_words`: a hard ceiling sampled from a truncated-normal verbal-fluency
  prior.
- `nonverbal_cue`: an optional stage-appropriate non-verbal action string
  (e.g. "*gazes blankly*"), sampled from the cue bank when it fires.
- `memory_fog`: whether the patient should present as unable to retrieve
  the requested memory at all this turn.
- `factual_inversion`: whether a factual misattribution should occur.
- `anomia`: whether word-finding difficulty / substitution should occur.
- `distress_escalation`: whether this turn should escalate distress.

The current distress state modulates the base stage priors (e.g. an
agitated patient is more likely to produce non-verbal cues and escalate;
a withdrawn patient produces far fewer words).
"""

from __future__ import annotations

import numpy as np

from core.models import DistressState, NormalPrior, Stage, StatisticalPriors, SymptomConstraints

# Multiplicative modifiers applied to base stage priors, keyed by distress
# state. All probabilities are clipped to [0, 1] after modification.
_DISTRESS_MODIFIERS: dict[DistressState, dict[str, float]] = {
    DistressState.CALM: {
        "word_count_mult": 1.0,
        "nonverbal_mult": 1.0,
        "inversion_mult": 1.0,
        "anomia_mult": 1.0,
        "escalation_mult": 1.0,
    },
    DistressState.CONFUSED: {
        "word_count_mult": 0.9,
        "nonverbal_mult": 1.15,
        "inversion_mult": 1.4,
        "anomia_mult": 1.3,
        "escalation_mult": 1.2,
    },
    DistressState.AGITATED: {
        "word_count_mult": 0.7,
        "nonverbal_mult": 1.6,
        "inversion_mult": 1.1,
        "anomia_mult": 1.1,
        "escalation_mult": 1.8,
    },
    DistressState.WITHDRAWN: {
        "word_count_mult": 0.4,
        "nonverbal_mult": 1.7,
        "inversion_mult": 0.9,
        "anomia_mult": 1.0,
        "escalation_mult": 0.6,
    },
}


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _sample_truncated_normal(prior: NormalPrior, mult: float, rng: np.random.Generator) -> int:
    """Sample a word count from a truncated-normal prior, scaled by `mult`."""
    from scipy.stats import truncnorm

    mean = prior.mean * mult
    std = max(prior.std * mult, 1e-6)
    a = (prior.min - mean) / std
    b = (prior.max - mean) / std
    value = truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)
    return max(0, round(float(value)))


def sample_constraints(
    stage: Stage,
    distress: DistressState,
    priors: StatisticalPriors,
    rng: np.random.Generator | None = None,
) -> SymptomConstraints:
    """Sample a full set of behavioral constraints for the next patient turn."""
    rng = rng or np.random.default_rng()
    profile = priors.get_stage_profile(stage)
    mods = _DISTRESS_MODIFIERS[distress]

    max_words = _sample_truncated_normal(
        profile.verbal_fluency_mean_wpm, mods["word_count_mult"], rng
    )

    nonverbal_p = _clip01(profile.non_verbal_cue_probability.value * mods["nonverbal_mult"])
    nonverbal_cue: str | None = None
    if rng.random() < nonverbal_p:
        bank = priors.get_nonverbal_bank(stage)
        if bank:
            nonverbal_cue = str(rng.choice(bank))

    inversion_p = _clip01(profile.factual_inversion_rate.value * mods["inversion_mult"])
    factual_inversion = bool(rng.random() < inversion_p)

    anomia_p = _clip01(profile.anomia_rate.value * mods["anomia_mult"])
    anomia = bool(rng.random() < anomia_p)

    # Memory fog (total retrieval failure) scales with anomia severity and
    # stage: later stages are more likely to present total retrieval fog
    # rather than a partial word-finding substitution.
    stage_severity = 0.4 + 0.2 * (stage.value - Stage.FIVE.value)
    fog_p = _clip01(anomia_p * stage_severity)
    memory_fog = bool(rng.random() < fog_p)

    escalation_p = _clip01(profile.distress_escalation_prob.value * mods["escalation_mult"])
    distress_escalation = bool(rng.random() < escalation_p)

    return SymptomConstraints(
        stage=stage,
        distress_state=distress,
        max_words=max_words,
        nonverbal_cue=nonverbal_cue,
        memory_fog=memory_fog,
        factual_inversion=factual_inversion,
        anomia=anomia,
        distress_escalation=distress_escalation,
    )
