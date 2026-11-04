"""Reports which statistical priors are literature-sourced and which are not.

Every prior in `config/statistical_priors.yaml` carries a `source` string that
starts with a provenance label:

- ``CITED:``       the value comes from a published study.
- ``ASSUMPTION:``  a modeling choice; the literature supports only its direction
                   or rough scale.
- ``PLACEHOLDER:`` no source found; the number is illustrative.

Only ``CITED`` counts as sourced. ``ASSUMPTION`` values are *not* treated as
sourced, because they have no citation for the number itself. Leaving them out of
the gate keeps an uncited value from passing a research-output check.
"""

from __future__ import annotations

from core.models import BernoulliPrior, NormalPrior, StatisticalPriors

CITED = "CITED"
ASSUMPTION = "ASSUMPTION"
PLACEHOLDER = "PLACEHOLDER"
UNLABELED = "UNLABELED"

_LABELS = (CITED, ASSUMPTION, PLACEHOLDER)

PRIOR_FIELDS = (
    "verbal_fluency_mean_wpm",
    "non_verbal_cue_probability",
    "factual_inversion_rate",
    "anomia_rate",
    "distress_escalation_prob",
)


def provenance_label(source: str) -> str:
    """Return the provenance label of a `source` string, or ``UNLABELED``."""
    for label in _LABELS:
        if source.startswith(f"{label}:"):
            return label
    return UNLABELED


def prior_sources(priors: StatisticalPriors) -> dict[str, str]:
    """Map ``stage_N.field`` to its `source` string for every stage prior."""
    sources: dict[str, str] = {}
    for stage_key, profile in priors.stages.items():
        for field in PRIOR_FIELDS:
            prior: NormalPrior | BernoulliPrior = getattr(profile, field)
            sources[f"{stage_key}.{field}"] = prior.source
    return sources


def assert_no_placeholders(priors: StatisticalPriors) -> list[str]:
    """Return the keys whose value is not backed by a citation.

    An empty list means every prior is ``CITED``. Despite the name this does not
    raise; callers decide whether remaining keys are fatal (the ``priors-status``
    CLI command exits non-zero).
    """
    return [
        key
        for key, source in prior_sources(priors).items()
        if not source.strip() or provenance_label(source) != CITED
    ]
