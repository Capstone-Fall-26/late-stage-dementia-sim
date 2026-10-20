"""Noise model simulating degradation of patient memory.

Three kinds of noise are modeled, all driven by an injected RNG so that
simulations are reproducible:

- **Text corruption** (`corrupt_text`): entity substitution (e.g. Susan ->
  Ellen, daughter -> sister), numeric drift (28 years -> 31 years), and
  word dropout (``...``), scaled by a severity in [0, 1].
- **Vector perturbation** (`perturb_vector`): Gaussian noise on an embedding,
  so a degraded memory is also harder to *find*, not just wrong when found.
- **Encoding failure** (`encoding_fails`): new experiences are increasingly
  unlikely to be stored at all as the stage worsens (anterograde amnesia).

Stage-dependent magnitudes are simulation placeholders, like the numeric
priors in `config/statistical_priors.yaml`; they are not clinical values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from core.models import PatientProfile, Stage

_NUMBER_RE = re.compile(r"\b\d+\b")
_WORD_RE = re.compile(r"[A-Za-z']+")

# Generic relationship words people with dementia commonly mix up.
_RELATION_CONFUSIONS: dict[str, str] = {
    "daughter": "sister",
    "sister": "daughter",
    "son": "brother",
    "brother": "son",
    "husband": "father",
    "mother": "sister",
}


@dataclass(frozen=True)
class NoiseConfig:
    """Per-stage noise magnitudes."""

    # Probability a new experience is never stored.
    encoding_failure: dict[Stage, float] = field(
        default_factory=lambda: {Stage.FIVE: 0.25, Stage.SIX: 0.5, Stage.SEVEN: 0.8}
    )
    # Text corruption severity applied to a newly encoded experience.
    encoding_severity: dict[Stage, float] = field(
        default_factory=lambda: {Stage.FIVE: 0.2, Stage.SIX: 0.4, Stage.SEVEN: 0.6}
    )
    # Fraction of stored records touched by one `decay_pass`.
    decay_fraction: dict[Stage, float] = field(
        default_factory=lambda: {Stage.FIVE: 0.1, Stage.SIX: 0.25, Stage.SEVEN: 0.4}
    )
    # Stddev-like magnitude of the vector perturbation (unit-vector scale).
    vector_sigma: dict[Stage, float] = field(
        default_factory=lambda: {Stage.FIVE: 0.05, Stage.SIX: 0.1, Stage.SEVEN: 0.2}
    )


@dataclass(frozen=True)
class CorruptionResult:
    text: str
    changed: bool
    ops: tuple[str, ...] = ()


def build_confusion_map(profile: PatientProfile) -> dict[str, str]:
    """Derive plausible name confusions from a patient profile.

    A family member whose notes mention another member by first name (e.g.
    "Margaret often mistakes her for her own sister, Ellen") is mapped to
    that member. Remaining names are mapped cyclically so every member can
    be confused with someone.
    """
    first_names = {m.name.split()[0]: m for m in profile.family_members}
    mapping: dict[str, str] = {}
    for name, member in first_names.items():
        notes = member.notes or ""
        for other in first_names:
            if other != name and re.search(rf"\b{re.escape(other)}\b", notes):
                mapping[name] = other
                break
    names = list(first_names)
    for i, name in enumerate(names):
        if name not in mapping and len(names) > 1:
            mapping[name] = names[(i + 1) % len(names)]
    return mapping


class MemoryNoiseModel:
    """Applies stage-scaled noise to memory text and vectors."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        config: NoiseConfig | None = None,
        confusion_map: dict[str, str] | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.config = config or NoiseConfig()
        self.confusion_map = confusion_map or {}

    # -- stage lookups -------------------------------------------------- #

    def encoding_fails(self, stage: Stage) -> bool:
        return bool(self.rng.random() < self.config.encoding_failure[stage])

    def encoding_severity(self, stage: Stage) -> float:
        return self.config.encoding_severity[stage]

    def vector_sigma(self, stage: Stage) -> float:
        return self.config.vector_sigma[stage]

    # -- noise ops ------------------------------------------------------ #

    def corrupt_text(self, text: str, severity: float) -> CorruptionResult:
        """Corrupt `text` with probability scaled by `severity` in [0, 1].

        Each eligible token (a confusable name/relation word, a number, or a
        content word) is independently altered with a probability derived
        from `severity`, so higher severity means more edits on average.
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        if severity <= 0.0 or not text:
            return CorruptionResult(text, False)

        ops: list[str] = []

        def sub_word(match: re.Match[str]) -> str:
            word = match.group(0)
            replacement = self.confusion_map.get(word)
            if replacement is None:
                lower = _RELATION_CONFUSIONS.get(word.lower())
                if lower is not None:
                    replacement = lower.capitalize() if word[0].isupper() else lower
            if replacement is not None and self.rng.random() < severity:
                ops.append("substitution")
                return replacement
            if len(word) > 4 and self.rng.random() < severity * 0.2:
                ops.append("dropout")
                return "..."
            return word

        def sub_number(match: re.Match[str]) -> str:
            value = int(match.group(0))
            if self.rng.random() < severity:
                shift = int(self.rng.integers(1, max(2, int(3 + value * 0.1))))
                ops.append("numeric_drift")
                return str(max(0, value + shift * (1 if self.rng.random() < 0.5 else -1)))
            return match.group(0)

        out = _NUMBER_RE.sub(sub_number, text)
        out = _WORD_RE.sub(sub_word, out)
        return CorruptionResult(out, out != text, tuple(dict.fromkeys(ops)))

    def perturb_vector(self, vector: np.ndarray, sigma: float) -> np.ndarray:
        """Add random-direction noise of magnitude `sigma`, keeping unit norm."""
        if sigma <= 0.0:
            return vector
        direction = np.asarray(self.rng.standard_normal(vector.shape[0]), dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        out = vector + sigma * direction
        norm = float(np.linalg.norm(out))
        return np.asarray(out / norm, dtype=np.float32) if norm > 0 else vector
