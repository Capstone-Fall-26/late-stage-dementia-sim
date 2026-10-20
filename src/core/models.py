"""Shared domain models for the dementia simulation framework.

This module has no dependency on `memory`, `behavior`, or any LLM client so
that it can be safely imported from anywhere in the codebase without risking
circular imports.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Stage(int, Enum):
    """GDS/FAST cognitive decline stage. Only 5-7 are modeled by this project."""

    FIVE = 5
    SIX = 6
    SEVEN = 7

    @property
    def yaml_key(self) -> str:
        return f"stage_{self.value}"


class DistressState(str, Enum):
    """Coarse emotional/behavioral state used by the distress transition matrix."""

    CALM = "calm"
    CONFUSED = "confused"
    AGITATED = "agitated"
    WITHDRAWN = "withdrawn"


# --------------------------------------------------------------------------- #
# Statistical priors
# --------------------------------------------------------------------------- #


class NormalPrior(BaseModel):
    """A truncated-normal prior, e.g. words-per-turn verbal fluency."""

    distribution: str = "normal"
    mean: float
    std: float = Field(gt=0)
    min: float
    max: float
    unit: str
    source: str

    @model_validator(mode="after")
    def _check_bounds(self) -> NormalPrior:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self


class BernoulliPrior(BaseModel):
    """A single-probability prior, e.g. probability of a non-verbal cue firing."""

    distribution: str = "bernoulli"
    value: float = Field(ge=0.0, le=1.0)
    source: str


class StageProfile(BaseModel):
    """All priors associated with a single GDS/FAST stage."""

    label: str
    verbal_fluency_mean_wpm: NormalPrior
    non_verbal_cue_probability: BernoulliPrior
    factual_inversion_rate: BernoulliPrior
    anomia_rate: BernoulliPrior
    distress_escalation_prob: BernoulliPrior


class StatisticalPriors(BaseModel):
    """Fully parsed contents of config/statistical_priors.yaml."""

    version: int
    citation_note: str
    stages: dict[str, StageProfile]
    distress_transition_matrix: dict[str, dict[str, float]]
    non_verbal_cue_bank: dict[str, list[str]]
    redirection_keywords: list[str]
    pressuring_keywords: list[str]

    @field_validator("distress_transition_matrix")
    @classmethod
    def _rows_sum_to_one(cls, matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for row_name, row in matrix.items():
            total = sum(row.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"distress_transition_matrix row '{row_name}' sums to {total}, expected 1.0"
                )
        return matrix

    def get_stage_profile(self, stage: Stage) -> StageProfile:
        try:
            return self.stages[stage.yaml_key]
        except KeyError as exc:
            raise KeyError(f"No stage profile configured for {stage.yaml_key}") from exc

    def get_transition_row(self, state: DistressState) -> dict[DistressState, float]:
        row = self.distress_transition_matrix[state.value]
        return {DistressState(k): v for k, v in row.items()}

    def get_nonverbal_bank(self, stage: Stage) -> list[str]:
        return self.non_verbal_cue_bank.get(stage.yaml_key, [])


def load_statistical_priors(path: str | Path) -> StatisticalPriors:
    """Load and validate `config/statistical_priors.yaml`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Statistical priors file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return StatisticalPriors.model_validate(raw)


# --------------------------------------------------------------------------- #
# Patient profile
# --------------------------------------------------------------------------- #


class FamilyMember(BaseModel):
    name: str
    relation: str
    status: str
    notes: str | None = None


class BiographicalFact(BaseModel):
    fact: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class EpisodicMemorySeed(BaseModel):
    text: str
    emotional_valence: str
    era: str


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    preferred_name: str
    birth_year: int
    stage: Stage
    gender: str
    biography: dict[str, str]
    family_members: list[FamilyMember]
    biographical_facts: list[BiographicalFact]
    episodic_memories: list[EpisodicMemorySeed]
    sensitive_topics: list[str]
    distress_triggers: list[str]
    redirection_notes: str


def load_patient_profile(path: str | Path) -> PatientProfile:
    """Load and validate a patient profile JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patient profile file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return PatientProfile.model_validate(raw)


# --------------------------------------------------------------------------- #
# Runtime / turn-level models
# --------------------------------------------------------------------------- #


class SymptomConstraints(BaseModel):
    """Output of symptom_sampler.sample_constraints for a single patient turn."""

    stage: Stage
    distress_state: DistressState
    max_words: int = Field(ge=0)
    nonverbal_cue: str | None = None
    memory_fog: bool = False
    factual_inversion: bool = False
    anomia: bool = False
    distress_escalation: bool = False


class MemoryHit(BaseModel):
    """A single result returned by memory.recall, possibly noise-injected."""

    text: str
    score: float
    collection: str
    confused: bool = False
    # Why this hit is unreliable: "swap" (a neighboring memory surfaced
    # instead), "substitution" (details altered at retrieval time) or
    # "stored_corruption" (the stored record was already degraded). None
    # means the hit is faithful to what was originally stored.
    noise_type: str | None = None
    # Pre-corruption ground truth, when known.
    original_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineTurnResult(BaseModel):
    """Everything produced by PatientEngine.respond() for a single turn."""

    response_text: str
    distress_state: DistressState
    constraints: SymptomConstraints
    memory_hits: list[MemoryHit]
    memory_encoded: bool = False
    latency_ms: float
    timestamp: float
