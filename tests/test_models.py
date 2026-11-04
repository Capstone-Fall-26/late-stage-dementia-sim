from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import DistressState, Stage, StatisticalPriors


def test_priors_load(priors: StatisticalPriors) -> None:
    assert priors.version == 2
    for stage in (Stage.FIVE, Stage.SIX, Stage.SEVEN):
        stage_profile = priors.get_stage_profile(stage)
        assert 0.0 <= stage_profile.non_verbal_cue_probability.value <= 1.0
        assert stage_profile.verbal_fluency_mean_wpm.min < stage_profile.verbal_fluency_mean_wpm.max


def test_transition_rows_sum_to_one(priors: StatisticalPriors) -> None:
    for state in DistressState:
        row = priors.get_transition_row(state)
        assert abs(sum(row.values()) - 1.0) < 1e-6


def test_nonverbal_bank_nonempty(priors: StatisticalPriors) -> None:
    for stage in (Stage.FIVE, Stage.SIX, Stage.SEVEN):
        assert len(priors.get_nonverbal_bank(stage)) > 0


def test_invalid_transition_matrix_rejected() -> None:
    bad_data = {
        "version": 1,
        "citation_note": "test",
        "stages": {},
        "distress_transition_matrix": {"calm": {"calm": 0.5, "confused": 0.1}},
        "non_verbal_cue_bank": {},
        "redirection_keywords": [],
        "pressuring_keywords": [],
    }
    with pytest.raises(ValidationError):
        StatisticalPriors.model_validate(bad_data)


def test_profile_load(profile) -> None:
    assert profile.name == "Margaret Hale"
    assert profile.stage == Stage.SEVEN
    assert len(profile.biographical_facts) > 0
    assert len(profile.episodic_memories) > 0
    relations = {member.relation for member in profile.family_members}
    assert "husband" in relations
