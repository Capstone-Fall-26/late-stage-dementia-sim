from __future__ import annotations

import numpy as np

from core.models import PatientProfile, Stage
from memory.noise import MemoryNoiseModel, build_confusion_map


def test_confusion_map_uses_profile_notes(profile: PatientProfile) -> None:
    mapping = build_confusion_map(profile)
    assert mapping["Susan"] == "Ellen"
    assert mapping["Ellen"] == "Susan"
    assert set(mapping) == {"Walter", "Susan", "David", "Ellen"}


def test_zero_severity_is_identity() -> None:
    model = MemoryNoiseModel(rng=np.random.default_rng(0), confusion_map={"Susan": "Ellen"})
    result = model.corrupt_text("Susan visited for 28 years.", 0.0)
    assert not result.changed
    assert result.text == "Susan visited for 28 years."


def test_full_severity_substitutes_names_and_drifts_numbers() -> None:
    model = MemoryNoiseModel(rng=np.random.default_rng(0), confusion_map={"Susan": "Ellen"})
    result = model.corrupt_text("Susan is my daughter and taught for 28 years.", 1.0)
    assert result.changed
    assert "Ellen" in result.text
    assert "28" not in result.text
    assert {"substitution", "numeric_drift"} <= set(result.ops)


def test_corruption_is_reproducible_for_a_seed() -> None:
    text = "Susan is my daughter and Walter proposed in 1962 by the lake."
    a = MemoryNoiseModel(np.random.default_rng(5), confusion_map={"Susan": "Ellen"})
    b = MemoryNoiseModel(np.random.default_rng(5), confusion_map={"Susan": "Ellen"})
    assert a.corrupt_text(text, 0.6) == b.corrupt_text(text, 0.6)


def test_perturb_vector_keeps_unit_norm_and_moves_it() -> None:
    model = MemoryNoiseModel(rng=np.random.default_rng(1))
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = 1.0
    out = model.perturb_vector(vec, 0.2)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
    assert float(out @ vec) < 1.0
    assert np.array_equal(model.perturb_vector(vec, 0.0), vec)


def test_encoding_failure_rate_rises_with_stage() -> None:
    model = MemoryNoiseModel(rng=np.random.default_rng(2))
    rates = {
        stage: np.mean([model.encoding_fails(stage) for _ in range(2000)])
        for stage in (Stage.FIVE, Stage.SEVEN)
    }
    assert rates[Stage.SEVEN] > rates[Stage.FIVE] + 0.3
