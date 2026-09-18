"""Shared pytest fixtures for the dementia simulation test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.models import (
    PatientProfile,
    StatisticalPriors,
    load_patient_profile,
    load_statistical_priors,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRIORS_PATH = _REPO_ROOT / "config" / "statistical_priors.yaml"
_PROFILE_PATH = _REPO_ROOT / "config" / "patient_profiles" / "stage7_late.json"


@pytest.fixture(scope="session")
def priors() -> StatisticalPriors:
    return load_statistical_priors(_PRIORS_PATH)


@pytest.fixture(scope="session")
def profile() -> PatientProfile:
    return load_patient_profile(_PROFILE_PATH)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
