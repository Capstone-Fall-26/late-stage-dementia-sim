"""Core patient simulation engine.

`PatientEngine` orchestrates every turn of a simulated dementia-patient
conversation:

1. Assess the trainee's utterance for sensitive-topic/pressuring/redirection
   signals (`behavior.distress.evaluate_trigger`).
2. Transition the patient's coarse distress state (`behavior.distress.DistressTracker`).
3. Sample literature-grounded behavioral constraints for this turn
   (`behavior.symptom_sampler.sample_constraints`).
4. Recall cognitively-degraded episodic/biographical memories relevant to
   the utterance (`memory.recall.MemoryRecall`), unless memory fog fired.
5. Compose a dynamic system prompt that encodes the constraints and
   recalled memories without ever breaking character.
6. Call the configured LLM backend and defensively post-process the
   response to guarantee the hard constraints (word cap, required
   non-verbal cue) hold even if the model does not follow instructions
   perfectly.
7. Log a structured prompt trace and return a typed `EngineTurnResult`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from behavior.distress import DistressTracker, evaluate_trigger
from behavior.symptom_sampler import sample_constraints
from core.llm_providers import LLMProvider, get_llm_provider
from core.logging_config import get_logger, log_prompt_trace
from core.models import (
    DistressState,
    EngineTurnResult,
    MemoryHit,
    PatientProfile,
    Stage,
    StatisticalPriors,
    SymptomConstraints,
    load_patient_profile,
    load_statistical_priors,
)
from memory.client import PatientMemoryClient
from memory.embeddings import Embedder, get_embedder
from memory.recall import MemoryRecall

_DISTRESS_INSTRUCTIONS: dict[DistressState, str] = {
    DistressState.CALM: "You feel settled and at ease right now.",
    DistressState.CONFUSED: "You feel muddled; time, place, and people blur together.",
    DistressState.AGITATED: "You feel unsettled and on edge; your tone should carry irritation or fear.",
    DistressState.WITHDRAWN: "You feel distant and unreachable; you have little will to engage.",
}

_MAX_HISTORY_MESSAGES = 16


class PatientEngine:
    """Stateful, single-patient conversation engine."""

    def __init__(
        self,
        profile: PatientProfile,
        priors: StatisticalPriors,
        llm: LLMProvider,
        memory_client: PatientMemoryClient,
        embedder: Embedder,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.profile = profile
        self.priors = priors
        self.llm = llm
        self.memory_client = memory_client
        self.embedder = embedder
        self.rng = rng or np.random.default_rng()
        self.recall = MemoryRecall(memory_client, embedder, priors, rng=self.rng)
        self.distress = DistressTracker(priors, rng=self.rng)
        self.logger = get_logger("patient_engine")
        self._conversation: list[dict[str, str]] = []

    @classmethod
    def from_config(
        cls,
        profile_path: str | Path,
        priors_path: str | Path,
        backend: str,
        model: str,
        api_key: str | None = None,
        host: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        db_path: str | Path = "./milvus_patient_memory.db",
        embedding_backend: str = "hashing",
        seed: int | None = None,
    ) -> PatientEngine:
        """Convenience constructor: load config, wire dependencies, seed memory."""
        profile = load_patient_profile(profile_path)
        priors = load_statistical_priors(priors_path)
        llm = get_llm_provider(
            backend=backend,
            model=model,
            api_key=api_key,
            host=host,
            endpoint=endpoint,
            api_version=api_version,
        )
        embedder = get_embedder(embedding_backend)
        memory_client = PatientMemoryClient(db_path=db_path)
        rng = np.random.default_rng(seed)
        engine = cls(profile, priors, llm, memory_client, embedder, rng=rng)
        engine.seed_memory()
        return engine

    def seed_memory(self) -> dict[str, int]:
        """Seed episodic/biographical memory collections from the patient profile."""
        return self.memory_client.seed_from_profile(self.profile, self.embedder)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _constraint_instructions(self, constraints: SymptomConstraints) -> list[str]:
        lines: list[str] = []
        if constraints.max_words <= 0:
            lines.append(
                "You cannot produce any spoken words this turn. Respond with ONLY a "
                "non-verbal action description in asterisks, nothing else."
            )
        else:
            lines.append(f"Your entire response must be at most {constraints.max_words} words.")
        if constraints.nonverbal_cue:
            lines.append(
                f"You must include this exact non-verbal action somewhere in your response: "
                f"{constraints.nonverbal_cue}"
            )
        if constraints.memory_fog:
            lines.append(
                "You cannot recall what is being asked right now. Express confusion or an "
                "inability to remember. Do not confidently state a specific, invented answer."
            )
        if constraints.factual_inversion:
            lines.append(
                "You confidently misattribute a fact (wrong person, place, date, or relationship) "
                "without noticing the error or correcting yourself."
            )
        if constraints.anomia:
            lines.append(
                "You struggle to find the right word. Substitute a vague word, a wrong word, or "
                "trail off mid-sentence instead of the word you meant."
            )
        if constraints.distress_escalation:
            lines.append("Your emotional distress escalates visibly during this turn.")
        return lines

    def _memory_context(self, memory_hits: list[MemoryHit]) -> str:
        if not memory_hits:
            return "Nothing specific surfaces in your mind right now."
        lines = []
        for hit in memory_hits:
            tag = (
                " (this memory feels uncertain, possibly mixed up with something else)"
                if hit.confused
                else ""
            )
            lines.append(f"- {hit.text}{tag}")
        return (
            "These fragments surface in your mind right now. Weave them in loosely and "
            "imprecisely if relevant -- do not quote them verbatim like a script:\n"
            + "\n".join(lines)
        )

    def build_system_prompt(
        self, constraints: SymptomConstraints, memory_hits: list[MemoryHit]
    ) -> str:
        """Compose the dynamic, in-character system prompt for this turn."""
        profile = self.profile
        biography_lines = "\n".join(f"- {k}: {v}" for k, v in profile.biography.items())
        family_lines = "\n".join(
            f"- {m.name} ({m.relation}, {m.status})" for m in profile.family_members
        )

        header = (
            f"You ARE {profile.name} (preferred name: {profile.preferred_name}), a "
            f"{profile.gender} patient born in {profile.birth_year}, living with stage "
            f"{constraints.stage.value} dementia (GDS/FAST). You are being spoken with by a "
            "caregiving trainee. Stay completely in character at all times. Never mention "
            "being an AI, a simulation, or a language model. Never break the fourth wall. "
            "Output ONLY what the patient would say and/or do -- no narration labels, no "
            "explanations, no character name prefix."
        )

        biography_block = f"Your life (draw on this loosely, as your own memory, not a script):\n{biography_lines}"
        family_block = f"Your family:\n{family_lines}"
        distress_block = f"Current emotional state ({constraints.distress_state.value}): {_DISTRESS_INSTRUCTIONS[constraints.distress_state]}"
        constraint_block = "Formatting rules for THIS turn, follow strictly:\n" + "\n".join(
            f"- {line}" for line in self._constraint_instructions(constraints)
        )
        memory_block = self._memory_context(memory_hits)

        stage_style = ""
        if constraints.stage == Stage.SEVEN:
            stage_style = (
                "Speech should be sparse, often monosyllabic or fragmentary, with heavy use "
                "of non-verbal descriptors in asterisks (*gazes blankly*, *a low moan*). Full "
                "grammatical sentences are rare."
            )

        return "\n\n".join(
            part
            for part in [
                header,
                biography_block,
                family_block,
                distress_block,
                constraint_block,
                memory_block,
                stage_style,
            ]
            if part
        )

    # ------------------------------------------------------------------ #
    # Turn execution
    # ------------------------------------------------------------------ #

    def _postprocess(self, text: str, constraints: SymptomConstraints) -> str:
        text = text.strip()
        if constraints.max_words <= 0:
            return constraints.nonverbal_cue or "*silence*"

        cue = constraints.nonverbal_cue
        cue_already_present = cue is not None and cue in text
        cue_word_count = len(cue.split()) if (cue and not cue_already_present) else 0
        spoken_budget = max(0, constraints.max_words - cue_word_count)

        words = text.split()
        if len(words) > spoken_budget:
            text = " ".join(words[:spoken_budget])

        if cue and not cue_already_present:
            text = f"{cue} {text}".strip()

        return text or (cue or "*silence*")

    def respond(self, user_input: str) -> EngineTurnResult:
        """Process one trainee utterance and produce the patient's next turn."""
        start = time.time()

        assessment = evaluate_trigger(user_input, self.profile, self.priors)
        distress_state = self.distress.transition(assessment)
        constraints = sample_constraints(
            self.profile.stage, distress_state, self.priors, rng=self.rng
        )

        memory_hits: list[MemoryHit] = []
        if not constraints.memory_fog:
            memory_hits = [
                *self.recall.recall_episodic(user_input, self.profile.stage, top_k=2),
                *self.recall.recall_facts(user_input, self.profile.stage, top_k=2),
            ]

        system_prompt = self.build_system_prompt(constraints, memory_hits)

        self._conversation.append({"role": "user", "content": user_input})
        history_window = self._conversation[-_MAX_HISTORY_MESSAGES:]

        raw_response = self.llm.generate(system_prompt, history_window)
        response_text = self._postprocess(raw_response, constraints)

        self._conversation.append({"role": "assistant", "content": response_text})

        latency_ms = (time.time() - start) * 1000.0
        log_prompt_trace(
            backend=type(self.llm).__name__,
            model=getattr(self.llm, "model", "unknown"),
            stage=self.profile.stage.value,
            distress_state=distress_state.value,
            latency_ms=latency_ms,
            memory_hits_count=len(memory_hits),
            memory_scores=[hit.score for hit in memory_hits],
            response_preview=response_text,
        )

        return EngineTurnResult(
            response_text=response_text,
            distress_state=distress_state,
            constraints=constraints,
            memory_hits=memory_hits,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

    def close(self) -> None:
        self.memory_client.close()
