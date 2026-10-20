"""Write path for patient memory: encoding new experiences and ongoing decay.

`MemoryEncoder` simulates two processes that the read-only `MemoryRecall`
cannot:

- **Encoding** (`encode_episode`): a new experience (e.g. something the
  trainee just said) is stored as an episodic memory -- unless encoding
  fails, which becomes more likely at later stages. What *is* stored is
  already noisy (corrupted text, perturbed vector), with the clean original
  kept in `original_text` for evaluation.
- **Decay** (`decay_pass`): a stage-scaled fraction of all stored records is
  further corrupted, simulating progression over time. The text is
  re-embedded after corruption so the vector drifts together with the
  content, then perturbed with Gaussian noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.models import Stage
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import Embedder
from memory.noise import MemoryNoiseModel

_TEXT_FIELD = {EPISODIC_COLLECTION: "text", FACTS_COLLECTION: "fact"}
_MAX_CORRUPTION = 1.0
_DECAY_SEVERITY_FLOOR = 0.15


@dataclass(frozen=True)
class EncodeResult:
    record_id: int
    stored_text: str
    original_text: str
    corrupted: bool


class MemoryEncoder:
    """Stores new memories and applies progressive decay to existing ones."""

    def __init__(
        self, client: PatientMemoryClient, embedder: Embedder, noise: MemoryNoiseModel
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.noise = noise

    def encode_episode(
        self,
        text: str,
        stage: Stage,
        emotional_valence: str = "neutral",
        era: str = "recent",
        source: str = "conversation",
    ) -> EncodeResult | None:
        """Try to store a new episodic memory; None if encoding failed."""
        text = text.strip()
        if not text or self.noise.encoding_fails(stage):
            return None

        severity = self.noise.encoding_severity(stage)
        result = self.noise.corrupt_text(text, severity)
        vector = self.noise.perturb_vector(
            self.embedder.embed(result.text), self.noise.vector_sigma(stage)
        )
        extra: dict[str, object] = {"source": source}
        if result.changed:
            extra.update(
                corrupted=True,
                corruption_level=severity,
                original_text=text,
                corruption_ops=",".join(result.ops),
            )
        record_id = self.client.insert_episodic_memory(
            text=result.text,
            embedding=vector,
            emotional_valence=emotional_valence,
            era=era,
            extra=extra,
        )
        return EncodeResult(record_id, result.text, text, result.changed)

    def decay_pass(self, stage: Stage) -> dict[str, int]:
        """Corrupt a stage-scaled random fraction of every stored record.

        Returns the number of records altered per collection. A record is
        "altered" only if its text actually changed; vector noise is applied
        to every selected record regardless.
        """
        fraction = self.noise.config.decay_fraction[stage]
        severity = max(self.noise.encoding_severity(stage) * 0.5, _DECAY_SEVERITY_FLOOR)
        sigma = self.noise.vector_sigma(stage)
        altered = {EPISODIC_COLLECTION: 0, FACTS_COLLECTION: 0}

        for collection, text_field in _TEXT_FIELD.items():
            for record in self.client.all_records(collection):
                if self.noise.rng.random() >= fraction:
                    continue
                current = str(record.get(text_field, ""))
                result = self.noise.corrupt_text(current, severity)
                changes: dict[str, object] = {"last_decayed": time.time()}
                if result.changed:
                    original_key = f"original_{text_field}"
                    changes.update(
                        {
                            text_field: result.text,
                            "corrupted": True,
                            "corruption_level": min(
                                float(record.get("corruption_level", 0.0)) + severity,
                                _MAX_CORRUPTION,
                            ),
                            original_key: record.get(original_key, current),
                        }
                    )
                    altered[collection] += 1
                vector = self.noise.perturb_vector(self.embedder.embed(result.text), sigma)
                self.client.update_record(collection, int(record["id"]), changes, vector=vector)
        return altered
