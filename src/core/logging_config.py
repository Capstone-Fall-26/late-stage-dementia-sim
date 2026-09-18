"""Centralized loguru logging configuration.

Configures three sinks once per process:

1. Console: human-readable, colorized, INFO and above.
2. Rotating file (`logs/dementia_sim_*.log`): DEBUG and above, all modules,
   rotated at 10 MB with 10 backups retained, gzip-compressed.
3. Structured prompt-trace file (`logs/prompt_traces.jsonl`): one JSON line
   per LLM call, capturing backend/model, stage, distress state, latency,
   and memory-retrieval scores -- intended for offline analysis alongside
   the evaluation suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _logger

_CONFIGURED = False


def configure_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Idempotently configure loguru sinks. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()

    _logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> - <level>{message}</level>"
        ),
        filter=lambda record: not record["extra"].get("trace", False),
    )

    _logger.add(
        log_dir / "dementia_sim_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention=10,
        compression="zip",
        enqueue=True,
        format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]} | {message}"),
        filter=lambda record: not record["extra"].get("trace", False),
    )

    _logger.add(
        log_dir / "prompt_traces.jsonl",
        level="INFO",
        rotation="10 MB",
        retention=10,
        enqueue=True,
        serialize=True,
        filter=lambda record: record["extra"].get("trace", False),
    )

    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with a module name for console/file sinks."""
    configure_logging()
    return _logger.bind(module=name, trace=False)


def log_prompt_trace(
    *,
    backend: str,
    model: str,
    stage: int,
    distress_state: str,
    latency_ms: float,
    memory_hits_count: int,
    memory_scores: list[float],
    response_preview: str,
) -> None:
    """Write one structured JSON line to `logs/prompt_traces.jsonl`."""
    configure_logging()
    _logger.bind(trace=True).info(
        "prompt_trace",
        backend=backend,
        model=model,
        stage=stage,
        distress_state=distress_state,
        latency_ms=round(latency_ms, 2),
        memory_hits_count=memory_hits_count,
        memory_scores=memory_scores,
        response_preview=response_preview[:200],
    )
