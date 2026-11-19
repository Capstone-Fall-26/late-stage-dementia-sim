"""Read-only 2D visual map of a patient's memory vectors with degradation drift.

Pipeline: `load_memory_records` -> `build_memory_map` (PCA fitted on the *stored*
vectors; ground-truth originals and an optional query are projected into the same
basis) -> `write_memory_map` (single offline HTML: vendored Pico.css + inlined Plotly).

Nothing here writes to the store, bumps `access_count`, or calls an LLM.
"""

from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import Embedder

_PICO_CSS_PATH = Path(__file__).parent / "assets" / "pico.min.css"

CLEAN_COLOR = "#7f8c8d"
CORRUPT_COLOR = "#e67e22"
FACT_COLOR = "#2980b9"
QUERY_COLOR = "#c0392b"

_EPISODIC_FIELDS = [
    "id",
    "vector",
    "text",
    "emotional_valence",
    "era",
    "source",
    "access_count",
    "corrupted",
    "corruption_level",
    "corruption_ops",
    "original_text",
]
_FACT_FIELDS = [
    "id",
    "vector",
    "fact",
    "category",
    "confidence",
    "source",
    "access_count",
    "corrupted",
    "corruption_level",
    "corruption_ops",
    "original_fact",
]
_COLUMNS = [
    "key",
    "kind",
    "text",
    "original_text",
    "corrupted",
    "corruption_level",
    "corruption_ops",
    "era",
    "emotional_valence",
    "category",
    "access_count",
    "source",
    "vector",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _normalize(row: dict[str, Any], kind: str) -> dict[str, Any]:
    text_key = "text" if kind == "episodic" else "fact"
    text = str(row.get(text_key, ""))
    corrupted = bool(row.get("corrupted", False))
    original = str(row.get(f"original_{text_key}") or text) if corrupted else text
    return {
        "key": f"{kind}:{row.get('id')}",
        "kind": kind,
        "text": text,
        "original_text": original,
        "corrupted": corrupted,
        "corruption_level": float(row.get("corruption_level", 0.0) or 0.0),
        "corruption_ops": str(row.get("corruption_ops", "") or ""),
        "era": str(row.get("era", "") or ""),
        "emotional_valence": str(row.get("emotional_valence", "") or ""),
        "category": str(row.get("category", "") or ""),
        "access_count": int(row.get("access_count", 0) or 0),
        "source": str(row.get("source", "") or ""),
        "vector": np.asarray(row["vector"], dtype=np.float32),
    }


def load_memory_records(client: PatientMemoryClient) -> pd.DataFrame:
    """Read every episodic memory and biographical fact (with vectors) into one DataFrame."""
    records: list[dict[str, Any]] = []
    for collection, kind, fields in (
        (EPISODIC_COLLECTION, "episodic", _EPISODIC_FIELDS),
        (FACTS_COLLECTION, "fact", _FACT_FIELDS),
    ):
        records.extend(_normalize(r, kind) for r in client.list_all(collection, fields))
    return pd.DataFrame(records, columns=_COLUMNS)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


class PCAProjector:
    """Deterministic 2-component PCA via SVD, with out-of-sample `transform`.

    Sign convention: the largest-magnitude loading of each component is positive.
    Degenerate inputs (0/1 rows, <2 dims, zero variance) yield zero coordinates.
    """

    n_components = 2

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.zeros(0)
        self.components_: np.ndarray = np.zeros((self.n_components, 0))
        self.explained_variance_ratio_: np.ndarray = np.zeros(self.n_components)

    def fit(self, vectors: np.ndarray) -> PCAProjector:
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2:
            x = x.reshape(len(x), -1) if x.size else np.zeros((0, 0))
        n, d = x.shape
        self.mean_ = x.mean(axis=0) if n else np.zeros(d)
        self.components_ = np.zeros((self.n_components, d))
        self.explained_variance_ratio_ = np.zeros(self.n_components)
        if n < 2 or d == 0:
            return self
        centered = x - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        var = s**2 / (n - 1)
        total = float(var.sum())
        if total <= 1e-12:
            return self
        k = min(self.n_components, len(s))
        comps = vt[:k].copy()
        for i in range(k):
            if comps[i, np.argmax(np.abs(comps[i]))] < 0:
                comps[i] = -comps[i]
        self.components_[:k] = comps
        self.explained_variance_ratio_[:k] = var[:k] / total
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        if x.size == 0:
            return np.zeros((0, self.n_components))
        return np.asarray((x - self.mean_) @ self.components_.T)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.fit(vectors).transform(vectors)


def project_2d(vectors: np.ndarray) -> np.ndarray:
    """Project `(n, d)` vectors to `(n, 2)` with deterministic PCA."""
    return PCAProjector().fit_transform(vectors)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise `1 - cos(a, b)`; 0.0 where either vector has zero norm."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sim = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    return np.asarray(1.0 - sim)


def _wrap(text: str, width: int = 60) -> str:
    return "<br>".join(html.escape(line) for line in textwrap.wrap(text, width) or [""])


def _hover(row: pd.Series, drift: float | None) -> str:
    parts = [f"<b>{_wrap(str(row['text']))}</b>"]
    if row["corrupted"]:
        parts.append(f"Original: {_wrap(str(row['original_text']))}")
        if drift is not None:
            parts.append(f"Cosine drift: {drift:.3f}")
    if row["era"]:
        parts.append(f"Era: {html.escape(str(row['era']))}")
    if row["kind"] == "fact" and row["category"]:
        parts.append(f"Category: {html.escape(str(row['category']))}")
    if row["emotional_valence"]:
        parts.append(f"Valence: {html.escape(str(row['emotional_valence']))}")
    parts.append(f"Access count: {int(row['access_count'])}")
    parts.append(f"Source: {html.escape(str(row['source']))}")
    if row["corruption_ops"]:
        parts.append(f"Corruption ops: {html.escape(str(row['corruption_ops']))}")
    return "<br>".join(parts)


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="No memories found",
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=420,
    )
    fig.layout.meta = {"drift": {}}
    return fig


def build_memory_map(
    df: pd.DataFrame,
    embedder: Embedder,
    *,
    query_vector: np.ndarray | None = None,
    query_label: str | None = None,
    stage: int | None = None,
) -> go.Figure:
    """Build the 2D map. Per-record cosine drift is stored in `fig.layout.meta["drift"]`."""
    if df.empty:
        return _empty_figure()

    df = df.reset_index(drop=True)
    stored = np.stack(df["vector"].to_numpy())
    pca = PCAProjector().fit(stored)
    xy = pca.transform(stored)

    # Ground-truth overlay: re-embed originals with the store's embedder, same PCA basis.
    corrupted_idx = np.flatnonzero(df["corrupted"].to_numpy(dtype=bool))
    drift_by_pos: dict[int, float] = {}
    orig_xy = np.zeros((0, 2))
    if len(corrupted_idx):
        orig_vecs = np.asarray(
            embedder.embed_batch([str(df.at[i, "original_text"]) for i in corrupted_idx])
        )
        orig_xy = pca.transform(orig_vecs)
        drifts = _cosine_distance(orig_vecs, stored[corrupted_idx])
        drift_by_pos = {int(i): float(d) for i, d in zip(corrupted_idx, drifts, strict=True)}

    fig = go.Figure()
    opacity = np.clip(0.35 + 0.13 * df["access_count"].to_numpy(dtype=float), 0.35, 1.0)
    is_fact = (df["kind"] == "fact").to_numpy()
    is_corrupt = df["corrupted"].to_numpy(dtype=bool)

    groups = [
        ("Episodic (intact)", ~is_fact & ~is_corrupt, "circle", CLEAN_COLOR),
        ("Episodic (corrupted)", ~is_fact & is_corrupt, "circle", CORRUPT_COLOR),
        ("Biographical fact", is_fact & ~is_corrupt, "diamond", FACT_COLOR),
        ("Biographical fact (corrupted)", is_fact & is_corrupt, "diamond", FACT_COLOR),
    ]
    for name, mask, symbol, color in groups:
        pos = np.flatnonzero(mask)
        if not len(pos):
            continue
        sub = df.iloc[pos]
        corrupted_group = "corrupted" in name
        if corrupted_group and symbol == "circle":
            size = (8 + 10 * sub["corruption_level"].clip(0, 1)).to_numpy()
        else:
            size = np.full(len(pos), 11 if symbol == "diamond" else 9)
        fig.add_trace(
            go.Scatter(
                x=xy[pos, 0],
                y=xy[pos, 1],
                mode="markers",
                name=name,
                marker={
                    "symbol": symbol,
                    "color": color,
                    "size": size,
                    "opacity": opacity[pos],
                    "line": {"color": CORRUPT_COLOR if corrupted_group else "white", "width": 2}
                    if symbol == "diamond"
                    else {"color": "white", "width": 1},
                },
                hovertext=[
                    _hover(r, drift_by_pos.get(int(p)))
                    for p, (_, r) in zip(pos, sub.iterrows(), strict=True)
                ],
                hoverinfo="text",
            )
        )

    for i, (pos, (ox, oy)) in enumerate(zip(corrupted_idx.tolist(), orig_xy, strict=True)):
        row = df.iloc[pos]
        drift = drift_by_pos[int(pos)]
        hover = (
            f"Cosine drift: {drift:.3f}<br>Ops: {html.escape(str(row['corruption_ops']) or 'n/a')}"
            f"<br>Original: {_wrap(str(row['original_text']))}"
        )
        fig.add_trace(
            go.Scatter(
                x=[ox, xy[pos, 0]],
                y=[oy, xy[pos, 1]],
                mode="lines+markers",
                line={"color": CORRUPT_COLOR, "width": 1},
                marker={
                    "symbol": ["circle-open", "arrow"],
                    "size": [6, 11],
                    "angleref": "previous",
                    "color": CORRUPT_COLOR,
                },
                legendgroup="drift",
                showlegend=i == 0,
                name="Degradation drift",
                hovertext=[hover, hover],
                hoverinfo="text",
            )
        )

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float64).reshape(1, -1)
        qxy = pca.transform(q)[0]
        sims = stored @ q[0] / np.maximum(np.linalg.norm(stored, axis=1) * np.linalg.norm(q), 1e-12)
        nearest = np.argsort(-sims)[:3]
        lx: list[float | None] = []
        ly: list[float | None] = []
        for n in nearest:
            lx += [float(qxy[0]), float(xy[n, 0]), None]
            ly += [float(qxy[1]), float(xy[n, 1]), None]
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line={"color": QUERY_COLOR, "width": 1, "dash": "dot"},
                opacity=0.5,
                name="Top-3 nearest",
                legendgroup="query",
                hoverinfo="skip",
            )
        )
        label = query_label or "query"
        fig.add_trace(
            go.Scatter(
                x=[float(qxy[0])],
                y=[float(qxy[1])],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": QUERY_COLOR},
                text=[label],
                textposition="top center",
                name="Query",
                legendgroup="query",
                hovertext=[f"Query: {_wrap(label)}"],
                hoverinfo="text",
            )
        )

    ev = pca.explained_variance_ratio_ * 100
    title = "Patient memory map" + (f" (stage {stage})" if stage is not None else "")
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=f"PC1 ({ev[0]:.1f}%)",
        yaxis_title=f"PC2 ({ev[1]:.1f}%)",
        height=640,
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
    )
    fig.layout.meta = {"drift": {str(df.at[i, "key"]): d for i, d in drift_by_pos.items()}}
    return fig


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _kpi(label: str, value: str) -> str:
    return f"<article><small>{html.escape(label)}</small><h3>{html.escape(value)}</h3></article>"


def _table(df: pd.DataFrame, drift: dict[str, float]) -> str:
    rows = df[df["corrupted"]].copy()
    rows["drift"] = rows["key"].map(drift).fillna(0.0)
    rows = rows.sort_values("drift", ascending=False)
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(r['original_text']))}</td>"
        f"<td>{html.escape(str(r['text']))}</td>"
        f"<td>{r['drift']:.3f}</td>"
        f"<td>{html.escape(str(r['corruption_ops']) or '-')}</td>"
        f"<td>{html.escape(str(r['era']) or '-')}</td>"
        "</tr>"
        for _, r in rows.iterrows()
    )
    head = "".join(
        f"<th>{h}</th>" for h in ("Original", "Stored", "Cosine Drift", "Corruption Ops", "Era")
    )
    return (
        f"<details><summary>Degraded memories ({len(rows)})</summary>"
        f"<figure><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></figure>"
        "</details>"
    )


def write_memory_map(
    fig: go.Figure,
    path: Path,
    *,
    df: pd.DataFrame | None = None,
    stage: int | None = None,
    query_label: str | None = None,
) -> Path:
    """Write the standalone offline report (Pico.css + inlined Plotly) and return `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df if df is not None else pd.DataFrame(columns=_COLUMNS)
    drift: dict[str, float] = dict((fig.layout.meta or {}).get("drift", {}))

    total = len(frame)
    corrupted = int(frame["corrupted"].sum()) if total else 0
    facts = int((frame["kind"] == "fact").sum()) if total else 0
    mean_drift = f"{np.mean(list(drift.values())):.3f}" if drift else "n/a"
    pct = f" ({100 * corrupted / total:.0f}%)" if total else ""

    badge = f"<mark>Stage {stage}</mark> " if stage is not None else ""
    banner = (
        f"<p>Query: <code>{html.escape(query_label)}</code> (red star; dotted lines to the 3 "
        "nearest stored memories)</p>"
        if query_label
        else ""
    )
    plot = fig.to_html(include_plotlyjs=True, full_html=False)
    css = _PICO_CSS_PATH.read_text(encoding="utf-8")
    title = "Patient memory map" + (f" - stage {stage}" if stage is not None else "")
    doc = f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<main class="container">
<header>
<h1>{badge}Patient memory map</h1>
<p>PCA projection of stored memory vectors. Arrows run from the ground-truth embedding to the
stored (degraded) one.</p>
{banner}
</header>
<section class="grid">
{_kpi("Total memories", str(total))}
{_kpi("Corrupted", f"{corrupted}{pct}")}
{_kpi("Biographical facts", str(facts))}
{_kpi("Mean cosine drift", mean_drift)}
</section>
<article>{plot}</article>
{_table(frame, drift) if total else ""}
</main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
