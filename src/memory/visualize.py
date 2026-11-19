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

BG_COLOR = "#0f0d10"
CLEAN_COLOR = "#e8e8ea"
CORRUPT_COLOR = "#ff9f43"
FACT_COLOR = "#4aa3df"
QUERY_COLOR = "#2ecc71"
HUB_COLOR = "#e63946"
SPOKE_COLOR = "rgba(255,255,255,0.35)"

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
# Layout: projection + clustering
# --------------------------------------------------------------------------- #

_MIN_TSNE_POINTS = 5


def _layout_2d(
    stored: np.ndarray, extra: np.ndarray, projection: str
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """Return (stored_xy, extra_xy, x_title, y_title, method_used).

    PCA is fitted on stored vectors only and `extra` (ground-truth originals, query) is
    projected into that basis. t-SNE has no out-of-sample transform, so it embeds
    stored + extra jointly (as in the Milvus visualization guide). t-SNE needs a handful
    of points, so tiny inputs fall back to PCA.
    """
    n = len(stored)
    if projection == "tsne" and n + len(extra) >= _MIN_TSNE_POINTS:
        from sklearn.manifold import TSNE

        allv = np.vstack([stored, extra]) if len(extra) else stored
        xy = TSNE(
            n_components=2,
            perplexity=min(30.0, (len(allv) - 1) / 3),
            init="pca",
            metric="cosine",
            learning_rate="auto",
            random_state=0,
        ).fit_transform(allv.astype(np.float64))
        return xy[:n], xy[n:], "t-SNE 1", "t-SNE 2", "tsne"
    pca = PCAProjector().fit(stored)
    ev = pca.explained_variance_ratio_ * 100
    return (
        pca.transform(stored),
        pca.transform(extra) if len(extra) else np.zeros((0, 2)),
        f"PC1 ({ev[0]:.1f}%)",
        f"PC2 ({ev[1]:.1f}%)",
        "pca",
    )


def _cluster(stored: np.ndarray) -> np.ndarray:
    """Group memories by similarity in the original high-dimensional space."""
    n = len(stored)
    if n < 3:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import KMeans

    k = int(min(n, max(2, round(float(np.sqrt(n / 2))))))
    return np.asarray(KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(stored))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

_KIND_BLURB = {
    "intact": "Intact episodic memory: stored text still matches what originally happened.",
    "corrupted": "Corrupted episodic memory: the stored text/vector has degraded from the original.",
    "fact": "Biographical fact: a discrete fact the patient 'knows' (name, job, family...).",
    "fact_corrupted": "Corrupted biographical fact: the stored fact no longer matches the truth.",
}


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise `1 - cos(a, b)`; 0.0 where either vector has zero norm."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sim = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    return np.asarray(1.0 - sim)


def _wrap(text: str, width: int = 60) -> str:
    return "<br>".join(html.escape(line) for line in textwrap.wrap(text, width) or [""])


def _short(text: str, n: int = 26) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _hover(row: pd.Series, drift: float | None) -> str:
    corrupted = bool(row["corrupted"])
    if row["kind"] == "fact":
        blurb = _KIND_BLURB["fact_corrupted" if corrupted else "fact"]
    else:
        blurb = _KIND_BLURB["corrupted" if corrupted else "intact"]
    parts = [f"<i>{html.escape(blurb)}</i>", f"<b>Stored:</b> {_wrap(str(row['text']))}"]
    if corrupted:
        parts.append(f"<b>Original:</b> {_wrap(str(row['original_text']))}")
        if drift is not None:
            parts.append(f"<b>Cosine drift:</b> {drift:.3f} (0 = unchanged, higher = further)")
        if row["corruption_ops"]:
            parts.append(f"<b>What went wrong:</b> {html.escape(str(row['corruption_ops']))}")
    if row["era"]:
        parts.append(f"<b>Era:</b> {html.escape(str(row['era']))}")
    if row["kind"] == "fact" and row["category"]:
        parts.append(f"<b>Category:</b> {html.escape(str(row['category']))}")
    if row["emotional_valence"]:
        parts.append(f"<b>Emotional valence:</b> {html.escape(str(row['emotional_valence']))}")
    parts.append(
        f"<b>Recalled:</b> {int(row['access_count'])}x (brighter = rehearsed more) "
        f"· <b>Source:</b> {html.escape(str(row['source']))}"
    )
    return "<br>".join(parts)


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="No memories found",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=420,
    )
    fig.layout.meta = {"drift": {}, "projection": "none"}
    return fig


def build_memory_map(
    df: pd.DataFrame,
    embedder: Embedder,
    *,
    query_vector: np.ndarray | None = None,
    query_label: str | None = None,
    stage: int | None = None,
    projection: str = "tsne",
) -> go.Figure:
    """Build the 2D map. Per-record cosine drift is stored in `fig.layout.meta["drift"]`."""
    if df.empty:
        return _empty_figure()

    df = df.reset_index(drop=True)
    stored = np.stack(df["vector"].to_numpy())

    # Ground truth: re-embed originals with the store's own embedder.
    corrupted_idx = np.flatnonzero(df["corrupted"].to_numpy(dtype=bool)).tolist()
    orig_vecs = np.zeros((0, stored.shape[1]))
    drift_by_pos: dict[int, float] = {}
    if corrupted_idx:
        orig_vecs = np.asarray(
            embedder.embed_batch([str(df.at[i, "original_text"]) for i in corrupted_idx]),
            dtype=np.float64,
        )
        drifts = _cosine_distance(orig_vecs, stored[corrupted_idx])
        drift_by_pos = {i: float(d) for i, d in zip(corrupted_idx, drifts, strict=True)}
    q = None if query_vector is None else np.asarray(query_vector, dtype=np.float64).reshape(1, -1)
    extra = np.vstack([orig_vecs, q]) if q is not None else orig_vecs
    xy, extra_xy, x_title, y_title, used = _layout_2d(stored, extra, projection)
    orig_xy = extra_xy[: len(corrupted_idx)]

    fig = go.Figure()

    # --- cluster hubs and spokes (the "constellation" structure) ---------------
    labels = _cluster(stored)
    spoke_x: list[float | None] = []
    spoke_y: list[float | None] = []
    hubs: list[tuple[float, float, str, str]] = []
    for c in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == c)
        hx, hy = float(xy[members, 0].mean()), float(xy[members, 1].mean())
        for m in members:
            spoke_x += [hx, float(xy[m, 0]), None]
            spoke_y += [hy, float(xy[m, 1]), None]
        centroid = stored[members].mean(axis=0)
        rep = members[int(np.argmin(np.linalg.norm(stored[members] - centroid, axis=1)))]
        n_bad = int(df.iloc[members]["corrupted"].sum())
        label = f"{_short(str(df.at[rep, 'text']))} · {len(members)}"
        tip = (
            f"<b>Memory cluster {c + 1}</b><br>{len(members)} memories that are similar in "
            f"meaning, {n_bad} corrupted.<br>Most representative: {_wrap(str(df.at[rep, 'text']))}"
            "<br><i>The hub is the average position of the group; spokes run to each member.</i>"
        )
        hubs.append((hx, hy, label, tip))
    fig.add_trace(
        go.Scatter(
            x=spoke_x,
            y=spoke_y,
            mode="lines",
            line={"color": SPOKE_COLOR, "width": 0.8},
            name="Cluster spokes",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[h[0] for h in hubs],
            y=[h[1] for h in hubs],
            mode="markers",
            marker={
                "symbol": "circle",
                "size": 9,
                "color": HUB_COLOR,
                "line": {"color": "white", "width": 1},
            },
            name="Cluster hub",
            hovertext=[h[3] for h in hubs],
            hoverinfo="text",
        )
    )
    for i, (hx, hy, label, tip) in enumerate(hubs):
        fig.add_annotation(
            x=hx,
            y=hy,
            text=label,
            showarrow=True,
            arrowhead=0,
            arrowcolor=HUB_COLOR,
            arrowwidth=1.2,
            ax=46 if i % 2 == 0 else -46,
            ay=-34 if i % 3 else 34,
            bgcolor=HUB_COLOR,
            font={"color": "white", "size": 11},
            borderpad=3,
            hovertext=tip,
            captureevents=True,
        )

    # --- memory points -----------------------------------------------------------
    opacity = np.clip(0.7 + 0.1 * df["access_count"].to_numpy(dtype=float), 0.7, 1.0)
    is_fact = (df["kind"] == "fact").to_numpy()
    is_bad = df["corrupted"].to_numpy(dtype=bool)
    groups = [
        ("Intact episodic memory", ~is_fact & ~is_bad, "square", CLEAN_COLOR),
        ("Corrupted episodic memory", ~is_fact & is_bad, "square", CORRUPT_COLOR),
        ("Biographical fact", is_fact & ~is_bad, "diamond", FACT_COLOR),
        ("Corrupted biographical fact", is_fact & is_bad, "diamond", FACT_COLOR),
    ]
    for name, mask, symbol, color in groups:
        pos = np.flatnonzero(mask)
        if not len(pos):
            continue
        sub = df.iloc[pos]
        bad = name.startswith("Corrupted")
        size = (
            (8 + 9 * sub["corruption_level"].clip(0, 1)).to_numpy()
            if bad and symbol == "square"
            else np.full(len(pos), 12 if symbol == "diamond" else 9)
        )
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
                    "line": {
                        "color": CORRUPT_COLOR if bad else BG_COLOR,
                        "width": 2 if bad else 0.5,
                    },
                },
                hovertext=[
                    _hover(r, drift_by_pos.get(int(p)))
                    for p, (_, r) in zip(pos, sub.iterrows(), strict=True)
                ],
                hoverinfo="text",
            )
        )

    # --- degradation drift arrows: ground truth -> stored ---------------------------
    for i, (pos, (ox, oy)) in enumerate(zip(corrupted_idx, orig_xy, strict=True)):
        row = df.iloc[pos]
        hover = (
            "<i>Degradation drift: arrow runs from where this memory SHOULD sit (original "
            "text) to where it is actually stored.</i>"
            f"<br><b>Cosine drift:</b> {drift_by_pos[pos]:.3f}"
            f"<br><b>Ops:</b> {html.escape(str(row['corruption_ops']) or 'n/a')}"
            f"<br><b>Original:</b> {_wrap(str(row['original_text']))}"
            f"<br><b>Stored:</b> {_wrap(str(row['text']))}"
        )
        fig.add_trace(
            go.Scatter(
                x=[ox, xy[pos, 0]],
                y=[oy, xy[pos, 1]],
                mode="lines+markers",
                line={"color": CORRUPT_COLOR, "width": 1.2},
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

    # --- optional query ------------------------------------------------------------
    if q is not None:
        qxy = extra_xy[-1]
        sims = stored @ q[0] / np.maximum(np.linalg.norm(stored, axis=1) * np.linalg.norm(q), 1e-12)
        nearest = np.argsort(-sims)[:3]
        lx: list[float | None] = []
        ly: list[float | None] = []
        for k in nearest:
            lx += [float(qxy[0]), float(xy[k, 0]), None]
            ly += [float(qxy[1]), float(xy[k, 1]), None]
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line={"color": QUERY_COLOR, "width": 1.2, "dash": "dot"},
                name="Top-3 nearest to query",
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
                textfont={"color": QUERY_COLOR},
                name="Query",
                legendgroup="query",
                hovertext=[
                    f"<b>Your cue:</b> {_wrap(label)}<br><i>Dotted lines point to the 3 stored "
                    "memories most similar to it: what the patient would most likely recall.</i>"
                ],
                hoverinfo="text",
            )
        )

    title = "Patient memory map" + (f" (stage {stage})" if stage is not None else "")
    axis = {"showgrid": False, "zeroline": False}
    if used == "tsne":
        axis |= {"showticklabels": False}
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis={"title": x_title, **axis},
        yaxis={"title": y_title, **axis},
        height=700,
        legend={"orientation": "h", "y": -0.1},
        margin={"l": 40, "r": 20, "t": 50, "b": 60},
        hoverlabel={"align": "left", "bgcolor": "#1c1a1f", "font": {"size": 12}},
    )
    fig.layout.meta = {
        "drift": {str(df.at[i, "key"]): d for i, d in drift_by_pos.items()},
        "projection": used,
    }
    return fig


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_EXTRA_CSS = f"""
:root{{--pico-background-color:{BG_COLOR};--pico-card-background-color:#17141a;}}
body{{background:{BG_COLOR};overflow-x:hidden;}}
article{{overflow:hidden;}}
article h3{{margin-bottom:0;}}
.key{{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1rem;}}
.key span{{border:1px solid #3a3640;border-radius:999px;padding:.15rem .7rem;font-size:.8rem;
cursor:help;border-bottom:1px solid #3a3640;text-decoration:none;}}
.sw{{display:inline-block;width:.7rem;height:.7rem;margin-right:.4rem;}}
"""

_KEY = [
    (CLEAN_COLOR, "0", "Intact memory", "A memory whose stored text still matches the original."),
    (
        CORRUPT_COLOR,
        "0",
        "Corrupted memory",
        "Text was altered by encoding noise or decay. Bigger = more corrupted.",
    ),
    (
        FACT_COLOR,
        "45deg",
        "Biographical fact",
        "A discrete fact (name, job, family). Orange outline = corrupted.",
    ),
    (
        CORRUPT_COLOR,
        "0",
        "Orange arrow",
        "Drift: from where the memory should sit (original text) to where it is stored.",
    ),
    (
        HUB_COLOR,
        "50%",
        "Red label",
        "A cluster hub: the centre of a group of similar memories, with spokes to each.",
    ),
    (
        QUERY_COLOR,
        "50%",
        "Green star",
        "Your --query cue; dotted lines go to its 3 nearest memories.",
    ),
]


def _kpi(label: str, value: str, tip: str) -> str:
    return (
        f'<article data-tooltip="{html.escape(tip)}" data-placement="bottom">'
        f"<small>{html.escape(label)}</small><h3>{html.escape(value)}</h3></article>"
    )


def _key_html() -> str:
    chips = "".join(
        f'<span data-tooltip="{html.escape(tip)}"><i class="sw" style="background:{c};'
        f'border-radius:{r}"></i>{html.escape(name)}</span>'
        for c, r, name, tip in _KEY
    )
    return f'<div class="key">{chips}</div>'


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
        f"<details><summary>Degraded memories ({len(rows)}): original vs. what is stored</summary>"
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
    meta = fig.layout.meta or {}
    drift: dict[str, float] = dict(meta.get("drift", {}))
    used = str(meta.get("projection", "none"))

    total = len(frame)
    corrupted = int(frame["corrupted"].sum()) if total else 0
    facts = int((frame["kind"] == "fact").sum()) if total else 0
    mean_drift = f"{np.mean(list(drift.values())):.3f}" if drift else "n/a"
    pct = f" ({100 * corrupted / total:.0f}%)" if total else ""

    badge = f"<mark>Stage {stage}</mark> " if stage is not None else ""
    banner = (
        f"<p>Query: <code>{html.escape(query_label)}</code>: green star, dotted lines to the 3 "
        "nearest stored memories.</p>"
        if query_label
        else ""
    )
    how = {
        "tsne": "Each dot is one memory, placed by <b>t-SNE</b> so that memories with similar "
        "meaning sit close together. Distances inside a cluster are meaningful; the axes and "
        "the gaps between far-apart clusters are not.",
        "pca": "Each dot is one memory, placed by <b>PCA</b> (the two directions of greatest "
        "variation). Axis titles show how much variance each captures.",
    }.get(used, "")
    plot = fig.to_html(include_plotlyjs=True, full_html=False)
    css = _PICO_CSS_PATH.read_text(encoding="utf-8") + _EXTRA_CSS
    title = "Patient memory map" + (f" - stage {stage}" if stage is not None else "")
    doc = f"""<!doctype html>
<html lang="en" data-theme="dark">
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
<p>A researcher's view of what the simulated patient remembers, and how it has degraded.
Hover anything for an explanation.</p>
{banner}
</header>
<section class="grid">
{_kpi("Total memories", str(total), "Every episodic memory and biographical fact stored for this patient.")}
{_kpi("Corrupted", f"{corrupted}{pct}", "Memories whose stored text was altered by encoding noise or decay passes.")}
{_kpi("Biographical facts", str(facts), "Discrete facts about the patient (as opposed to episodes).")}
{_kpi("Mean cosine drift", mean_drift, "Average of 1 - cosine similarity between each corrupted memory's original and stored vectors. 0 = identical, higher = more degraded.")}
</section>
{_key_html()}
<article>{plot}</article>
<details open><summary>How to read this map</summary>
<p>{how}</p>
<p>Groups of dots joined by thin spokes are <b>clusters</b> of related memories; the red label
names the most representative one and the group size. An <b>orange arrow</b> shows one
memory's drift from its ground truth (where its original text would sit) to where the
degraded vector actually sits: a longer arrow means a worse distortion. Brighter dots have
been recalled more often (rehearsal).</p>
</details>
{_table(frame, drift) if total else ""}
</main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
