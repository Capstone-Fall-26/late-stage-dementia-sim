"""Read-only 2D visual map of a patient's memory vectors with degradation drift.

Pipeline: `load_memory_records` -> `build_memory_map` (PCA fitted on the *stored*
vectors; ground-truth originals and an optional query are projected into the same
basis) -> `write_memory_map` (single offline HTML: vendored Pico.css + inlined Plotly).

Nothing here writes to the store, bumps `access_count`, or calls an LLM.
"""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import Embedder

_ASSETS = Path(__file__).parent / "assets"
_PICO_CSS_PATH = _ASSETS / "pico.min.css"

BG_COLOR = "#0b0d14"
CLEAN_COLOR = "#e8e8ea"
CORRUPT_COLOR = "#ff9f43"
SEQ_COLOR = "#22d3ee"
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


def load_transcript(path: Path) -> list[dict[str, Any]]:
    """Read the `turns` list from a transcript JSON saved by `chat`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = data.get("turns", []) if isinstance(data, dict) else data
    return [t for t in turns if isinstance(t, dict) and t.get("trainee")]


def _locate_hit(df: pd.DataFrame, hit: dict[str, Any]) -> int | None:
    """Find the map row for a recalled hit: by collection+id, else by text."""
    kind = "episodic" if hit.get("collection") == EPISODIC_COLLECTION else "fact"
    hit_id = (hit.get("metadata") or {}).get("id")
    rows = df.index[df["key"] == f"{kind}:{hit_id}"]
    if len(rows):
        return int(rows[0])
    texts = {str(hit.get("text", "")), str(hit.get("original_text") or "")} - {""}
    same_kind = df[df["kind"] == kind]
    for col in ("text", "original_text"):
        found = same_kind.index[same_kind[col].isin(texts)]
        if len(found):
            return int(found[0])
    return None


def _turn_payload(
    df: pd.DataFrame, xy: np.ndarray, turns: list[dict[str, Any]], turn_xy: np.ndarray
) -> list[dict[str, Any]]:
    """Per-turn data for the in-page communication-sequence replay."""
    out: list[dict[str, Any]] = []
    for i, (turn, (qx, qy)) in enumerate(zip(turns, turn_xy, strict=True)):
        hits = []
        for hit in turn.get("memory_hits", []) or []:
            pos = _locate_hit(df, hit)
            if pos is None:
                continue
            hits.append(
                {
                    "x": float(xy[pos, 0]),
                    "y": float(xy[pos, 1]),
                    "score": float(hit.get("score", 0.0)),
                    "noise": hit.get("noise_type"),
                    "coll": hit.get("collection", ""),
                    "text": str(hit.get("text", "")),
                }
            )
        out.append(
            {
                "n": int(turn.get("turn", i + 1)),
                "trainee": str(turn.get("trainee", "")),
                "patient": str(turn.get("patient", "")),
                "distress": str(turn.get("distress_state", "")),
                "constraints": turn.get("constraints") or {},
                "latency_ms": float(turn.get("latency_ms", 0.0) or 0.0),
                "encoded": turn.get("memory_encoded"),
                "q": [float(qx), float(qy)],
                "hits": hits,
            }
        )
    return out


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
    turns: list[dict[str, Any]] | None = None,
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
    turn_vecs = (
        np.asarray(
            embedder.embed_batch([str(t.get("trainee", "")) for t in turns]), dtype=np.float64
        )
        if turns
        else np.zeros((0, stored.shape[1]))
    )
    extra = np.vstack([orig_vecs, q, turn_vecs] if q is not None else [orig_vecs, turn_vecs])
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
            ax=52 if i % 2 == 0 else -52,
            ay=-40 if i % 3 else 40,
            bgcolor=HUB_COLOR,
            font={"color": "white", "size": 12},
            borderpad=5,
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
        qxy = extra_xy[len(corrupted_idx)]
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

    turn_payload: list[dict[str, Any]] = []
    if turns:
        turn_xy = extra_xy[len(extra_xy) - len(turns) :]
        turn_payload = _turn_payload(df, xy, turns, turn_xy)
        seq_style = {"legendgroup": "sequence", "hoverinfo": "skip"}
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Sequence recall",
                line={"color": SEQ_COLOR, "width": 2, "dash": "dash"},
                **seq_style,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                name="Sequence hits",
                marker={"symbol": "circle-open", "size": 24, "line": {"width": 3}},
                **seq_style,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers+text",
                name="Sequence cue",
                marker={
                    "symbol": "star",
                    "size": 22,
                    "color": SEQ_COLOR,
                    "line": {"color": "white", "width": 1},
                },
                text=[""],
                textposition="bottom center",
                textfont={"color": SEQ_COLOR, "size": 13},
                **seq_style,
            )
        )

    title = ""
    axis = {"showgrid": False, "zeroline": False}
    if used == "tsne":
        axis |= {"showticklabels": False}
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"title": x_title, **axis},
        yaxis={"title": y_title, **axis},
        height=700,
        legend={"orientation": "h", "y": -0.1},
        margin={"l": 70, "r": 40, "t": 30, "b": 80},
        font={"size": 13},
        hoverlabel={"align": "left", "bgcolor": "#161a28", "font": {"size": 12}},
    )
    fig.layout.meta = {
        "drift": {str(df.at[i, "key"]): d for i, d in drift_by_pos.items()},
        "projection": used,
        "turns": turn_payload,
    }
    return fig


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_ICONS = {
    "zoom-in": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M11 8v6M8 11h6"/>',
    "zoom-out": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "pan": '<path d="M12 2v20M2 12h20M12 2l-3 3M12 2l3 3M12 22l-3-3M12 22l3-3M2 12l3-3M2 12l3 3M22 12l-3-3M22 12l-3 3"/>',
    "full": '<path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/>',
    "png": '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
}

# (layer id, swatch colour, swatch radius, label, tooltip)
_LAYERS = [
    (
        "episodic",
        CLEAN_COLOR,
        "2px",
        "Episodes",
        "Episodic memories (squares). Click to show/hide. Orange = corrupted; bigger = more corrupted.",
    ),
    (
        "facts",
        FACT_COLOR,
        "2px",
        "Facts",
        "Biographical facts (diamonds). Orange outline = corrupted. Click to show/hide.",
    ),
    (
        "clusters",
        HUB_COLOR,
        "50%",
        "Clusters",
        "Hubs, spokes and red labels grouping similar memories. Click to show/hide.",
    ),
    (
        "drift",
        CORRUPT_COLOR,
        "50%",
        "Drift arrows",
        "Arrows from where a memory should sit (original text) to where it is stored. Click to show/hide.",
    ),
    (
        "query",
        QUERY_COLOR,
        "50%",
        "Query",
        "Your --query cue (green star) and its 3 nearest memories. Click to show/hide.",
    ),
]

_TOOLS = [
    ("zoom-in", "Zoom in"),
    ("zoom-out", "Zoom out"),
    ("reset", "Reset view"),
    ("sep", ""),
    ("pan", "Pan mode (drag to move). Off = drag a box to zoom"),
    ("full", "Fullscreen"),
    ("png", "Download as PNG"),
]


def _controls_html(has_query: bool, has_seq: bool) -> str:
    chips = "".join(
        f'<button type="button" class="chip" data-layer="{i}" aria-pressed="true" '
        f'data-tooltip="{html.escape(tip)}"><i style="background:{c};border-radius:{r}"></i>'
        f"{html.escape(label)}</button>"
        for i, c, r, label, tip in _LAYERS
        if (i != "query" or has_query) and (i != "sequence" or has_seq)
    )
    pressed = ' aria-pressed="false"'
    tools = "".join(
        '<span class="sep"></span>'
        if act == "sep"
        else f'<button type="button" class="tool" data-act="{act}" aria-label="{html.escape(tip)}" '
        f'data-tooltip="{html.escape(tip)}" data-placement="bottom"'
        f"{pressed if act == 'pan' else ''}>"
        f'<svg viewBox="0 0 24 24">{_ICONS[act]}</svg></button>'
        for act, tip in _TOOLS
    )
    return (
        f'<div class="bar"><div class="chips">{chips}</div><div class="tools">{tools}</div></div>'
    )


def _kpi(label: str, value: str, tip: str) -> str:
    return (
        f'<article data-tooltip="{html.escape(tip)}" data-placement="bottom">'
        f"<small>{html.escape(label)}</small><h3>{html.escape(value)}</h3></article>"
    )


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


# --------------------------------------------------------------------------- #
# Report page
# --------------------------------------------------------------------------- #

_STAGE_LABELS = {
    5: "GDS 5 / FAST 5 · Moderately severe decline",
    6: "GDS 6 / FAST 6 · Severe decline",
    7: "GDS 7 / FAST 7 · Very severe decline (speech limited to <6 words/day at 7a)",
}

_PIPELINE = [
    "Trainee utterance",
    "Trigger assessment",
    "Symptom sampling",
    "Memory recall",
    "Prompt assembly",
    "LLM generation",
    "Post-processing",
    "Memory encoding",
]


def _svg(inner: str) -> str:
    return f'<svg viewBox="0 0 32 32" aria-hidden="true">{inner}</svg>'


def _legend_html(has_query: bool, has_seq: bool) -> str:
    sq = lambda c, extra="": f'<rect x="7" y="7" width="18" height="18" rx="2" fill="{c}" {extra}/>'  # noqa: E731
    dia = lambda c, st="": f'<path d="M16 4L28 16L16 28L4 16Z" fill="{c}" {st}/>'  # noqa: E731
    items = [
        (
            _svg(sq(CLEAN_COLOR)),
            "Intact engram",
            "A stored episode whose text still matches what originally happened. Brighter = recalled more often.",
        ),
        (
            _svg(sq(CORRUPT_COLOR)),
            "Degraded engram",
            "Text altered by encoding noise or a decay pass. Larger = more corrupted.",
        ),
        (
            _svg(dia(FACT_COLOR)),
            "Biographical fact",
            "A discrete fact the patient 'knows' (name, job, family).",
        ),
        (
            _svg(dia(FACT_COLOR, f'stroke="{CORRUPT_COLOR}" stroke-width="3"')),
            "Degraded fact",
            "A fact whose stored text no longer matches the truth (amber outline).",
        ),
        (
            _svg(
                f'<circle cx="16" cy="16" r="7" fill="{HUB_COLOR}"/><path d="M16 16L30 6" stroke="{HUB_COLOR}" stroke-width="2"/>'
            ),
            "Cluster hub + red label",
            "The centre of a group of memories that are similar in meaning. Spokes run to each member; the label names the most representative one and the group size.",
        ),
        (
            _svg(
                f'<path d="M4 24L24 10" stroke="{CORRUPT_COLOR}" stroke-width="2.5"/><path d="M28 8L18 9L25 17Z" fill="{CORRUPT_COLOR}"/><circle cx="4" cy="24" r="3" fill="none" stroke="{CORRUPT_COLOR}" stroke-width="2"/>'
            ),
            "Drift arrow",
            "From where the memory should sit (its original text) to where the degraded vector actually sits. A longer arrow means a worse distortion.",
        ),
    ]
    if has_query:
        items.append(
            (
                _svg(
                    f'<path d="M16 3l3.9 8.2 9 1.1-6.6 6.2 1.7 8.9L16 23l-8 4.4 1.7-8.9-6.6-6.2 9-1.1z" fill="{QUERY_COLOR}"/>'
                ),
                "Query cue",
                "Your --query text placed in the same space; dotted lines point to the 3 stored memories most similar to it.",
            )
        )
    if has_seq:
        items.append(
            (
                _svg(
                    f'<path d="M16 3l3.9 8.2 9 1.1-6.6 6.2 1.7 8.9L16 23l-8 4.4 1.7-8.9-6.6-6.2 9-1.1z" fill="{SEQ_COLOR}"/>'
                ),
                "Sequence cue & recalled rings",
                "In the replay, the star is the trainee's turn and rings mark what the patient recalled: teal = faithful, amber = already degraded, red = noise injected at retrieval.",
            )
        )
    body = "".join(
        f'<div class="item">{icon}<div><b>{html.escape(t)}</b><span>{html.escape(d)}</span></div></div>'
        for icon, t, d in items
    )
    return f'<div class="legend"><h4>Legend</h4><div class="items">{body}</div></div>'


def _neurons_svg() -> str:
    """Deterministic decorative neuron network for the page header."""
    rng = np.random.default_rng(11)
    pts = rng.uniform([10, 10], [390, 110], size=(22, 2))
    lines, dots = [], []
    for i, (x, y) in enumerate(pts):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:3]:
            lines.append(
                f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{pts[j, 0]:.0f}" y2="{pts[j, 1]:.0f}"/>'
            )
        dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{2 + (i % 3)}"/>')
    return (
        '<svg class="neurons" viewBox="0 0 400 120" aria-hidden="true">'
        f'<g stroke="rgba(94,234,212,.45)" stroke-width="1">{"".join(lines)}</g>'
        f'<g fill="rgba(196,181,253,.8)">{"".join(dots)}</g></svg>'
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
    turns: list[dict[str, Any]] = list(meta.get("turns", []))

    total = len(frame)
    corrupted = int(frame["corrupted"].sum()) if total else 0
    facts = int((frame["kind"] == "fact").sum()) if total else 0
    mean_drift = f"{np.mean(list(drift.values())):.3f}" if drift else "n/a"
    pct = f" ({100 * corrupted / total:.0f}%)" if total else ""

    badges = ""
    if stage is not None:
        badges += f'<span class="badge stage-badge">{html.escape(_STAGE_LABELS.get(stage, f"Stage {stage}").split(" (")[0])}</span>'
    badges += f'<span class="badge">layout: {"t-SNE" if used == "tsne" else used.upper()}</span>'
    if query_label:
        badges += f'<span class="badge">query: {html.escape(query_label)}</span>'
    if turns:
        badges += f'<span class="badge">replay: {len(turns)} turns</span>'

    how = {
        "tsne": "Each point is one memory, embedded as a vector and placed by <b>t-SNE</b> so "
        "that memories with similar meaning sit close together. Distances inside a cluster "
        "are meaningful; the axes and the gaps between far-apart clusters are not.",
        "pca": "Each point is one memory, placed by <b>PCA</b> (the two directions of greatest "
        "variation). Axis titles show the variance each captures.",
    }.get(used, "")
    stage_txt = f" at {html.escape(_STAGE_LABELS[stage])}" if stage in _STAGE_LABELS else ""
    caption = (
        '<figcaption class="cap"><b>Figure 1.</b> Two-dimensional embedding of the simulated '
        f"patient's memory store{stage_txt}. {how} "
        "Amber arrows show degradation drift from each record's ground truth to its stored "
        "vector (cosine distance, hover for values).</figcaption>"
    )

    plot = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="memory-map",
        config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
    )

    seq_html = ""
    seq_js = ""
    if turns:
        pills = "".join(
            f'<button type="button" class="pill" aria-pressed="false">Turn {t["n"]}</button>'
            for t in turns
        )
        steps = "".join(
            f'<li><button type="button" class="step"><span class="n">{i + 1}</span>{html.escape(n)}</button></li>'
            for i, n in enumerate(_PIPELINE)
        )
        data = json.dumps(turns).replace("</", "<\\/")
        seq_html = f"""<aside class="seq" id="seq">
<h2>Communication sequence</h2>
<p class="sub">Step through one trainee turn. The recall step lights up the memories it pulled on the map. Arrow keys work too.</p>
<div class="row"><div class="chips">{pills}</div>
<div class="tools" style="padding:.3rem .5rem;gap:.4rem">
<button type="button" class="chip" id="seq-prev">Back</button>
<button type="button" class="chip" id="seq-play">Play</button>
<button type="button" class="chip" id="seq-next">Next</button></div></div>
<ol class="pipe">{steps}</ol>
<div class="detail" id="seq-detail"></div>
<script id="turns-data" type="application/json">{data}</script>
</aside>"""
        seq_js = f"<script>{(_ASSETS / 'sequence.js').read_text(encoding='utf-8')}</script>"

    css = _PICO_CSS_PATH.read_text(encoding="utf-8") + (_ASSETS / "report.css").read_text(
        encoding="utf-8"
    )
    report_js = (_ASSETS / "report.js").read_text(encoding="utf-8")
    title = "Memory degradation atlas" + (f" - stage {stage}" if stage is not None else "")
    doc = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<main class="container{" wide" if turns else ""}">
<header class="hero">
{_neurons_svg()}
<p class="eyebrow">Late-stage dementia simulation · researcher view</p>
<h1>Memory degradation atlas</h1>
<p class="lede">What the simulated patient remembers, how it has degraded, and how a conversation
reaches into it. Hover anything for an explanation.</p>
<div class="badges">{badges}</div>
</header>
<section class="grid">
{_kpi("Engrams stored", str(total), "Every episodic memory and biographical fact stored for this patient.")}
{_kpi("Degraded", f"{corrupted}{pct}", "Memories whose stored text was altered by encoding noise or decay passes.")}
{_kpi("Biographical facts", str(facts), "Discrete facts about the patient (as opposed to episodes).")}
{_kpi("Mean drift (1 - cos)", mean_drift, "Average of 1 - cosine similarity between each degraded memory's original and stored vectors. 0 = identical, higher = more degraded.")}
</section>
<div class="layout{" with-aside" if turns else ""}">
<article id="stage" class="stage">{_controls_html(bool(query_label), bool(turns))}{plot}{_legend_html(bool(query_label), bool(turns))}{caption}</article>
{seq_html}
</div>
{_table(frame, drift) if total else ""}
<footer class="foot">
<b>Methods.</b> Vectors come from the store's own embedder (384-d, L2-normalised). Ground truth
is re-embedded from <code>original_text</code> / <code>original_fact</code>; drift is
1 - cosine similarity to the stored vector. Clusters are k-means in the original vector
space. This view is read-only: it never changes the store. <b>Caveat.</b> Noise magnitudes
and most behavioural priors are simulation assumptions, not clinical measurements (see the
README's provenance table).
</footer>
</main>
<script>{report_js}</script>
{seq_js}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
