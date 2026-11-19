"""Tests for the read-only memory map (`memory.visualize` and the `memory-map` CLI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from cli.app import app
from memory.client import PatientMemoryClient
from memory.embeddings import HashingEmbedder
from memory.visualize import (
    _COLUMNS,
    build_memory_map,
    load_memory_records,
    project_2d,
    write_memory_map,
)

_PROFILE = Path(__file__).resolve().parents[1] / "config/patient_profiles/stage7_late.json"


def _frame(embedder: HashingEmbedder, n_clean: int, n_corrupt: int) -> pd.DataFrame:
    rows = []
    for i in range(n_clean + n_corrupt):
        corrupted = i >= n_clean
        text = f"memory about garden {i}"
        rows.append(
            {
                "key": f"episodic:{i}",
                "kind": "fact" if i == 0 else "episodic",
                "text": text + " <b>x</b>",
                "original_text": f"original {text}" if corrupted else text,
                "corrupted": corrupted,
                "corruption_level": 0.5 if corrupted else 0.0,
                "corruption_ops": "swap" if corrupted else "",
                "era": "childhood",
                "emotional_valence": "positive",
                "category": "family",
                "access_count": i,
                "source": "seed",
                "vector": embedder.embed(text),
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)


def test_project_2d_deterministic_and_separates_clusters() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (10, 8)) + np.eye(8)[0]
    b = rng.normal(0, 0.05, (10, 8)) + np.eye(8)[1]
    x = np.vstack([a, b])
    out = project_2d(x)
    assert out.shape == (20, 2)
    np.testing.assert_array_equal(out, project_2d(x))
    gap = abs(out[:10, 0].mean() - out[10:, 0].mean())
    assert gap > 5 * max(out[:10, 0].std(), out[10:, 0].std())


def test_project_2d_edge_cases() -> None:
    assert project_2d(np.zeros((0, 8))).shape == (0, 2)
    np.testing.assert_array_equal(project_2d(np.ones((1, 8))), [[0.0, 0.0]])
    assert np.all(project_2d(np.ones((4, 8))) == 0)
    assert np.all(np.isfinite(project_2d(np.arange(6.0).reshape(6, 1))))


def test_drift_traces_match_corrupted_count() -> None:
    emb = HashingEmbedder()
    fig = build_memory_map(_frame(emb, 3, 2), emb, stage=7)
    drift = [t for t in fig.data if t.legendgroup == "drift"]
    assert len(drift) == 2
    assert sum(bool(t.showlegend) for t in drift) == 1


def test_empty_frame_renders() -> None:
    emb = HashingEmbedder()
    fig = build_memory_map(pd.DataFrame(columns=_COLUMNS), emb)
    assert "no memories" in fig.layout.title.text.lower()


def test_write_is_offline_and_styled(tmp_path: Path) -> None:
    emb = HashingEmbedder()
    df = _frame(emb, 3, 2)
    fig = build_memory_map(df, emb, query_vector=emb.embed("garden"), query_label="garden")
    path = write_memory_map(fig, tmp_path / "sub" / "m.html", df=df, stage=7, query_label="garden")
    text = path.read_text(encoding="utf-8")
    assert '<script src="http' not in text
    assert '<link href="http' not in text
    assert "<style>" in text and "Pico CSS" in text
    assert "&lt;b&gt;x&lt;/b&gt;" in text  # stored text is escaped in the table


def test_cli_memory_map_end_to_end(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    out = tmp_path / "map.html"
    runner = CliRunner()
    common = ["--db-path", str(db), "--profile", str(_PROFILE)]
    assert (
        runner.invoke(
            app, ["age-memory", "--stage", "7", "--passes", "2", "--seed", "1", *common]
        ).exit_code
        == 0
    )

    def snapshot() -> list[dict]:
        client = PatientMemoryClient(db_path=db)
        try:
            return [
                r for c in ("episodic_memory", "biographical_facts") for r in client.all_records(c)
            ]
        finally:
            client.close()

    before = snapshot()
    result = runner.invoke(
        app,
        [
            "memory-map",
            "--stage",
            "7",
            "--db-path",
            str(db),
            "--out",
            str(out),
            "--query",
            "tell me about your garden",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert snapshot() == before  # strictly read-only

    client = PatientMemoryClient(db_path=db)
    try:
        df = load_memory_records(client)
    finally:
        client.close()
    fig = build_memory_map(
        df, HashingEmbedder(), query_vector=HashingEmbedder().embed("q"), query_label="q"
    )
    arrows = [t for t in fig.data if t.legendgroup == "drift"]
    assert len(arrows) == int(df["corrupted"].sum()) > 0
    assert any(t.name == "Query" for t in fig.data)
    assert "Query" in out.read_text(encoding="utf-8")


def test_cli_memory_map_missing_db(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["memory-map", "--db-path", str(tmp_path / "nope.db")])
    assert result.exit_code == 1


def test_tsne_layout_has_hubs_and_is_deterministic() -> None:
    emb = HashingEmbedder()
    df = _frame(emb, 10, 3)
    a = build_memory_map(df, emb, query_vector=emb.embed("garden"), query_label="garden")
    b = build_memory_map(df, emb, query_vector=emb.embed("garden"), query_label="garden")
    assert a.layout.meta["projection"] == "tsne"
    assert len(a.layout.annotations) >= 1  # one red hub label per cluster
    assert any(t.name == "Cluster hub" for t in a.data)
    np.testing.assert_allclose(a.data[1].x, b.data[1].x)
    assert a.layout.xaxis.title.text == "t-SNE 1"


def test_tiny_input_falls_back_to_pca_and_pca_flag() -> None:
    emb = HashingEmbedder()
    tiny = build_memory_map(_frame(emb, 2, 0), emb)
    assert tiny.layout.meta["projection"] == "pca"
    pca = build_memory_map(_frame(emb, 8, 2), emb, projection="pca")
    assert pca.layout.xaxis.title.text.startswith("PC1 (")


def _turns() -> list[dict]:
    hit = {
        "text": "memory about garden 3 <b>x</b>",
        "score": 0.4,
        "collection": "episodic_memory",
        "noise_type": "swap",
        "metadata": {"id": 3},
    }
    return [
        {
            "turn": 1,
            "trainee": "tell me about the garden",
            "patient": "*smiles*",
            "distress_state": "calm",
            "constraints": {"max_words": 3, "memory_fog": False},
            "memory_hits": [hit],
            "latency_ms": 12.0,
            "memory_encoded": True,
        },
        {
            "turn": 2,
            "trainee": "how are you",
            "patient": "*silence*",
            "distress_state": "confused",
            "constraints": {"max_words": 0, "memory_fog": True},
            "memory_hits": [],
        },
    ]


def test_transcript_replay_payload_and_panel(tmp_path: Path) -> None:
    emb = HashingEmbedder()
    df = _frame(emb, 8, 2)
    # hit id 3 -> key "episodic:3"; text fallback also works for unknown ids
    fig = build_memory_map(df, emb, stage=7, turns=_turns())
    payload = fig.layout.meta["turns"]
    assert len(payload) == 2 and len(payload[0]["hits"]) == 1 and payload[1]["hits"] == []
    assert payload[0]["hits"][0]["noise"] == "swap"
    names = {t.name for t in fig.data}
    assert {"Sequence recall", "Sequence hits", "Sequence cue"} <= names

    out = write_memory_map(fig, tmp_path / "seq.html", df=df, stage=7)
    text = out.read_text(encoding="utf-8")
    assert "Communication sequence" in text and 'id="turns-data"' in text
    assert "</script><script" not in text.split('id="turns-data"')[1].split("</script>")[0]
    assert '<script src="http' not in text


def test_cli_transcript_replay(tmp_path: Path) -> None:
    import json

    db, out, tr = tmp_path / "m.db", tmp_path / "o.html", tmp_path / "t.json"
    runner = CliRunner()
    assert runner.invoke(app, ["seed-memory", "--db-path", str(db)]).exit_code == 0
    tr.write_text(json.dumps({"turns": _turns()}), encoding="utf-8")
    res = runner.invoke(
        app,
        ["memory-map", "--db-path", str(db), "--out", str(out), "--transcript", str(tr)],
    )
    assert res.exit_code == 0, res.output
    assert "Communication sequence" in out.read_text(encoding="utf-8")
    missing = runner.invoke(
        app, ["memory-map", "--db-path", str(db), "--transcript", str(tmp_path / "x.json")]
    )
    assert missing.exit_code == 1
