import hashlib  # isort: skip

from agent1_scoping import rag  # noqa: F401  # isort: skip (chromadb must load before pandas)
from agent1_scoping.train_estimator import embed_cached, run  # isort: skip

import numpy as np  # isort: skip
import pandas as pd  # isort: skip


def _embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in t.lower().replace(".", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        out.append(v)
    return np.array(out)


def _make(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        project = rng.choice(["a", "b"])
        scale = 2 if project == "a" else 10
        big = rng.random() < 0.5
        rows.append(
            dict(project=project, title="big migration task" if big else "small text fix", description="",
                 storypoint=scale * (3 if big else 1))
        )
    return pd.DataFrame(rows)


def test_text_signal_beats_project_median(tmp_path):
    tr, va, te = _make(400, 1), _make(100, 2), _make(100, 3)
    res = run(tr, va, te, _embed(list(tr.title)), _embed(list(va.title)), _embed(list(te.title)))
    m = res["metrics"]
    assert m["lightgbm_team_scaled"]["mae"] < m["project_median"]["mae"]
    assert res["projects"] == 2
    # the planted signal lives in the text, so removing the text must hurt
    assert m["lightgbm_team_scaled"]["mae"] < m["lightgbm_no_text"]["mae"]
    mean, lo, hi = res["bootstrap"]["text model vs no-text model"]
    assert hi < 0 and lo <= mean <= hi


def test_bootstrap_interval_contains_mean():
    from agent1_scoping.train_estimator import paired_bootstrap

    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = paired_bootstrap(a, b)
    assert mean == -1.0 and lo == hi == -1.0


def test_embedding_cache_is_reused(tmp_path):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return _embed(texts)

    cache = str(tmp_path / "e.npz")
    a = embed_cached(["x y", "z"], embed, cache)
    b = embed_cached(["x y", "z"], embed, cache)
    assert np.allclose(a, b) and len(calls) == 1
