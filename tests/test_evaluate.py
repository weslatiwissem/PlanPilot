from agent1_scoping import rag  # isort: skip  (chromadb must load before pandas)
from agent1_scoping.evaluate_estimates import evaluate  # isort: skip
from agent1_scoping.index_stories import index_stories  # isort: skip

import hashlib  # isort: skip

import pandas as pd  # isort: skip


class FakeEmbedding:
    """Bag-of-words hashing: no model download in tests."""

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 64
            for w in text.lower().replace(".", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out

    def name(self):
        return "fake"


def _df(rows, split):
    return pd.DataFrame(
        [dict(project=p, title=t, description="", storypoint=s, split_mark=split) for p, t, s in rows]
    )


def test_evaluate_returns_all_predictors(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "c"))
    train = _df(
        [("a", "google login oauth", 5), ("a", "pdf report export", 3), ("b", "google login", 20), ("b", "pdf export", 13)],
        "train",
    )
    test = _df([("a", "google login", 5), ("b", "pdf report", 13)], "test")
    ef = FakeEmbedding()
    col = rag.get_collection(embedding_function=ef, create=True)
    index_stories(train, col)

    res = evaluate(train, test, col, ef, k=1)
    assert set(res) == {"global_median", "project_median", "knn_cross_project", "knn_within_project"}
    # within-project neighbours know each team's scale -> perfect on this toy data
    assert res["knn_within_project"]["mae"] == 0.0
    assert res["global_median"]["mae"] > 0
