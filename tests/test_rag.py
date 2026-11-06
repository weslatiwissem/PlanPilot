import hashlib

import pytest

from agent1_scoping import rag  # isort: skip  (chromadb must load before pandas)
from agent1_scoping.index_stories import index_stories, load_stories  # isort: skip

import pandas as pd  # isort: skip


class FakeEmbedding:
    """Bag-of-words hashing: similar words -> similar vectors, no model download."""

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


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "sp"
    d.mkdir()
    pd.DataFrame(
        {
            "issuekey": ["A-1", "A-2", "A-3", "A-4"],
            "title": ["Google login with OAuth", "Export report to PDF", "Fix typo", "Hidden test row"],
            "description": ["Users sign in with Google", "Generate a pdf report", "", "x"],
            "storypoint": [5, 3, 1, 8],
            "split_mark": ["train", "train", "train", "test"],
        }
    ).to_csv(d / "projA.csv", index=False)
    return str(d)


def test_load_keeps_train_rows_only(data_dir):
    df = load_stories(data_dir)
    assert len(df) == 3 and "Hidden test row" not in set(df["title"])
    assert len(load_stories(data_dir, train_only=False)) == 4


def test_missing_columns_give_clear_error(tmp_path):
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_stories(str(tmp_path))


def test_index_then_retrieve(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    col = rag.get_collection(embedding_function=FakeEmbedding(), create=True)
    assert index_stories(load_stories(data_dir), col) == 3

    hits = rag.retrieve_similar_stories("sign in with google", k=2, collection=col)
    assert hits[0].title == "Google login with OAuth" and hits[0].points == 5
    assert hits[0].project == "projA"


def test_missing_index_returns_no_context(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "empty"))
    assert rag.retrieve_similar("anything") == []
