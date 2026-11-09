import hashlib  # isort: skip

from agent1_scoping import rag  # noqa: F401  # isort: skip (chromadb must load before pandas)
from agent1_scoping import estimator  # isort: skip
from agent1_scoping.agent import scope_project  # isort: skip
from agent1_scoping.train_estimator import run  # isort: skip
from orchestrator.schemas import Epic, Scope, Story  # isort: skip

import numpy as np  # isort: skip
import pandas as pd  # isort: skip
import pytest  # isort: skip


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
        big = rng.random() < 0.5
        rows.append(dict(project=project, title="big migration task" if big else "small text fix",
                         description="", storypoint=(2 if project == "a" else 10) * (3 if big else 1)))
    return pd.DataFrame(rows)


@pytest.fixture()
def saved(tmp_path):
    tr, va, te = _make(400, 1), _make(100, 2), _make(100, 3)
    res = run(tr, va, te, _embed(list(tr.title)), _embed(list(va.title)), _embed(list(te.title)))
    a = res["artifacts"]
    path = str(tmp_path / "est")
    estimator.save_estimator(a["model"], a["team_medians"], a["global_median"], 32, path)
    return path


def test_roundtrip_predicts_in_team_units(saved):
    big, small = estimator.predict_points([("big migration task", ""), ("small text fix", "")], "b", _embed, saved)
    assert big > 2 * small          # text signal survives save/load
    big_a, _ = estimator.predict_points([("big migration task", ""), ("x", "")], "a", _embed, saved)
    assert big_a < big              # team a's scale is smaller than team b's
    assert estimator.known_teams(saved) == ["a", "b"]


def test_unknown_team_and_missing_model(saved, tmp_path):
    with pytest.raises(KeyError, match="Unknown team"):
        estimator.predict_points([("x", "")], "zzz", _embed, saved)
    with pytest.raises(FileNotFoundError):
        estimator.load_estimator(str(tmp_path / "nothing"))


def test_calibrate_scope_rewrites_points(saved):
    scope = Scope(epics=[Epic(title="E", stories=[Story(title="big migration task", estimate_points=5)])])
    out = estimator.calibrate_scope(scope, "b", _embed, saved)
    assert out.epics[0].stories[0].estimate_points >= 10


def test_agent_uses_calibration_when_team_given(monkeypatch):
    def fake(scope, team_id, **kw):
        scope.epics[0].stories[0].estimate_points = 42
        return scope

    monkeypatch.setattr("agent1_scoping.estimator.calibrate_scope", fake)
    out = scope_project({"project_description": "Build a todo app", "team_id": "mesos"})
    assert out["scope"]["epics"][0]["stories"][0]["estimate_points"] == 42


def test_agent_keeps_gemini_estimates_if_calibration_fails(monkeypatch):
    def boom(*a, **k):
        raise KeyError("Unknown team")

    monkeypatch.setattr("agent1_scoping.estimator.calibrate_scope", boom)
    out = scope_project({"project_description": "Build a todo app", "team_id": "nope"})
    assert "errors" not in out and out["scope"]["epics"][0]["stories"][0]["estimate_points"] == 3
