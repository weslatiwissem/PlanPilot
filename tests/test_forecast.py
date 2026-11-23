import json
import logging
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent2_ops import agent as a2
from agent3_risk import check_serving, serve, train_final
from agent3_risk import sprint_data as sd
from agent3_risk.agent import assess_risk
from agent3_risk.synthetic import write_synthetic
from orchestrator.main import app

client = TestClient(app)
SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "d1", "estimate_points": 3},
        {"title": "Auth API", "description": "d2", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [{"title": "Task list", "description": "d3", "estimate_points": 2}]},
]}
PAST = [{"committed": 10, "delivered": 8}, {"committed": 12, "delivered": 6}]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("risk")
    write_synthetic(d / "data", boards=16, sprints_per_board=6, seed=5)
    train_final.main(["--data-dir", str(d / "data"), "--out", str(d / "models"), "--save"])
    return d / "models"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("SPRINT_CAPACITY", raising=False)
    monkeypatch.setenv("RISK_MODEL_DIR", str(tmp_path / "no_model_here"))  # default: no model
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: ("Alice", 0.9) for s in stories}, "stub"))
    serve.clear_cache()
    yield
    serve.clear_cache()


@pytest.fixture
def with_model(monkeypatch, trained):
    monkeypatch.setenv("RISK_MODEL_DIR", str(trained))
    serve.clear_cache()


def _state(**extra):
    return {"scope": SCOPE, **a2.plan_sprints({"scope": SCOPE}), **extra}


# ------------------------------------------------------------------ the forecast itself
def test_without_a_model_there_is_no_forecast_and_the_rules_still_run():
    out = assess_risk(_state())
    assert out["forecasts"] == [] and isinstance(out["risks"], list)


def test_one_forecast_per_sprint_without_history(with_model, monkeypatch):
    monkeypatch.setenv("SPRINT_CAPACITY", "6")  # three sprints
    out = assess_risk(_state())
    forecasts = out["forecasts"]
    assert [f["sprint"] for f in forecasts] == [1, 2, 3]
    for f in forecasts:
        assert f["model"] == "B1_plus" and f["history_used"] == 0
        assert 0 < f["probability"] < 1 and 0 < f["reference_rate"] < 1


def test_past_sprints_switch_to_the_history_model(with_model):
    f = assess_risk(_state(past_sprints=PAST))["forecasts"][0]
    assert f["model"] == "B2_plus" and f["history_used"] == 2
    many = [{"committed": 10, "delivered": 9}] * 5
    assert assess_risk(_state(past_sprints=many))["forecasts"][0]["history_used"] == 3  # the model reads the last 3


def test_a_changed_feature_layout_disables_the_forecast_instead_of_guessing(trained, monkeypatch, tmp_path, caplog):
    bad = tmp_path / "bad_model"
    shutil.copytree(trained, bad)
    cfg = json.loads((bad / "config.json").read_text())
    cfg["models"]["B1_plus"]["features"][:2] = cfg["models"]["B1_plus"]["features"][:2][::-1]
    (bad / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("RISK_MODEL_DIR", str(bad))
    serve.clear_cache()
    with caplog.at_level(logging.WARNING):
        assert assess_risk(_state())["forecasts"] == []
    assert "retrain" in caplog.text


def test_forecast_never_raises_on_a_broken_state(with_model):
    assert serve.forecast_sprints({}) == []
    assert serve.forecast_sprints({"sprint_plan": "nonsense"}) == []


# ------------------------------------------------------------------ what the server feeds the model
def test_sprint_days_team_size_and_cap_reach_the_features(with_model, monkeypatch):
    seen = []
    real = serve.build_features

    def spy(sprints, past, team, planday, no_admin, history):
        seen.append(dict(team=team, planday=planday, no_admin=no_admin))
        return real(sprints, past, team, planday, no_admin, history)

    monkeypatch.setattr(serve, "build_features", spy)
    cfg = serve.load_models()
    assess_risk(_state())
    assert seen[-1]["planday"] == cfg["impute"]["planday"] and seen[-1]["no_admin"] == cfg["impute"]["no_admin"]
    assert seen[-1]["team"] == 1  # the stubbed matcher assigns everything to Alice: one distinct assignee
    assess_risk(_state(sprint_days=99, team_members=[{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]))
    assert seen[-1]["planday"] == 60.0 and seen[-1]["team"] == 4  # capped like in training; team list wins


def test_features_use_the_same_code_as_training():
    sprints = [{"number": 1, "n": 3, "points": 15.0}]
    X1, names1, X2, names2 = serve.build_features(sprints, PAST, team=3, planday=14, no_admin=1, history=3)
    col = {n: X2[0, i] for i, n in enumerate(names2)}
    assert col["commit"] == 15 and col["snap"] == 0 and col["vel_done"] == 0 and col["vel_todo"] == 15
    assert col["commit_per_member"] == pytest.approx(5.0) and col["points_per_issue"] == pytest.approx(5.0)
    assert col["n_prev"] == 2
    assert col["prev1_commit"] == pytest.approx(np.log1p(12)) and col["prev1_done"] == pytest.approx(np.log1p(6))
    assert col["last_ratio"] == pytest.approx(0.5) and col["past_fail_rate"] == pytest.approx(1.0)
    assert col["overcommit"] == pytest.approx(np.log1p(15) - np.mean(np.log1p([8, 6])), rel=1e-5)
    assert names1 == sd.tabular_plus(__import__("pandas").DataFrame(
        [dict(commit=1, planday=1, no_issue_starttime=1, vel_starttime=1, no_issue_added=0, vel_added=0, no_issue_removed=0,
              vel_removed=0, no_issuetodo=1, vel_todo=1, no_issueinprogress=0, vel_inprogress=0, no_issuedone=0, vel_done=0,
              no_admin=1, no_teammember=1, snap=0)]))[1]
    X1_alone, _, no_hist, _ = serve.build_features(sprints, [], team=3, planday=14, no_admin=1, history=3)
    assert no_hist is None and X1_alone.shape == X1.shape


# ------------------------------------------------------------------ the API and the graph
def test_the_graph_accepts_the_forecast_key(with_model, monkeypatch):
    import orchestrator.graph as g

    monkeypatch.setattr(g, "scope_project", lambda state: {"scope": SCOPE})
    out = g.build_graph().invoke({"project_description": "x"})  # LangGraph fails if `forecasts` is not in the state
    assert out["forecasts"] and out["risks"] is not None


def _tickets():
    return [dict(title=t, epic=e, points=p, sprint=1, assignee="Alice", confidence=0.9, key="k")
            for t, e, p in (("Login UI", "Auth", 3), ("Auth API", "Auth", 5), ("Task list", "Tasks", 2))]


def test_steps_return_forecasts(with_model):
    r = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(), "past_sprints": PAST, "sprint_days": 14})
    assert r.status_code == 200 and r.json()["forecasts"][0]["model"] == "B2_plus"
    r = client.post("/steps/sprints", json={"scope": SCOPE})
    assert r.status_code == 200 and r.json()["forecasts"][0]["model"] == "B1_plus"


def test_steps_work_without_a_model():
    r = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets()})
    assert r.status_code == 200 and r.json()["forecasts"] == []


@pytest.mark.parametrize("bad", [{"past_sprints": [{"committed": -1, "delivered": 2}]}, {"sprint_days": 0},
                                 {"sprint_days": 400}, {"past_sprints": [{"committed": 5}]}])
def test_bad_forecast_inputs_are_rejected(bad):
    assert client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(), **bad}).status_code == 422


def test_openapi_has_the_forecast_contract():
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"Forecast", "PastSprint"} <= set(schemas)
    assert "forecasts" in schemas["RisksStepResponse"]["properties"]


# ------------------------------------------------------------------ mapping check
def test_snapshot0_report(tmp_path):
    write_synthetic(tmp_path, boards=6, sprints_per_board=4, seed=1)
    s = sd.samples(sd.load_all(str(tmp_path)))
    report = check_serving.snapshot0_report(s[s["snap"] == 0])
    assert len(report) == 6 and all(0.0 <= v <= 1.0 for v in report.values())
    assert report["nothing done yet (vel_done == 0)"] == 1.0
