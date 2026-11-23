import json

import numpy as np
import pytest

from agent3_risk import plan_model, probe_forecast, serve, train_plan
from agent3_risk import eval_snapshots as ev
from agent3_risk import sprint_data as sd
from agent3_risk.agent import assess_risk
from agent3_risk.synthetic import write_synthetic
from orchestrator.schemas import Scope

SCOPE = {"epics": [{"title": "E", "stories": [
    {"title": "a", "description": "", "estimate_points": 5}, {"title": "b", "description": "", "estimate_points": 3}]}]}
PAST = [{"committed": 10, "delivered": 8}, {"committed": 12, "delivered": 11}]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("plan")
    write_synthetic(d / "data", boards=40, sprints_per_board=8, seed=7)
    train_plan.main(["--data-dir", str(d / "data"), "--out", str(d / "models"), "--save"])
    return d


def _use(monkeypatch, path):
    monkeypatch.setenv("RISK_MODEL_DIR", str(path))
    serve.clear_cache()


def test_monotone_vector_constrains_load_and_history_only():
    names = ["planday", "commit", "overcommit", "avg_past_ratio", "prev1_done", "points_per_issue"]
    assert plan_model.monotone_vector(names) == [0, 1, 1, -1, 0, 0]


def test_saved_config_describes_what_is_served(trained):
    cfg = json.loads((trained / "models" / "config.json").read_text())
    assert cfg["kind"] == "planning" and cfg["serve"] == {"history": "B2_plan", "no_history": "B1_plan"}
    assert set(cfg["models"]) == {"B1_plan", "B2_plan"} and 0 < cfg["base_rate"] < 1
    for spec in cfg["models"].values():
        assert len(spec["monotone"]) == len(spec["features"]) and any(spec["monotone"])


def test_the_forecast_never_falls_when_the_load_grows(trained, monkeypatch):
    _use(monkeypatch, trained / "models")
    table = probe_forecast.probe()
    for label, values in table.items():
        assert values is not None and all(0 < v < 1 for v in values), label
        assert probe_forecast.falls(values) == 0, (label, values)  # the monotone constraints guarantee this
    serve.clear_cache()


def test_a_worse_history_never_lowers_the_forecast(trained, monkeypatch):
    _use(monkeypatch, trained / "models")
    cfg = serve.load_models()
    sprints = [{"number": 1, "n": 3, "points": 12.0}]
    probs = []
    for delivered in (12, 9, 6, 3):  # same commitments, ever fewer points delivered
        past = [{"committed": 12, "delivered": delivered}] * 3
        _, _, X2, _ = serve.build_features(sprints, past, 3, cfg["impute"]["planday"], cfg["impute"]["no_admin"], 3)
        probs.append(float(cfg["boosters"]["B2_plan"].predict(X2)[0]))
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])), probs
    serve.clear_cache()


def test_no_b1_serves_nothing_without_past_sprints(tmp_path, monkeypatch):
    write_synthetic(tmp_path / "data", boards=30, sprints_per_board=8, seed=8)
    train_plan.main(["--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "m"), "--save", "--no-b1"])
    _use(monkeypatch, tmp_path / "m")
    state = {"scope": SCOPE, "sprint_plan": {"sprints": [{"number": 1, "items": ["a", "b"], "capacity_points": 28}]},
             "ops_result": {"tickets": [dict(key="k", title=t, epic="E", points=p, sprint=1, assignee="Alice")
                                        for t, p in (("a", 5), ("b", 3))]}}
    assert assess_risk(state)["forecasts"] == []
    with_history = assess_risk({**state, "past_sprints": PAST})["forecasts"]
    assert len(with_history) == 1 and with_history[0]["model"] == "B2_plan"
    serve.clear_cache()


def test_plan_oof_only_scores_snapshot_zero(tmp_path):
    write_synthetic(tmp_path, boards=20, sprints_per_board=6, seed=9)
    df = sd.load_all(str(tmp_path))
    s = sd.samples(df)
    out = ev.plan_oof(s, df, 3, 0, 3, sd.RAW.index("planday"), sd.RAW.index("no_admin"))
    snap0 = (s["snap"] == 0).to_numpy()
    assert set(out) == {"B1_plan", "B2_plan"}
    for name, p in out.items():
        assert np.isfinite(p[snap0]).all() and ((p[snap0] > 0) & (p[snap0] < 1)).all(), name
        assert np.isnan(p[~snap0]).all(), name


def test_planning_view_is_shared_between_training_and_evaluation():
    assert ev.planning_view is plan_model.planning_view
    assert Scope.model_validate(SCOPE)  # fixture sanity
