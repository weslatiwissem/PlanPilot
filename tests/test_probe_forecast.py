import pytest

from agent3_risk import probe_forecast, serve, train_final
from agent3_risk.synthetic import write_synthetic


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("probe")
    write_synthetic(d / "data", boards=16, sprints_per_board=6, seed=6)
    train_final.main(["--data-dir", str(d / "data"), "--out", str(d / "models"), "--save"])
    return d / "models"


def test_probe_returns_a_forecast_for_every_load_and_history(trained, monkeypatch):
    monkeypatch.setenv("RISK_MODEL_DIR", str(trained))
    serve.clear_cache()
    table = probe_forecast.probe()
    assert set(table) == set(probe_forecast.HISTORIES)
    for values in table.values():
        assert len(values) == len(probe_forecast.LOADS) and all(0 < v < 1 for v in values)
    serve.clear_cache()


def test_falls_counts_only_real_drops():
    assert probe_forecast.falls([0.2, 0.3, 0.5]) == 0
    assert probe_forecast.falls([0.2, 0.5, 0.3, 0.6, 0.55]) == 2
    assert probe_forecast.falls([0.50, 0.498]) == 0  # within tolerance


def test_probe_without_a_model_says_how_to_train_one(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_MODEL_DIR", str(tmp_path / "none"))
    serve.clear_cache()
    with pytest.raises(SystemExit, match="train_final"):
        probe_forecast.probe()
    serve.clear_cache()
