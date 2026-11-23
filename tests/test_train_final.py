import json

import lightgbm as lgb
import numpy as np

from agent3_risk import train_final
from agent3_risk.synthetic import write_synthetic
from agent3_risk.train_risk import lgbm_predict


def test_saved_models_match_their_config(tmp_path):
    write_synthetic(tmp_path / "data", boards=16, sprints_per_board=6, seed=2)
    out = tmp_path / "models"
    train_final.main(["--data-dir", str(tmp_path / "data"), "--out", str(out), "--save"])

    cfg = json.loads((out / "config.json").read_text())
    assert set(cfg["models"]) == {"B1_plus", "B2_plus"}
    assert set(cfg["impute"]) == {"planday", "no_admin"}
    assert 0 < cfg["base_rate"] < 1
    assert all(np.isfinite(v) for v in cfg["impute"].values())
    rng = np.random.default_rng(0)
    for name, spec in cfg["models"].items():
        booster = lgb.Booster(model_file=str(out / spec["file"]))
        assert booster.num_feature() == len(spec["features"]), name
        p = booster.predict(rng.random((5, len(spec["features"]))))
        assert ((p > 0) & (p < 1)).all(), name


def test_final_model_uses_the_same_settings_as_the_evaluated_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
    Xtr, ytr, Xva, yva, Xte = X[:200], y[:200], X[200:250], y[200:250], X[250:]
    expected = lgbm_predict(Xtr, ytr, Xva, yva, Xte, 7)
    got = train_final.fit_final(Xtr, ytr, Xva, yva, 7).predict_proba(Xte)[:, 1]
    assert np.allclose(expected, got)
