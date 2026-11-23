import numpy as np

from agent3_risk.eval_snapshots import board_bootstrap, interval


def _data(seed=0, n_boards=30, per=20):
    rng = np.random.default_rng(seed)
    boards = np.repeat(np.arange(n_boards), per)
    y = (rng.random(len(boards)) < 0.6).astype(int)
    informative = y + rng.normal(0, 0.4, len(y))
    noise = rng.normal(0, 1, len(y))
    return y, boards, informative, noise


def test_informative_score_clearly_beats_noise():
    y, boards, good, noise = _data()
    boot = board_bootstrap(y, {"good": good, "noise": noise}, boards, 200)
    lo, _ = interval(boot["good"] - boot["noise"])
    assert lo > 0


def test_identical_scores_have_zero_difference():
    y, boards, good, _ = _data()
    boot = board_bootstrap(y, {"a": good, "b": good.copy()}, boards, 100)
    assert np.allclose(boot["a"], boot["b"])
    lo, hi = interval(boot["a"] - boot["b"])
    assert abs(lo) < 1e-12 and abs(hi) < 1e-12


def test_resamples_are_paired_and_reproducible():
    y, boards, good, noise = _data()
    a = board_bootstrap(y, {"g": good, "n": noise}, boards, 50, seed=3)
    b = board_bootstrap(y, {"g": good, "n": noise}, boards, 50, seed=3)
    assert len(a["g"]) == len(a["n"])
    assert np.array_equal(a["g"], b["g"]) and np.array_equal(a["n"], b["n"])


def test_imputed_copy_uses_reference_median_and_keeps_original():
    from agent3_risk.eval_snapshots import _imputed

    X = np.array([[1.0, 10.0, 5.0], [2.0, 20.0, 6.0]])
    ref = np.array([[0.0, 100.0, 0.0], [0.0, 300.0, 0.0], [0.0, 200.0, 0.0]])
    out = _imputed(X, [1], ref)
    assert out[:, 1].tolist() == [200.0, 200.0]  # median of the reference column
    assert out[:, [0, 2]].tolist() == X[:, [0, 2]].tolist()  # other columns untouched
    assert X[:, 1].tolist() == [10.0, 20.0]  # original not modified


def test_out_of_fold_plumbing_with_gru_stand_in(tmp_path, monkeypatch):
    """Every model gets a held-out prediction for every row. torch is replaced by a stand-in trainer."""
    from agent3_risk import eval_snapshots as ev
    from agent3_risk import sprint_data as sd
    from agent3_risk.synthetic import write_synthetic

    write_synthetic(tmp_path, boards=14, sprints_per_board=6, seed=3)
    df = sd.load_all(str(tmp_path))
    s = sd.samples(df)
    mats = {
        "B1_snapshot": sd.tabular(s)[0],
        "B2_snapshot_history": sd.tabular_history(df, s, 3)[0],
        "B1_plus": sd.tabular_plus(s)[0],
        "B2_plus": sd.tabular_history_plus(df, s, 3)[0],
    }
    S, M = sd.sequences(df, s, 3)
    seen = {}

    def fake_train(Ztr, ytr, Zva, yva, seed, epochs, hidden):
        seen["n_in"] = Ztr.shape[2]
        return object(), 0.0

    monkeypatch.setattr(ev, "train_gru", fake_train)
    monkeypatch.setattr(ev, "gru_predict", lambda model, Z: np.full(len(Z), 0.5))
    cols = [sd.RAW.index(c) for c in ev.UNKNOWN_AT_PLANNING]
    preds = ev.out_of_fold(s, mats, 3, 0, cols, gru={"S": S, "M": M, "epochs": 1, "hidden": 4})

    expected = set(mats) | {"B1_plus_imputed", "B2_plus_imputed", "GRU"}
    assert set(preds) == expected
    for name, p in preds.items():
        assert len(p) == len(s) and np.isfinite(p).all(), name
        assert (p > 0).all() and (p < 1).all(), name
    assert seen["n_in"] == len(sd.STEP_FEATURES) + 1  # step features + the mask channel


def test_planning_view_describes_a_freshly_planned_sprint():
    import pandas as pd

    from agent3_risk import eval_snapshots as ev

    s = pd.DataFrame([
        dict(commit=12.0, vel_starttime=10.0, vel_added=3.0, vel_removed=1.0, vel_done=4.0, vel_inprogress=2.0,
             vel_todo=6.0, no_issue_starttime=5, no_issue_added=1, no_issue_removed=0, no_issuedone=2,
             no_issueinprogress=1, no_issuetodo=3, snap=30),
        dict(commit=7.0, vel_starttime=0.0, vel_added=7.0, vel_removed=0.0, vel_done=0.0, vel_inprogress=0.0,
             vel_todo=7.0, no_issue_starttime=0, no_issue_added=2, no_issue_removed=0, no_issuedone=0,
             no_issueinprogress=0, no_issuetodo=2, snap=0),
    ])
    v = ev.planning_view(s)
    assert v["commit"].tolist() == [10.0, 7.0]  # the start-of-sprint commitment; an empty start keeps the real one
    assert v["vel_todo"].tolist() == v["commit"].tolist() and v["no_issuetodo"].tolist() == [5, 0]
    assert (v[list(ev.PLANNING_ZEROS)] == 0).all().all() and (v["snap"] == 0).all()
    assert s["commit"].tolist() == [12.0, 7.0]  # the input is not modified


def test_out_of_fold_scores_the_planning_view_with_models_trained_on_real_rows(tmp_path):
    from agent3_risk import eval_snapshots as ev
    from agent3_risk import sprint_data as sd
    from agent3_risk.synthetic import write_synthetic

    write_synthetic(tmp_path, boards=14, sprints_per_board=6, seed=4)
    df = sd.load_all(str(tmp_path))
    s = sd.samples(df)
    mats = {"B1_plus": sd.tabular_plus(s)[0], "B2_plus": sd.tabular_history_plus(df, s, 3)[0]}
    view = ev.planning_view(s)
    sim = {"B1_plus": sd.tabular_plus(view)[0], "B2_plus": sd.tabular_history_plus(df, view, 3)[0]}
    cols = [sd.RAW.index(c) for c in ev.UNKNOWN_AT_PLANNING]
    preds = ev.out_of_fold(s, mats, 3, 0, cols, sim=sim)
    assert set(preds) == {"B1_plus", "B2_plus", "B1_plus_imputed", "B2_plus_imputed", "B1_plus_served", "B2_plus_served"}
    for name, p in preds.items():
        assert len(p) == len(s) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), name
    assert not np.allclose(preds["B1_plus"], preds["B1_plus_served"])  # the rewrite really changes the input
