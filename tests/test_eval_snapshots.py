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
