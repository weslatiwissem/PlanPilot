import numpy as np
import pytest

from agent3_risk import sprint_data as sd
from agent3_risk.synthetic import FRACTION, HEADER

# one board, 4 sprints in time order: (sprintid, committed points, points really delivered)
SPRINTS = [(101, 10.0, 12.0), (102, 10.0, 5.0), (103, 20.0, 10.0), (104, 15.0, 18.0)]


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    d = tmp_path_factory.mktemp("mini")
    for snap, frac in FRACTION.items():
        lines = []
        for sid, commit, final in SPRINTS:
            done = round(final * frac, 1)
            todo = round(max(commit - done, 0.0), 1)
            values = [14, 5, commit, 0, 0.0, 0, 0.0, 3, todo, 0, 0.0, 1, done, 1, 3, final - commit]
            lines.append(",".join(["1", str(sid), "s"] + [str(v) for v in values]))
        (d / f"jira_iteration_{snap}.csv").write_text(",".join(HEADER) + "\n" + "\n".join(lines) + "\n", encoding="latin-1")
    df = sd.load_all(str(d))
    return df, sd.samples(df)


def _row(s, sprint_id, snap):
    return s[(s.sprintid == sprint_id) & (s.snap == snap)].index[0]


def test_current_snapshot_ratios(mini):
    df, s = mini
    i = _row(s, 104, 30)  # commit 15, done 5.4 at 30 % of the sprint
    f = dict(zip(sd.EXTRA_CURRENT, sd.engineered_current(s)[i]))
    assert f["done_share"] == pytest.approx(5.4 / 15, rel=1e-4)
    assert f["burn_gap"] == pytest.approx(0.3 - 5.4 / 15, rel=1e-3)  # ahead of schedule: negative
    assert f["todo_share"] == pytest.approx((15 - 5.4) / 15, rel=1e-4)
    assert f["added_share"] == 0.0
    assert f["points_per_issue"] == pytest.approx(3.0)
    assert f["commit_per_member"] == pytest.approx(5.0)


def test_history_ratios_use_only_previous_sprints(mini):
    df, s = mini
    i = _row(s, 104, 0)  # previous sprints delivered 12/10, 5/10 and 10/20
    h = dict(zip(sd.EXTRA_HISTORY, sd.engineered_history(df, s, 3)[i]))
    assert h["avg_past_ratio"] == pytest.approx((1.2 + 0.5 + 0.5) / 3, rel=1e-4)
    assert h["last_ratio"] == pytest.approx(0.5)
    assert h["past_fail_rate"] == pytest.approx(2 / 3, rel=1e-4)
    mean_done = np.mean(np.log1p([12.0, 5.0, 10.0]))
    assert h["overcommit"] == pytest.approx(np.log1p(15.0) - mean_done, rel=1e-4)
    assert h["commit_vs_last"] == pytest.approx(np.log1p(15.0) - np.log1p(20.0), rel=1e-4)


def test_first_sprint_has_nan_history_not_zero(mini):
    df, s = mini
    first = sd.engineered_history(df, s, 3)[_row(s, 101, 0)]
    assert np.isnan(first).all()


def test_progress_is_a_step_feature(mini):
    df, s = mini
    seq, mask = sd.sequences(df, s, 3)
    i, p = _row(s, 104, 30), sd.STEP_FEATURES.index("progress")
    assert seq[i, 3, p] == pytest.approx(0.3)  # current sprint: share elapsed
    assert (seq[i, :3, p] == 1.0).all() and mask[i, :3].all()  # finished sprints
    assert seq[i, 3, sd.STEP_FEATURES.index("is_current")] == 1.0


def test_label_never_leaks_into_features(mini):
    df, s = mini
    for X in (sd.tabular_plus(s)[0], sd.tabular_history_plus(df, s, 3)[0]):
        assert not np.isin(s["vel_diff"].to_numpy(), np.nan_to_num(X[:, :1])).all()
    assert "vel_diff" not in sd.tabular_plus(s)[1]
