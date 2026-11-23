"""The model PlanPilot serves: trained on what it will be fed, and constrained to behave sensibly.

Why: check_serving.py showed that real snapshot-0 rows are often NOT "nothing done, everything still to do", and the
*_served rows of eval_snapshots showed that a model trained on all snapshots loses a lot (and the no-history one
becomes useless) when it is fed such a clean row. So the served models are trained only on snapshot-0 sprints,
rewritten the way PlanPilot describes a planned sprint (`planning_view`).

Monotone constraints: with ~1,100 sprints the trees can learn odd shapes. A heavier sprint, a bigger commitment
compared with the team's own history, or a history of missed sprints must never LOWER the forecast, so the load and
history features are constrained (LightGBM `monotone_constraints`).
"""
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np

from agent3_risk import sprint_data as sd

PLANNING_ZEROS = ("vel_added", "vel_removed", "vel_done", "vel_inprogress", "no_issue_added", "no_issue_removed",
                  "no_issuedone", "no_issueinprogress")
# Smaller trees than train_final.LGBM_PARAMS: about 1,100 training sprints instead of 4,900. Fixed in advance.
PLAN_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, verbose=-1)
# +1: a larger value must not lower the forecast. -1: a larger value must not raise it.
MONOTONE = {
    "commit": 1, "vel_starttime": 1, "vel_todo": 1, "commit_per_member": 1,      # load
    "no_issue_starttime": 1, "no_issuetodo": 1,                                   # number of stories
    "overcommit": 1, "commit_vs_last": 1, "past_fail_rate": 1,                    # against the team's own history
    "avg_past_ratio": -1, "last_ratio": -1, "prev1_ratio": -1, "prev2_ratio": -1, "prev3_ratio": -1,
}


def planning_view(s):
    """The same sprints as PlanPilot would describe them at planning time (what the server feeds the model).

    Nothing done or in progress, no scope change, everything still to do. A sprint whose start-of-sprint commitment
    is 0 (everything was added later) keeps its real commitment, because PlanPilot never plans an empty sprint.
    """
    v = s.copy()
    v["commit"] = v["vel_starttime"].where(v["vel_starttime"] > 0, v["commit"])
    for col in PLANNING_ZEROS:
        v[col] = 0.0
    v["vel_todo"] = v["commit"]
    v["no_issuetodo"] = v["no_issue_starttime"]
    v["snap"] = 0
    return v


def monotone_vector(names: List[str]) -> List[int]:
    return [MONOTONE.get(n, 0) for n in names]


def plan_matrices(df, s, history: int) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Feature matrices of the planning view of every row of s (callers keep the snapshot-0 rows)."""
    view = planning_view(s)
    return {"B1_plan": sd.tabular_plus(view), "B2_plan": sd.tabular_history_plus(df, view, history)}


def fit_plan(Xtr, ytr, Xva, yva, names: List[str], seed: int) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(random_state=seed, monotone_constraints=monotone_vector(names), **PLAN_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    return m
