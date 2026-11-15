"""Data layer for M5 (sprint under-delivery risk). Pure numpy/pandas: no torch needed.

Source: TSE2017 "delivery capability" Jira data, data/raw/tse2017_jira/jira_iteration_{0,30,50,80}.csv.
Each file is the SAME 1,873 sprints observed at 0 / 30 / 50 / 80 % of the way through the sprint.

Facts verified on the real data (100% of rows):
    commitment_t = vel_starttime + vel_added - vel_removed            (what is promised at snapshot t)
    vel_diff_t   = final_done - commitment_t                          (final_done never changes across snapshots)
So `vel_diff` is the LABEL (it contains the final outcome) and must never be used as a feature.

Task: at snapshot t, will the sprint UNDER-deliver?   y = 1  <=>  final_done < commitment_t.
Sprints whose commitment is 0 at snapshot t are dropped (nothing was promised, so nothing can be missed).

Two views of the same samples, so the sequence model can be compared fairly with tabular models:
    tabular(df)            current-snapshot features only                 (proposal's "single-sprint snapshot" baseline)
    tabular_history(...)   current snapshot + last H sprints flattened    (tabular model that ALSO sees history)
    sequences(...)         (n, H+1, F) array + mask: last H finished sprints of the same board, then the
                           current sprint at snapshot t                    (input for the GRU)
"""
import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SNAPSHOTS = (0, 30, 50, 80)
RAW = ["planday", "no_issue_starttime", "vel_starttime", "no_issue_added", "vel_added", "no_issue_removed",
       "vel_removed", "no_issuetodo", "vel_todo", "no_issueinprogress", "vel_inprogress", "no_issuedone",
       "vel_done", "no_admin", "no_teammember"]
STEP_FEATURES = ["commit", "done", "todo", "inprog", "added", "removed", "planday", "team", "admin", "ratio", "is_current"]
N_STEP = len(STEP_FEATURES)
PLANDAY_CAP = 60  # planday has outliers up to 5873 days


def _l(x) -> float:
    """log1p of a count. A few rows have small negative values (vel_todo down to -7): clip at 0 so log1p stays finite."""
    return float(np.log1p(max(float(x), 0.0)))


def load_iteration_csv(path: str) -> pd.DataFrame:
    """Read one iteration CSV. The file is latin-1 and some sprint names contain unquoted commas
    (e.g. 'atlassian-selenium, anyone?'), which shifts the columns. The last 16 fields are always numeric, so
    everything between the 2nd field and those 16 is the name."""
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))
    head, k = rows[0], len(rows[0]) - 3
    out = [r[:2] + [",".join(r[2 : len(r) - k])] + r[len(r) - k :] for r in rows[1:] if len(r) >= len(head)]
    d = pd.DataFrame(out, columns=head)
    for c in head:
        if c != "name":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna().reset_index(drop=True)


def load_all(data_dir: str) -> pd.DataFrame:
    """One row per (sprint, snapshot) with commitment, final_done and the label."""
    frames = []
    for s in SNAPSHOTS:
        d = load_iteration_csv(f"{data_dir}/jira_iteration_{s}.csv")
        d["snap"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["commit"] = df.vel_starttime + df.vel_added - df.vel_removed
    df["final_done"] = df.vel_diff + df.commit
    df["y"] = (df.vel_diff < 0).astype(int)
    df = df.sort_values(["boardid", "sprintid", "snap"]).reset_index(drop=True)
    # position of each sprint inside its board (sprint ids grow with time: verified on the data)
    order = df.drop_duplicates(["boardid", "sprintid"])[["boardid", "sprintid"]].copy()
    order["pos"] = order.groupby("boardid").cumcount()
    return df.merge(order, on=["boardid", "sprintid"])


def samples(df: pd.DataFrame) -> pd.DataFrame:
    """Rows we can learn from: something was committed at that snapshot."""
    return df[df["commit"] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- tabular views
def tabular(s: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Current-snapshot features only. vel_diff is excluded on purpose (it is the label)."""
    X = s[RAW].copy()
    X["planday"] = X["planday"].clip(upper=PLANDAY_CAP)
    X["commit"], X["snap"] = s["commit"], s["snap"]
    return X.to_numpy(dtype=np.float32), list(X.columns)


def _finished(df: pd.DataFrame) -> Dict[Tuple[int, int], dict]:
    """What is known about a sprint once it is over: its 80% view with the real final_done."""
    last = df[df["snap"] == 80]
    return {(r.boardid, r.pos): {"commit": r.commit, "done": r.final_done, "team": r.no_teammember, "admin": r.no_admin,
                                 "issues": r.no_issue_starttime + r.no_issue_added - r.no_issue_removed}
            for r in last.itertuples()}


def _step_current(r) -> np.ndarray:
    ratio = r.vel_done / max(r.commit, 1.0)
    return np.array([_l(r.commit), _l(r.vel_done), _l(r.vel_todo), _l(r.vel_inprogress),
                     _l(r.vel_added), _l(r.vel_removed), _l(min(r.planday, PLANDAY_CAP)),
                     r.no_teammember, r.no_admin, min(max(ratio, 0.0), 3.0), 1.0], dtype=np.float32)


def _step_past(p: dict) -> np.ndarray:
    ratio = p["done"] / max(p["commit"], 1.0)
    return np.array([_l(p["commit"]), _l(p["done"]), 0.0, 0.0, 0.0, 0.0, 0.0,
                     p["team"], p["admin"], min(max(ratio, 0.0), 3.0), 0.0], dtype=np.float32)


def sequences(df: pd.DataFrame, s: pd.DataFrame, history: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """(n, history+1, N_STEP) and a (n, history+1) mask. Oldest finished sprint first, current sprint last.
    Boards with fewer than `history` earlier sprints are left-padded with zeros (mask = 0)."""
    fin = _finished(df)
    X = np.zeros((len(s), history + 1, N_STEP), dtype=np.float32)
    M = np.zeros((len(s), history + 1), dtype=np.float32)
    for i, r in enumerate(s.itertuples()):
        for k in range(1, history + 1):  # k sprints back
            p = fin.get((r.boardid, r.pos - k))
            if p is not None:
                X[i, history - k] = _step_past(p)
                M[i, history - k] = 1.0
        X[i, history] = _step_current(r)
        M[i, history] = 1.0
    assert np.isfinite(X).all(), "non-finite value in sequences (check the raw columns)"
    return X, M


def tabular_history(df: pd.DataFrame, s: pd.DataFrame, history: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Current snapshot + the previous `history` sprints flattened (a tabular model that also sees history)."""
    base, names = tabular(s)
    seq, mask = sequences(df, s, history)
    past = seq[:, :history, :]  # drop the current step (already in `base`)
    keep = [STEP_FEATURES.index(c) for c in ("commit", "done", "ratio")]
    flat = past[:, :, keep].reshape(len(s), -1)
    names = names + [f"prev{history - j}_{STEP_FEATURES[c]}" for j in range(history) for c in keep]
    return np.hstack([base, flat, mask[:, :history].sum(axis=1, keepdims=True)]).astype(np.float32), names + ["n_prev"]


def board_folds(boards: np.ndarray, n_folds: int = 5, seed: int = 42) -> Dict[int, int]:
    """Assign whole boards to folds: no board is ever in both train and test (split hygiene)."""
    ids = np.array(sorted(set(boards)))
    np.random.default_rng(seed).shuffle(ids)
    return {int(b): i % n_folds for i, b in enumerate(ids)}
