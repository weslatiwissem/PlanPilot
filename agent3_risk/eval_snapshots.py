"""M5 follow-up: how good is sprint-risk prediction at each point of the sprint, and against simple rules?

    python -m agent3_risk.eval_snapshots

train_risk.py pools the four snapshots (0/30/50/80 % of the sprint). Late snapshots are easy (the work
already done reveals the outcome), so the pooled PR-AUC can hide how weak the model is at 0 %, which is the
moment PlanPilot plans a sprint. This script reports every snapshot separately, next to two rules:

  overcommit  log(commit) - mean log(done) of the board's previous sprints   (planning-time rule)
  burn_rate   share of the sprint elapsed minus share of the commitment done (progress rule; 0 at snapshot 0)

B1 / B2 are trained exactly like in train_risk.py (same folds, same early stopping); the predictions of
the held-out boards are collected and scored per snapshot.

Uncertainty: whole boards are resampled (sprints of one board are correlated). Every model is scored on the
SAME resamples, so the "paired differences" are fair: they ask "is A better than B on the boards we have?",
which is a sharper question than "do the two intervals overlap?". Percentile intervals from about 60 boards
are approximate, and 4 snapshots x 4 comparisons = 16 intervals, so expect an occasional "clear" by chance.
"""
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from agent3_risk import sprint_data as sd
from agent3_risk.train_risk import DATA_DIR, _val_boards, lgbm_predict

SHORT = {"B1_snapshot": "B1", "B2_snapshot_history": "B2", "overcommit": "overcommit", "burn_rate": "burn_rate"}
COMPARISONS = [
    ("B2_snapshot_history", "B1_snapshot"),
    ("B1_snapshot", "overcommit"),
    ("B2_snapshot_history", "overcommit"),
    ("B1_snapshot", "burn_rate"),
]


def out_of_fold(s, X1, X2, folds: int, seed: int):
    y, boards = s["y"].to_numpy(), s["boardid"].to_numpy()
    board_fold = sd.board_folds(boards, folds, seed)
    fold = np.array([board_fold[int(b)] for b in boards])
    p1, p2 = np.zeros(len(y)), np.zeros(len(y))
    for k in range(folds):
        te = fold == k
        tr = ~te
        va = tr & np.isin(boards, list(_val_boards(boards[tr], seed + k)))
        tr2 = tr & ~va
        p1[te] = lgbm_predict(X1[tr2], y[tr2], X1[va], y[va], X1[te], seed)
        p2[te] = lgbm_predict(X2[tr2], y[tr2], X2[va], y[va], X2[te], seed)
    return p1, p2


def rule_scores(df, s, history: int):
    """Two rules that need no training. Higher score = riskier."""
    seq, mask = sd.sequences(df, s, history)
    n_prev = mask[:, :history].sum(axis=1)
    past_done = seq[:, :history, sd.STEP_FEATURES.index("done")]  # log1p(done), 0 where padded
    avg_past = past_done.sum(axis=1) / np.maximum(n_prev, 1)
    overcommit = np.where(n_prev > 0, np.log1p(s["commit"].to_numpy(float)) - avg_past, 0.0)  # no history: neutral
    progress = s["snap"].to_numpy(float) / 100.0
    done_share = s["vel_done"].to_numpy(float) / s["commit"].to_numpy(float)
    burn_rate = np.clip(progress - done_share, 0.0, None)
    return {"overcommit": overcommit, "burn_rate": burn_rate}, n_prev


def board_bootstrap(y: np.ndarray, scores: Dict[str, np.ndarray], boards: np.ndarray,
                    n_boot: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """PR-AUC of every model on the same board-level resamples. Returns {model: array(n_resamples)}."""
    rng = np.random.default_rng(seed)
    ids = np.unique(boards)
    rows = {b: np.where(boards == b)[0] for b in ids}
    out = {name: [] for name in scores}
    for _ in range(n_boot):
        idx = np.concatenate([rows[b] for b in rng.choice(ids, len(ids), replace=True)])
        if y[idx].min() == y[idx].max():  # one class only: PR-AUC undefined, skip for every model
            continue
        for name, sc in scores.items():
            out[name].append(average_precision_score(y[idx], sc[idx]))
    return {name: np.array(v) for name, v in out.items()}


def interval(values: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)

    df = sd.load_all(args.data_dir)
    s = sd.samples(df)
    y, boards, snap = s["y"].to_numpy(), s["boardid"].to_numpy(), s["snap"].to_numpy()
    X1, _ = sd.tabular(s)
    X2, _ = sd.tabular_history(df, s, args.history)

    varies = float((df.groupby(["boardid", "sprintid"])["planday"].nunique() > 1).mean())
    rules, n_prev = rule_scores(df, s, args.history)
    print(f"{len(s)} samples, {len(set(boards))} boards")
    print(f"planday differs between the 4 snapshots of the same sprint in {varies:.1%} of sprints "
          f"(~0% = it is the sprint length, NOT how far into the sprint we are)")
    print(f"samples without any previous sprint on their board: {(n_prev == 0).mean():.1%}")

    p1, p2 = out_of_fold(s, X1, X2, args.folds, args.seed)
    scores = {**rules, "B1_snapshot": p1, "B2_snapshot_history": p2}

    for t in sd.SNAPSHOTS:
        m = snap == t
        sub = {name: sc[m] for name, sc in scores.items()}
        boot = board_bootstrap(y[m], sub, boards[m], args.boot, args.seed)
        print(f"\n=== snapshot {t}%: {m.sum()} sprints, under-delivery rate {y[m].mean():.1%} (PR-AUC of a coin flip)")
        print(f"{'model':<22}{'PR-AUC':>8}   95% interval (boards)   {'ROC-AUC':>8}")
        full = {}
        for name, sc in sub.items():
            full[name] = average_precision_score(y[m], sc)
            lo, hi = interval(boot[name])
            print(f"{name:<22}{full[name]:>8.3f}   [{lo:.3f}, {hi:.3f}]{'':<10}{roc_auc_score(y[m], sc):>8.3f}")

        print("paired differences in PR-AUC (first minus second, same board resamples)")
        for a, b in COMPARISONS:
            if t == 0 and "burn_rate" in (a, b):
                continue  # nothing is done at 0 %, so burn_rate is constant there
            d = boot[a] - boot[b]
            lo, hi = interval(d)
            verdict = "clear" if lo > 0 or hi < 0 else "not clear"
            print(f"  {SHORT[a]} - {SHORT[b]:<11}{full[a] - full[b]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"first better in {np.mean(d > 0):>4.0%} of resamples   {verdict}")


if __name__ == "__main__":
    main()
