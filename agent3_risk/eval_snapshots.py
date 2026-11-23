"""M5 follow-up: how good is sprint-risk prediction at each point of the sprint, and against simple rules?

    python -m agent3_risk.eval_snapshots              # LightGBM models + rules (no torch needed)
    python -m agent3_risk.eval_snapshots --gru        # also the GRU (needs torch; retrained with `progress`)

train_risk.py pools the four snapshots (0/30/50/80 % of the sprint). Late snapshots are easy (the work
already done reveals the outcome), so the pooled PR-AUC can hide how weak a model is at 0 %, which is the
moment PlanPilot plans a sprint. This script reports every snapshot separately.

Models (same board-level folds and early stopping as train_risk.py):
  overcommit / burn_rate   two rules that need no training (see rule_scores)
  B1_snapshot, B2_snapshot_history   LightGBM on the raw snapshot, without / with the previous sprints
  B1_plus, B2_plus                   the same plus engineered ratios (sprint_data.tabular_plus ...)
  GRU                                sequence model, now with the `progress` input (--gru)

Serving check: PlanPilot cannot supply `planday` (sprint length) or `no_admin` at planning time. The *_imputed
variants of B1_plus / B2_plus are trained on the real values and served with both replaced by the training
median, which is what the live pipeline would have to do.

Served simulation (*_served, snapshot 0 only): check_serving.py showed that real snapshot-0 rows are often NOT
"nothing done, everything still to do". The server sends exactly such a clean row. These variants are trained on
the real rows and then scored on the SAME sprints rewritten the way PlanPilot would describe them (planning_view:
nothing done or in progress, no scope change, everything to do, plus the two unknown inputs imputed). This is the
number that tells whether the model survives the move into the product.

Planning models (B1_plan / B2_plan, snapshot 0 only): trained on the planning view of snapshot-0 sprints only, with
monotone constraints (plan_model.py), the models PlanPilot serves. Their rows are directly comparable with the
*_served rows: both are scored on PlanPilot-style input.

Uncertainty: whole boards are resampled (sprints of one board are correlated) and every model is scored on
the SAME resamples, so "paired differences" ask "is A better than B on the boards we have?". Only the three
comparisons tagged (primary) were fixed in advance; the others are exploratory. With ~60 boards the intervals
are approximate, and with this many comparisons an occasional "clear" can be chance.

Brier score (lower is better) checks that the probabilities themselves are trustworthy, which matters
because PlanPilot will show them to users. A coin flip scores p * (1 - p) for a base rate p.
"""
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from agent3_risk import sprint_data as sd
from agent3_risk.plan_model import PLANNING_ZEROS, fit_plan, plan_matrices, planning_view  # noqa: F401
from agent3_risk.train_risk import DATA_DIR, _val_boards, gru_predict, lgbm_predict, normalise_stats, prepare, train_gru

RULES = ("overcommit", "burn_rate")
SERVED = ("B1_plus", "B2_plus")  # the models PlanPilot would serve
UNKNOWN_AT_PLANNING = ("planday", "no_admin")  # inputs PlanPilot cannot provide
PRIMARY = {("B1_plus", "B1_snapshot"), ("B2_plus", "B2_snapshot_history"), ("B2_plus", "overcommit")}
COMPARISONS = [
    ("B1_plus", "B1_snapshot"),
    ("B2_plus", "B2_snapshot_history"),
    ("B2_plus", "overcommit"),
    ("B2_plus", "B1_plus"),
    ("B2_snapshot_history", "B1_snapshot"),
    ("B1_plus", "overcommit"),
    ("B1_plus", "burn_rate"),
    ("GRU", "B1_snapshot"),
    ("GRU", "B2_snapshot_history"),
    ("GRU", "B2_plus"),
    ("B1_plus_imputed", "B1_plus"),
    ("B2_plus_imputed", "B2_plus"),
    ("B1_plus_served", "B1_plus"),
    ("B2_plus_served", "B2_plus"),
    ("B2_plus_served", "overcommit"),
    ("B1_plus_served", "overcommit"),
    ("B2_plan", "overcommit"),
    ("B1_plan", "overcommit"),
    ("B2_plan", "B2_plus_served"),
    ("B1_plan", "B1_plus_served"),
    ("B2_plan", "B2_plus"),
]



def _imputed(X: np.ndarray, cols, reference: np.ndarray) -> np.ndarray:
    """Copy of X where `cols` hold the median of `reference` (what a server without those inputs would use)."""
    Xi = X.copy()
    Xi[:, cols] = np.median(reference[:, cols], axis=0)
    return Xi


def out_of_fold(s, mats: Dict[str, np.ndarray], folds: int, seed: int, cols=(), served=SERVED,
                gru: Optional[dict] = None, sim: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Held-out predictions of every model (boards never in both train and test).

    `sim` maps a served model to the feature matrix of the planning_view rows: those rows are scored by the model
    trained on the REAL rows and stored as <name>_served.
    """
    y, boards = s["y"].to_numpy(), s["boardid"].to_numpy()
    board_fold = sd.board_folds(boards, folds, seed)
    fold = np.array([board_fold[int(b)] for b in boards])
    names = (list(mats) + [f"{n}_imputed" for n in served if n in mats]
             + [f"{n}_served" for n in served if sim and n in sim] + (["GRU"] if gru else []))
    preds = {n: np.zeros(len(y)) for n in names}
    for k in range(folds):
        te = fold == k
        tr = ~te
        va = tr & np.isin(boards, list(_val_boards(boards[tr], seed + k)))
        tr2 = tr & ~va
        n_te = int(te.sum())
        for name, X in mats.items():
            if name in served:
                tests = [X[te], _imputed(X[te], cols, X[tr2])]  # real rows first, then imputed copies
                if sim and name in sim:
                    tests.append(_imputed(sim[name][te], cols, X[tr2]))  # then the planning-time rewrite
                both = lgbm_predict(X[tr2], y[tr2], X[va], y[va], np.vstack(tests), seed)
                preds[name][te], preds[f"{name}_imputed"][te] = both[:n_te], both[n_te:2 * n_te]
                if len(tests) == 3:
                    preds[f"{name}_served"][te] = both[2 * n_te:]
            else:
                preds[name][te] = lgbm_predict(X[tr2], y[tr2], X[va], y[va], X[te], seed)
        if gru:
            mu, sig = normalise_stats(gru["S"][tr2], gru["M"][tr2])
            Z = prepare(gru["S"], gru["M"], mu, sig)
            model, _ = train_gru(Z[tr2], y[tr2], Z[va], y[va], seed, gru["epochs"], gru["hidden"])
            preds["GRU"][te] = gru_predict(model, Z[te])
    return preds


def plan_oof(s, df, folds: int, seed: int, history: int, planday_col: int, admin_col: int) -> Dict[str, np.ndarray]:
    """B1_plan / B2_plan: trained on snapshot-0 planning rows only, scored on held-out boards' snapshot-0 rows.

    Training rows get the median `no_admin` (unknown at planning time); test rows also the median `planday`,
    the worst case at serving. Rows of other snapshots stay NaN: these models are only defined at planning time.
    """
    y, boards, snap = s["y"].to_numpy(), s["boardid"].to_numpy(), s["snap"].to_numpy()
    mats = plan_matrices(df, s, history)
    board_fold = sd.board_folds(boards, folds, seed)
    fold = np.array([board_fold[int(b)] for b in boards])
    out = {name: np.full(len(y), np.nan) for name in mats}
    for k in range(folds):
        te = (fold == k) & (snap == 0)
        tr = (fold != k) & (snap == 0)
        va = tr & np.isin(boards, list(_val_boards(boards[tr], seed + k)))
        tr2 = tr & ~va
        for name, (X, names) in mats.items():
            ref = X[tr2]
            model = fit_plan(_imputed(X[tr2], [admin_col], ref), y[tr2], _imputed(X[va], [admin_col], ref), y[va], names, seed)
            out[name][te] = model.predict_proba(_imputed(X[te], [planday_col, admin_col], ref))[:, 1]
    return out


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
    ap.add_argument("--gru", action="store_true", help="also train the GRU in every fold (needs torch)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args(argv)

    df = sd.load_all(args.data_dir)
    s = sd.samples(df)
    y, boards, snap = s["y"].to_numpy(), s["boardid"].to_numpy(), s["snap"].to_numpy()
    X1, _ = sd.tabular(s)
    X2, _ = sd.tabular_history(df, s, args.history)
    X1p, _ = sd.tabular_plus(s)
    X2p, _ = sd.tabular_history_plus(df, s, args.history)
    mats = {"B1_snapshot": X1, "B2_snapshot_history": X2, "B1_plus": X1p, "B2_plus": X2p}
    view = planning_view(s)
    sim = {"B1_plus": sd.tabular_plus(view)[0], "B2_plus": sd.tabular_history_plus(df, view, args.history)[0]}
    gru = None
    if args.gru:
        S, M = sd.sequences(df, s, args.history)
        gru = {"S": S, "M": M, "epochs": args.epochs, "hidden": args.hidden}

    varies = float((df.groupby(["boardid", "sprintid"])["planday"].nunique() > 1).mean())
    rules, n_prev = rule_scores(df, s, args.history)
    print(f"{len(s)} samples, {len(set(boards))} boards")
    print(f"planday differs between the 4 snapshots of the same sprint in {varies:.1%} of sprints "
          f"(~0% = it is the sprint length, NOT how far into the sprint we are)")
    print(f"samples without any previous sprint on their board: {(n_prev == 0).mean():.1%}")
    print(f"*_imputed: served WITHOUT {' and '.join(UNKNOWN_AT_PLANNING)} (replaced by the training median)")
    print("*_served (snapshot 0 only): the same sprints rewritten as PlanPilot describes them: nothing done, no scope "
          "change, everything to do, plus the two inputs above imputed")

    cols = [sd.RAW.index(c) for c in UNKNOWN_AT_PLANNING]  # base columns come first in every matrix
    scores = {**rules, **out_of_fold(s, mats, args.folds, args.seed, cols, SERVED, gru, sim)}
    scores.update(plan_oof(s, df, args.folds, args.seed, args.history, sd.RAW.index("planday"), sd.RAW.index("no_admin")))

    for t in sd.SNAPSHOTS:
        m = snap == t
        sub = {name: sc[m] for name, sc in scores.items() if t == 0 or not name.endswith(("_served", "_plan"))}
        boot = board_bootstrap(y[m], sub, boards[m], args.boot, args.seed)
        base = y[m].mean()
        print(f"\n=== snapshot {t}%: {m.sum()} sprints, under-delivery rate {base:.1%}"
              f" (coin-flip PR-AUC {base:.3f}, Brier {base * (1 - base):.3f})")
        print(f"{'model':<22}{'PR-AUC':>8}   95% interval (boards)   {'ROC-AUC':>8}{'Brier':>8}")
        full = {}
        for name, sc in sub.items():
            full[name] = average_precision_score(y[m], sc)
            lo, hi = interval(boot[name])
            brier = "-" if name in RULES else f"{brier_score_loss(y[m], sc):.3f}"
            print(f"{name:<22}{full[name]:>8.3f}   [{lo:.3f}, {hi:.3f}]{'':<10}{roc_auc_score(y[m], sc):>8.3f}{brier:>8}")

        print("paired differences in PR-AUC (first minus second, same board resamples)")
        for a, b in COMPARISONS:
            if a not in boot or b not in boot or (t == 0 and "burn_rate" in (a, b)):
                continue  # model not run, or burn_rate is constant at 0 % (nothing is done yet)
            d = boot[a] - boot[b]
            lo, hi = interval(d)
            verdict = "clear" if lo > 0 or hi < 0 else "not clear"
            tag = "  (primary)" if t == 0 and (a, b) in PRIMARY else ""
            print(f"  {a + ' - ' + b:<42}{full[a] - full[b]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"first better in {np.mean(d > 0):>4.0%}   {verdict}{tag}")


if __name__ == "__main__":
    main()
