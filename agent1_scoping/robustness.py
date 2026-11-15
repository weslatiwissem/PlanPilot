"""Two checks the proposal promises that train_estimator.run() does not cover.

    python -m agent1_scoping.robustness              # print results
    python -m agent1_scoping.robustness --mlflow     # ...and log them to MLflow

1. COLD START (leave-projects-out)
   train_estimator splits stories within each project, so every test project was also seen in training
   ("a team that already has history"). Here whole projects are held out: the model never saw the team, so it
   has no team median and falls back to the global median ("a brand-new team"). Projects are split into
   4 folds; the report shows MAE per fold and pooled, against the global-median baseline.

2. P50 / P80
   The proposal asks for a median (P50) and a pessimistic (P80) estimate. A P80 is only useful if it is
   calibrated: about 80% of real values should fall at or below it. Coverage and pinball loss are reported.
"""
import argparse
import os
from typing import Dict, Optional

# chromadb (via rag) must load before pandas / lightgbm: see rag.py
from agent1_scoping.rag import _default_embedding_function  # isort: skip
from agent1_scoping.index_stories import DEFAULT_DIR, load_stories  # isort: skip
from agent1_scoping.train_estimator import CACHE, _texts, embed_cached, make_features, paired_bootstrap  # isort: skip

import lightgbm as lgb  # isort: skip
import numpy as np  # isort: skip
import pandas as pd  # isort: skip

N_FOLDS = 4


def _lgbm(seed: int, alpha: Optional[float] = None):
    """Same settings as train_estimator._new_model; alpha=None -> median (L1), else a quantile."""
    kw = dict(objective="l1") if alpha is None else dict(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, random_state=seed, verbose=-1, **kw,
    )


def _fit(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# --------------------------------------------------------------------------- 1. cold start
def cold_start(df: pd.DataFrame, emb: np.ndarray, seed: int = 42, n_folds: int = N_FOLDS) -> Dict:
    """Leave-projects-out: every fold's test projects are completely unseen."""
    projects = sorted(df["project"].unique())
    fold_of = {p: i % n_folds for i, p in enumerate(projects)}
    fold = df["project"].map(fold_of).to_numpy()
    rng = np.random.default_rng(seed)

    folds, err_model, err_base = [], [], []
    for k in range(n_folds):
        test_m = fold == k
        pool = np.flatnonzero(~test_m)
        val_idx = rng.choice(pool, size=max(1, len(pool) // 10), replace=False)  # early-stopping rows
        val_m = np.zeros(len(df), dtype=bool)
        val_m[val_idx] = True
        train_m = ~test_m & ~val_m

        tr, va, te = df[train_m], df[val_m], df[test_m]
        gm = float(tr["storypoint"].median())
        tm = tr.groupby("project")["storypoint"].median()  # test projects are NOT in here -> global median

        Xtr, mtr = make_features(tr, emb[train_m], tm, gm)
        Xva, mva = make_features(va, emb[val_m], tm, gm)
        Xte, mte = make_features(te, emb[test_m], tm, gm)
        model = _fit(_lgbm(seed), Xtr, np.log(tr["storypoint"].to_numpy(float) / mtr),
                     Xva, np.log(va["storypoint"].to_numpy(float) / mva))

        y = te["storypoint"].to_numpy(float)
        pred = mte * np.exp(model.predict(Xte))
        e_m, e_b = np.abs(pred - y), np.abs(gm - y)
        err_model.append(e_m)
        err_base.append(e_b)
        folds.append({"fold": k, "n_projects": int(sum(1 for p in projects if fold_of[p] == k)),
                      "n_test": int(test_m.sum()), "mae_model": float(e_m.mean()), "mae_global_median": float(e_b.mean())})

    em, eb = np.concatenate(err_model), np.concatenate(err_base)
    mean, lo, hi = paired_bootstrap(em, eb)
    return {"folds": folds, "mae_model": float(em.mean()), "mae_global_median": float(eb.mean()),
            "diff": mean, "diff_lo": lo, "diff_hi": hi,
            "folds_won": int(sum(f["mae_model"] < f["mae_global_median"] for f in folds))}


# --------------------------------------------------------------------------- 2. P50 / P80
def quantiles(train, val, test, emb_train, emb_val, emb_test, seed: int = 42, q_hi: float = 0.8) -> Dict:
    """Train P50 and P80 on the standard split; report coverage and pinball loss on test."""
    gm = float(train["storypoint"].median())
    tm = train.groupby("project")["storypoint"].median()
    Xtr, mtr = make_features(train, emb_train, tm, gm)
    Xva, mva = make_features(val, emb_val, tm, gm)
    Xte, mte = make_features(test, emb_test, tm, gm)
    ytr = np.log(train["storypoint"].to_numpy(float) / mtr)
    yva = np.log(val["storypoint"].to_numpy(float) / mva)

    p50 = mte * np.exp(_fit(_lgbm(seed), Xtr, ytr, Xva, yva).predict(Xte))
    p80 = mte * np.exp(_fit(_lgbm(seed, q_hi), Xtr, ytr, Xva, yva).predict(Xte))
    p80 = np.maximum(p80, p50)  # a P80 below the P50 makes no sense; clip

    y = test["storypoint"].to_numpy(float)
    return {"coverage_p50": float((y <= p50).mean()), "coverage_p80": float((y <= p80).mean()),
            "pinball_p50": pinball(y, p50, 0.5), "pinball_p80": pinball(y, p80, q_hi),
            "mean_p50": float(p50.mean()), "mean_p80": float(p80.mean()),
            "mae_p50": float(np.abs(p50 - y).mean())}


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args(argv)

    df = load_stories(args.data_dir, train_only=False)
    marks = df["split_mark"].astype(str).str.lower()
    parts = {s: df[marks.str.startswith(s)].reset_index(drop=True) for s in ("train", "val", "test")}
    texts = sum((_texts(parts[s]) for s in ("train", "val", "test")), [])
    emb = embed_cached(texts, _default_embedding_function(), args.cache)  # same order as `texts`
    n_tr, n_va = len(parts["train"]), len(parts["val"])
    everything = pd.concat([parts["train"], parts["val"], parts["test"]], ignore_index=True)

    cs = cold_start(everything, emb, args.seed)
    print("\nCOLD START: whole projects held out (the model has never seen the team)")
    print(f"{'fold':<6}{'projects':>9}{'stories':>9}{'MAE model':>11}{'MAE global':>12}")
    for f in cs["folds"]:
        print(f"{f['fold']:<6}{f['n_projects']:>9}{f['n_test']:>9}{f['mae_model']:>11.2f}{f['mae_global_median']:>12.2f}")
    print(f"pooled: model {cs['mae_model']:.2f} vs global median {cs['mae_global_median']:.2f}; "
          f"difference {cs['diff']:+.3f} [{cs['diff_lo']:+.3f}, {cs['diff_hi']:+.3f}] (negative = model better); "
          f"model wins {cs['folds_won']}/{len(cs['folds'])} folds")

    qs = quantiles(parts["train"], parts["val"], parts["test"], emb[:n_tr], emb[n_tr : n_tr + n_va],
                   emb[n_tr + n_va :], args.seed)
    print("\nP50 / P80 (team already has history)")
    print(f"coverage P50: {qs['coverage_p50']:.0%} (target 50%)   coverage P80: {qs['coverage_p80']:.0%} (target 80%)")
    print(f"pinball loss P50 {qs['pinball_p50']:.3f}   P80 {qs['pinball_p80']:.3f}   mean estimate P50 {qs['mean_p50']:.2f} -> P80 {qs['mean_p80']:.2f}")

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("planpilot-m3-estimator")
        with mlflow.start_run(run_name="robustness-cold-start-and-quantiles"):
            mlflow.set_tag("kind", "robustness")
            mlflow.log_param("seed", args.seed)
            mlflow.log_metrics({
                "cold_mae_model": cs["mae_model"], "cold_mae_global_median": cs["mae_global_median"],
                "cold_diff": cs["diff"], "cold_diff_lo": cs["diff_lo"], "cold_diff_hi": cs["diff_hi"],
                "cold_folds_won": cs["folds_won"], **{f"q_{k}": v for k, v in qs.items()},
            })
            for f in cs["folds"]:
                mlflow.log_metric("cold_fold_mae_model", f["mae_model"], step=f["fold"])
                mlflow.log_metric("cold_fold_mae_global", f["mae_global_median"], step=f["fold"])
        print("\nlogged to MLflow experiment 'planpilot-m3-estimator'")


if __name__ == "__main__":
    main()
