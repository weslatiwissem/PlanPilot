"""Train the models PlanPilot serves and save them with everything needed to use them.

    python -m agent3_risk.train_final            # train and print a summary
    python -m agent3_risk.train_final --save     # ...and write models/risk_lgbm/ (then: dvc add models/risk_lgbm)

Decision taken from eval_snapshots (planning-time snapshot, whole boards held out):
  B2_plus  LightGBM on snapshot + previous sprints + engineered ratios   -> used when the user gives past sprints
  B1_plus  the same without history                                      -> fallback for a brand-new project
The GRU is not served: it never beat the tabular models.

Both models are trained on every board, ~1/7 of the boards held back only for early stopping (as in
train_risk.py). They are trained on all four snapshots with `snap` as a feature, exactly as they were evaluated;
the server asks for snapshot 0. Their quality is what eval_snapshots reports, not anything printed here.

`planday` (sprint length) and `no_admin` are unknown at planning time. config.json stores their training medians,
which the server must plug in (eval_snapshots checked that this costs little at snapshot 0).
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np

from agent3_risk import sprint_data as sd
from agent3_risk.train_risk import DATA_DIR, _val_boards

SAVE_DIR = "models/risk_lgbm"
UNKNOWN_AT_PLANNING = ("planday", "no_admin")
# Must stay identical to train_risk.lgbm_predict (tests/test_train_final.py compares the two).
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_final(Xtr, ytr, Xva, yva, seed: int) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out", default=SAVE_DIR)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true")
    args = p.parse_args(argv)

    df = sd.load_all(args.data_dir)
    s = sd.samples(df)
    y, boards = s["y"].to_numpy(), s["boardid"].to_numpy()
    X1, names1 = sd.tabular_plus(s)
    X2, names2 = sd.tabular_history_plus(df, s, args.history)

    va = np.isin(boards, list(_val_boards(boards, args.seed)))
    tr = ~va
    cols = [sd.RAW.index(c) for c in UNKNOWN_AT_PLANNING]
    impute = {c: float(np.median(X1[tr][:, i])) for c, i in zip(UNKNOWN_AT_PLANNING, cols)}

    models = {}
    for name, X, names in (("B1_plus", X1, names1), ("B2_plus", X2, names2)):
        models[name] = (fit_final(X[tr], y[tr], X[va], y[va], args.seed), names)
    print(f"{len(s)} samples, {len(set(boards))} boards ({len(set(boards[va]))} kept for early stopping)")
    for name, (m, names) in models.items():
        print(f"{name}: {len(names)} features, best iteration {int(m.best_iteration_ or 0)}")
    print(f"served without {' / '.join(UNKNOWN_AT_PLANNING)}: training medians {impute}")
    base_rate = float(y[s["snap"].to_numpy() == 0].mean())
    print(f"share of sprints that under-delivered at snapshot 0 (the reference rate): {base_rate:.1%}")

    if args.save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "label": "sprint delivers less than committed",
            "snapshot_served": 0,
            "history": args.history,
            "seed": args.seed,
            "n_samples": int(len(s)),
            "n_boards": int(len(set(boards))),
            "unknown_at_planning": list(UNKNOWN_AT_PLANNING),
            "impute": impute,
            "base_rate": base_rate,
            "models": {},
        }
        for name, (m, names) in models.items():
            fname = f"{name.lower()}.txt"
            m.booster_.save_model(str(out / fname))
            cfg["models"][name] = {"file": fname, "features": names, "best_iteration": int(m.best_iteration_ or 0)}
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"saved to {out}  (share it with: dvc add {out})")


if __name__ == "__main__":
    main()
