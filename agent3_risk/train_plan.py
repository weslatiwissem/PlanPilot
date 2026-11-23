"""Train the models PlanPilot serves (planning-time, monotone) and save them where serve.py looks.

    python -m agent3_risk.train_plan                  # train and print a summary
    python -m agent3_risk.train_plan --save           # write models/risk_plan/   (then: dvc add models/risk_plan)
    python -m agent3_risk.train_plan --save --no-b1   # serve NO forecast for projects without past sprints

Only snapshot-0 sprints are used, rewritten with plan_model.planning_view: the model is trained on exactly the kind
of row it will be given. `planday` stays real in training (the user can state it); `no_admin` is the median, as it
will be at serving. B2_plan needs past sprints; B1_plan does not and is weaker, so check eval_snapshots
(B1_plan rows) before serving it: if it does not beat chance clearly, use --no-b1.

Their quality is what eval_snapshots reports for B1_plan / B2_plan, not anything printed here.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from agent3_risk import sprint_data as sd
from agent3_risk.plan_model import fit_plan, monotone_vector, plan_matrices
from agent3_risk.train_risk import DATA_DIR, _val_boards

SAVE_DIR = "models/risk_plan"


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out", default=SAVE_DIR)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-b1", action="store_true", help="do not serve a forecast without past sprints")
    args = p.parse_args(argv)

    df = sd.load_all(args.data_dir)
    s = sd.samples(df)
    y, boards, snap = s["y"].to_numpy(), s["boardid"].to_numpy(), s["snap"].to_numpy()
    s0 = snap == 0
    va = s0 & np.isin(boards, list(_val_boards(boards[s0], args.seed)))
    tr = s0 & ~va
    planday_col, admin_col = sd.RAW.index("planday"), sd.RAW.index("no_admin")

    mats = plan_matrices(df, s, args.history)
    impute = {"planday": float(np.median(mats["B1_plan"][0][tr][:, planday_col])),
              "no_admin": float(np.median(mats["B1_plan"][0][tr][:, admin_col]))}
    models = {}
    for name, (X, names) in mats.items():
        Xtr, Xva = X[tr].copy(), X[va].copy()
        Xtr[:, admin_col] = Xva[:, admin_col] = impute["no_admin"]  # unknown at planning time: constant, as when serving
        models[name] = (fit_plan(Xtr, y[tr], Xva, y[va], names, args.seed), names)

    base_rate = float(y[s0].mean())
    print(f"{int(s0.sum())} planning-time sprints from {len(set(boards[s0]))} boards "
          f"({len(set(boards[va]))} kept for early stopping)")
    for name, (m, names) in models.items():
        print(f"{name}: {len(names)} features, {sum(1 for c in monotone_vector(names) if c)} monotone, "
              f"best iteration {int(m.best_iteration_ or 0)}")
    print(f"share of sprints that under-delivered at planning time (the reference rate): {base_rate:.1%}")
    print(f"training medians used when an input is unknown: {impute}")

    if args.save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "label": "sprint delivers less than committed",
            "kind": "planning",
            "snapshot_served": 0,
            "history": args.history,
            "seed": args.seed,
            "n_samples": int(s0.sum()),
            "n_boards": int(len(set(boards[s0]))),
            "unknown_at_planning": ["no_admin"],
            "impute": impute,
            "base_rate": base_rate,
            "serve": {"history": "B2_plan", "no_history": None if args.no_b1 else "B1_plan"},
            "models": {},
        }
        for name, (m, names) in models.items():
            fname = f"{name.lower()}.txt"
            m.booster_.save_model(str(out / fname))
            cfg["models"][name] = {"file": fname, "features": names, "best_iteration": int(m.best_iteration_ or 0),
                                   "monotone": monotone_vector(names)}
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"saved to {out}  (share it with: dvc add {out})")


if __name__ == "__main__":
    main()
