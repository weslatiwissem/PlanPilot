"""M5 - sprint under-delivery risk: GRU over sprint history vs tabular baselines.

    python -m agent3_risk.train_risk --no-gru              # baselines only (no torch needed)
    python -m agent3_risk.train_risk --epochs 2            # quick smoke test of the GRU
    python -m agent3_risk.train_risk --mlflow --save       # full run, log + register + save the model

Question: does reading the board's last sprints as a SEQUENCE beat tabular models?
  B1  LightGBM on the current snapshot only                      (proposal's baseline: "single-sprint snapshot")
  B2  LightGBM on snapshot + previous 3 sprints, flattened       (tabular model that also sees the history)
  GRU GRU over [3 previous sprints ... current snapshot]         (the sequence model)

Evaluation: 5-fold CV where WHOLE BOARDS are held out (no board is in both train and test), metric PR-AUC
(prevalence is ~64%, so ROC-AUC alone flatters), reported per fold with mean +- std because only ~60 boards exist.
Label = the sprint delivers less than it committed (see sprint_data.py; vel_diff is never a feature).
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from agent3_risk import sprint_data as sd

DATA_DIR = "data/raw/tse2017_jira"
EXPERIMENT = "planpilot-m5-risk"
MODEL_NAME = "planpilot-m5-risk-gru"
GATE_METRIC = "cv_pr_auc_gru"  # higher is better
SAVE_DIR = "models/risk_gru"


def _val_boards(train_boards: np.ndarray, seed: int) -> set:
    """~1/7 of the training boards are kept aside for early stopping."""
    ids = np.array(sorted(set(train_boards)))
    return set(np.random.default_rng(seed).choice(ids, max(1, len(ids) // 7), replace=False))


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


# --------------------------------------------------------------------------- LightGBM baselines
def lgbm_predict(Xtr, ytr, Xva, yva, Xte, seed: int) -> np.ndarray:
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    return m.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- GRU
def normalise_stats(S: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std per step-feature over the VALID steps of the training rows only."""
    valid = S[M > 0]
    return valid.mean(axis=0), valid.std(axis=0) + 1e-6


def prepare(S: np.ndarray, M: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Standardise, zero the padded steps and append the mask as an extra channel."""
    Z = (S - mu) / sig * M[:, :, None]
    return np.concatenate([Z, M[:, :, None]], axis=2).astype(np.float32)


def make_gru(n_in: int, hidden: int = 32, dropout: float = 0.2):
    import torch.nn as nn

    class SprintGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):  # x: (batch, steps, n_in) -> logit (batch,)
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    return SprintGRU()


def train_gru(Ztr, ytr, Zva, yva, seed: int, epochs: int = 60, hidden: int = 32, lr: float = 1e-3, patience: int = 8):
    """Adam + BCE, early-stopped on validation PR-AUC. Returns (best model, best validation PR-AUC)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = make_gru(Ztr.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt, Xv = torch.from_numpy(Ztr), torch.from_numpy(ytr.astype(np.float32)), torch.from_numpy(Zva)

    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(yva, torch.sigmoid(model(Xv)).numpy())
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, float(best)


def gru_predict(model, Z: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(Z))).numpy()


# --------------------------------------------------------------------------- registry + champion gate
def register_and_gate(run_id: str, new_value: float, artifact_path: str = "gru") -> Dict:
    """Register the run's model files and set aliases. 'challenger' always; 'champion' only if it beats the old one."""
    from mlflow.tracking import MlflowClient
    import mlflow

    client = MlflowClient()
    promote, old = True, None
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        old = client.get_run(champ.run_id).data.metrics.get(GATE_METRIC)
        promote = old is None or new_value > old
    except Exception:  # no champion yet
        pass
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    return {"version": str(mv.version), "promoted": promote, "champion_before": old}


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--no-gru", action="store_true", help="baselines only (no torch needed)")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--save", action="store_true", help="train a final GRU on all boards and save it to models/risk_gru")
    args = p.parse_args(argv)

    df = sd.load_all(args.data_dir)
    s = sd.samples(df)
    y = s["y"].to_numpy()
    X1, _ = sd.tabular(s)
    X2, _ = sd.tabular_history(df, s, args.history)
    S, M = sd.sequences(df, s, args.history)
    boards = s["boardid"].to_numpy()
    board_fold = sd.board_folds(boards, args.folds, args.seed)
    fold = np.array([board_fold[int(b)] for b in boards])
    print(f"{len(s)} samples from {len(set(boards))} boards; under-delivery rate {y.mean():.1%}; "
          f"{args.folds}-fold CV, whole boards held out")

    models = ["B1_snapshot", "B2_snapshot_history"] + ([] if args.no_gru else ["GRU"])
    res = {m: [] for m in models}
    for k in range(args.folds):
        te = fold == k
        tr = ~te
        va = tr & np.isin(boards, list(_val_boards(boards[tr], args.seed + k)))
        tr2 = tr & ~va
        res["B1_snapshot"].append(metrics(y[te], lgbm_predict(X1[tr2], y[tr2], X1[va], y[va], X1[te], args.seed)))
        res["B2_snapshot_history"].append(metrics(y[te], lgbm_predict(X2[tr2], y[tr2], X2[va], y[va], X2[te], args.seed)))
        if not args.no_gru:
            mu, sig = normalise_stats(S[tr2], M[tr2])
            Z = prepare(S, M, mu, sig)
            model, _ = train_gru(Z[tr2], y[tr2], Z[va], y[va], args.seed, args.epochs, args.hidden)
            res["GRU"].append(metrics(y[te], gru_predict(model, Z[te])))
        print(f"fold {k}: " + "   ".join(f"{m} PR-AUC {res[m][-1]['pr_auc']:.3f}" for m in models), flush=True)

    print(f"\nprevalence (a random model's PR-AUC): {y.mean():.3f}")
    summary = {}
    for m in models:
        ap = np.array([r["pr_auc"] for r in res[m]])
        auc = np.array([r["roc_auc"] for r in res[m]])
        summary[m] = {"pr_auc": float(ap.mean()), "pr_auc_std": float(ap.std()), "roc_auc": float(auc.mean())}
        print(f"{m:<22} PR-AUC {ap.mean():.3f} +- {ap.std():.3f}   ROC-AUC {auc.mean():.3f}")
    if "GRU" in res:
        for base in ("B1_snapshot", "B2_snapshot_history"):
            d = np.array([g["pr_auc"] - b["pr_auc"] for g, b in zip(res["GRU"], res[base])])
            print(f"GRU - {base}: mean {d.mean():+.3f}, GRU better in {(d > 0).sum()}/{len(d)} folds")

    out = {}
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="M5 sequence vs tabular") as run:
            mlflow.log_params({"folds": args.folds, "seed": args.seed, "history": args.history, "epochs": args.epochs,
                               "hidden": args.hidden, "n_samples": len(s), "n_boards": len(set(boards))})
            mlflow.log_metric("prevalence", float(y.mean()))
            for m in models:
                key = {"B1_snapshot": "b1", "B2_snapshot_history": "b2", "GRU": "gru"}[m]
                mlflow.log_metrics({f"cv_pr_auc_{key}": summary[m]["pr_auc"], f"cv_pr_auc_{key}_std": summary[m]["pr_auc_std"],
                                    f"cv_roc_auc_{key}": summary[m]["roc_auc"]})
                for k, r in enumerate(res[m]):
                    mlflow.log_metric(f"fold_pr_auc_{key}", r["pr_auc"], step=k)
            out["run_id"] = run.info.run_id

            if args.save and not args.no_gru:
                # final model: all boards, ~1/7 of them for early stopping
                va = np.isin(boards, list(_val_boards(boards, args.seed)))
                tr2 = ~va
                mu, sig = normalise_stats(S[tr2], M[tr2])
                Z = prepare(S, M, mu, sig)
                model, best = train_gru(Z[tr2], y[tr2], Z[va], y[va], args.seed, args.epochs, args.hidden)
                import torch

                d = Path(SAVE_DIR)
                d.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), d / "model.pt")
                (d / "config.json").write_text(json.dumps({
                    "n_in": int(Z.shape[2]), "hidden": args.hidden, "history": args.history, "steps": sd.STEP_FEATURES,
                    "mu": mu.tolist(), "sigma": sig.tolist(), "label": "sprint delivers less than committed"}, indent=2))
                mlflow.log_artifacts(str(d), artifact_path="gru")
                g = register_and_gate(run.info.run_id, summary["GRU"]["pr_auc"])
                mlflow.set_tag("gate", "promoted" if g["promoted"] else "rejected")
                print(f"\nMLflow: model v{g['version']}: {'PROMOTED to champion' if g['promoted'] else 'REJECTED, champion kept'}")
        print("logged to MLflow experiment", EXPERIMENT)


if __name__ == "__main__":
    main()
