"""MLflow tracking, model registry and champion-challenger gate for M3.

Used by train_estimator.py when run with --mlflow:
    python -m agent1_scoping.train_estimator --mlflow

What it records (all visible at http://localhost:5000):
  * params     : seed, split sizes, embedding width, LightGBM hyper-parameters
  * metrics    : test MAE / MdAE for every predictor, bootstrap CIs, per-project wins,
                 and the validation-loss curve (one point per boosting round)
  * artifacts  : the LightGBM model + meta.json (team medians) needed to reproduce a prediction
  * registry   : every run registers a new version of "planpilot-m3-estimator"
  * aliases    : the new version is always "challenger"; it becomes "champion" ONLY if its
                 test MAE beats the current champion's (the gate from the proposal)
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

EXPERIMENT = "planpilot-m3-estimator"
MODEL_NAME = "planpilot-m3-estimator"
GATE_METRIC = "test_mae_lightgbm_team_scaled"  # the number the gate compares


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def champion_gate(client: MlflowClient, new_value: float, name: str = MODEL_NAME) -> Dict:
    """Return {"promote": bool, "champion_mae": float|None}. Lower MAE is better."""
    try:
        champ = client.get_model_version_by_alias(name, "champion")
    except Exception:  # no champion yet (or model not registered yet)
        return {"promote": True, "champion_mae": None}
    old = client.get_run(champ.run_id).data.metrics.get(GATE_METRIC)
    if old is None:  # champion has no comparable metric -> don't block
        return {"promote": True, "champion_mae": None}
    return {"promote": new_value < old, "champion_mae": float(old)}


def log_run(res: Dict, sizes: Dict[str, int], embed_dim: int, seed: int = 42, tracking_uri: Optional[str] = None,
            run_name: Optional[str] = None) -> Dict:
    """Log one training result (the dict returned by train_estimator.run) to MLflow."""
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    art = res["artifacts"]
    model = art["model"]

    with mlflow.start_run(run_name=run_name) as run:
        # --- params ---
        mlflow.log_params({"seed": seed, "embed_dim": embed_dim, "best_iteration": res["best_iteration"],
                           "n_train": sizes["train"], "n_val": sizes["val"], "n_test": sizes["test"],
                           "n_projects": res["projects"]})
        mlflow.log_params({f"lgbm_{k}": v for k, v in model.get_params().items()
                           if k in ("objective", "learning_rate", "num_leaves", "subsample", "colsample_bytree")})

        # --- metrics: every predictor on the test split ---
        for name, m in res["metrics"].items():
            mlflow.log_metric(f"test_mae_{name}", m["mae"])
            mlflow.log_metric(f"test_mdae_{name}", m["mdae"])
        mlflow.log_metric("projects_won", res["projects_won"])
        for name, (mean, lo, hi) in res["bootstrap"].items():
            s = _slug(name)
            mlflow.log_metrics({f"boot_{s}_mean": mean, f"boot_{s}_lo": lo, f"boot_{s}_hi": hi})

        # --- validation curve: one point per boosting round (shows early stopping) ---
        curve = (getattr(model, "evals_result_", None) or {}).get("valid_0", {}).get("l1", [])
        for step, v in enumerate(curve):
            mlflow.log_metric("val_l1", float(v), step=step)

        # --- artifacts: model (+ registry) and what's needed to reproduce a prediction ---
        info = mlflow.lightgbm.log_model(model.booster_, artifact_path="model", registered_model_name=MODEL_NAME)
        meta = {"team_medians": art["team_medians"], "global_median": art["global_median"], "embed_dim": embed_dim}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "meta.json"
            p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(p))

        # --- champion-challenger gate ---
        new_mae = res["metrics"]["lightgbm_team_scaled"]["mae"]
        version = str(info.registered_model_version)
        gate = champion_gate(client, new_mae)
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        if gate["promote"]:
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
        mlflow.set_tag("gate", "promoted" if gate["promote"] else "rejected")
        mlflow.log_metric("champion_mae_before", gate["champion_mae"] if gate["champion_mae"] is not None else float("nan"))

        return {"run_id": run.info.run_id, "version": version, "promoted": gate["promote"],
                "new_mae": new_mae, "champion_mae": gate["champion_mae"]}
