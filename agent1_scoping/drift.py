"""Data-drift report for the M3 estimator, built with Evidently.

    python -m agent1_scoping.drift                 # writes reports/drift_train_vs_test.html
    python -m agent1_scoping.drift --mlflow        # ...and logs it to MLflow as an artifact

Compares the TRAIN split (reference) with the TEST split (current) on:
  * text-length features   (title length, description length, word count)
  * the target             (story points)
  * 5 principal components of the story embeddings (a compact view of "what the text is about")

Why it matters: the proposal monitors drift to decide when to retrain. A large drift share means the
data the model sees in production no longer looks like what it was trained on.
Embeddings come from the same cache as train_estimator (.cache/), so this is fast after the first run.
"""
import argparse
import os
from pathlib import Path
from typing import Optional

# chromadb (via rag) must load before pandas: see rag.py
from agent1_scoping.rag import _default_embedding_function  # isort: skip
from agent1_scoping.index_stories import DEFAULT_DIR, load_stories  # isort: skip
from agent1_scoping.train_estimator import CACHE, _texts, embed_cached  # isort: skip

import numpy as np  # isort: skip
import pandas as pd  # isort: skip

REPORT_PATH = "reports/drift_train_vs_test.html"
N_COMPONENTS = 5


def build_frame(df: pd.DataFrame, emb: np.ndarray, components: np.ndarray, mean: np.ndarray) -> pd.DataFrame:
    """One row per story: length features, target and PCA projection of its embedding."""
    out = pd.DataFrame(
        {
            "title_len": df["title"].str.len().to_numpy(dtype=float),
            "description_len": df["description"].str.len().to_numpy(dtype=float),
            "word_count": (df["title"] + " " + df["description"]).str.split().str.len().to_numpy(dtype=float),
            "storypoint": df["storypoint"].to_numpy(dtype=float),
        }
    )
    proj = (emb - mean) @ components.T
    for i in range(proj.shape[1]):
        out[f"text_pc{i + 1}"] = proj[:, i]
    return out


def fit_pca(emb: np.ndarray, k: int = N_COMPONENTS):
    """Plain-numpy PCA fitted on the reference (train) embeddings only."""
    mean = emb.mean(axis=0)
    _, _, vt = np.linalg.svd(emb - mean, full_matrices=False)
    return vt[:k], mean


def make_report(reference: pd.DataFrame, current: pd.DataFrame, path: str = REPORT_PATH) -> dict:
    """Run Evidently's data-drift preset, save the HTML and return a small summary dict."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(path)

    result = report.as_dict()["metrics"][0]["result"]
    drifted = [
        c for c, v in report.as_dict()["metrics"][1]["result"]["drift_by_columns"].items() if v["drift_detected"]
    ]
    return {
        "dataset_drift": bool(result["dataset_drift"]),
        "share_drifted": float(result["share_of_drifted_columns"]),
        "n_drifted": int(result["number_of_drifted_columns"]),
        "n_columns": int(result["number_of_columns"]),
        "drifted_columns": drifted,
    }


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--out", default=REPORT_PATH)
    p.add_argument("--mlflow", action="store_true", help="log the report and drift metrics to MLflow")
    args = p.parse_args(argv)

    df = load_stories(args.data_dir, train_only=False)
    marks = df["split_mark"].astype(str).str.lower()
    parts = {s: df[marks.str.startswith(s)].reset_index(drop=True) for s in ("train", "val", "test")}

    texts = sum((_texts(parts[s]) for s in ("train", "val", "test")), [])
    emb = embed_cached(texts, _default_embedding_function(), args.cache)
    n_tr, n_va = len(parts["train"]), len(parts["val"])
    emb_train, emb_test = emb[:n_tr], emb[n_tr + n_va :]

    components, mean = fit_pca(emb_train)
    reference = build_frame(parts["train"], emb_train, components, mean)
    current = build_frame(parts["test"], emb_test, components, mean)

    summary = make_report(reference, current, args.out)
    print(f"\nreport saved to {args.out}")
    print(f"drifted columns: {summary['n_drifted']}/{summary['n_columns']} ({summary['share_drifted']:.0%})")
    print(f"dataset drift detected: {summary['dataset_drift']}  ->  {summary['drifted_columns']}")

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("planpilot-monitoring")
        with mlflow.start_run(run_name="drift-train-vs-test"):
            mlflow.log_metrics(
                {
                    "drift_share": summary["share_drifted"],
                    "drifted_columns": summary["n_drifted"],
                    "dataset_drift": float(summary["dataset_drift"]),
                }
            )
            mlflow.set_tag("drifted_columns", ", ".join(summary["drifted_columns"]) or "none")
            mlflow.log_artifact(args.out)
        print("logged to MLflow experiment 'planpilot-monitoring'")


if __name__ == "__main__":
    main()
