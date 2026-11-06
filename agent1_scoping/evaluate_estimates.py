"""Does retrieval help to estimate story points? Measure it on held-out stories.

    python -m agent1_scoping.evaluate_estimates                # test split, k=10
    python -m agent1_scoping.evaluate_estimates --split val -k 5

Index must have been built from the TRAIN split (the default of index_stories).
Predictors compared (lower error is better):
  global_median       median points of all train stories (no information)
  project_median      median points of the same project's train stories (knows the team's scale)
  knn_cross_project   median points of the k most similar stories from ANY project
                      (scenario: a new team with no history)
  knn_within_project  median points of the k most similar stories of the SAME project
                      (scenario: a team that already has history -> "learns from past projects")
"""
import argparse
from typing import Dict, Optional

# chromadb (via rag) must load before pandas: see rag.py
from agent1_scoping.rag import get_collection  # isort: skip
from agent1_scoping.index_stories import DEFAULT_DIR, load_stories  # isort: skip

import numpy as np  # isort: skip
import pandas as pd  # isort: skip

QUERY_BATCH = 256


def _text(df: pd.DataFrame) -> list:
    return [f"{t}. {d}".strip() for t, d in zip(df["title"], df["description"])]


def _knn_median(collection, embeddings, k: int, where: Optional[dict]) -> np.ndarray:
    preds = []
    for start in range(0, len(embeddings), QUERY_BATCH):
        res = collection.query(
            query_embeddings=embeddings[start : start + QUERY_BATCH], n_results=k, where=where
        )
        for metas in res["metadatas"]:
            preds.append(float(np.median([m["points"] for m in metas])))
    return np.array(preds)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, collection, embed, k: int = 10) -> Dict[str, Dict[str, float]]:
    """Return {predictor: {"mae": ..., "mdae": ...}} on the test rows."""
    y_true, preds = [], {n: [] for n in ("global_median", "project_median", "knn_cross_project", "knn_within_project")}
    global_median = float(train["storypoint"].median())
    project_median = train.groupby("project")["storypoint"].median().to_dict()

    for project, part in test.groupby("project"):
        emb = np.asarray(embed(_text(part)), dtype=float).tolist()
        y_true.append(part["storypoint"].to_numpy(dtype=float))
        preds["global_median"].append(np.full(len(part), global_median))
        preds["project_median"].append(np.full(len(part), project_median.get(project, global_median)))
        preds["knn_cross_project"].append(_knn_median(collection, emb, k, None))
        preds["knn_within_project"].append(_knn_median(collection, emb, k, {"project": project}))

    y = np.concatenate(y_true)
    out = {}
    for name, chunks in preds.items():
        err = np.abs(np.concatenate(chunks) - y)
        out[name] = {"mae": float(err.mean()), "mdae": float(np.median(err))}
    return out


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("-k", type=int, default=10)
    args = p.parse_args(argv)

    from agent1_scoping.rag import _default_embedding_function

    everything = load_stories(args.data_dir, train_only=False)
    marks = everything["split_mark"].astype(str).str.lower()
    train = everything[marks.str.startswith("train")]
    target = everything[marks.str.startswith(args.split)]
    print(f"train={len(train)}  {args.split}={len(target)}  k={args.k}")

    results = evaluate(train, target, get_collection(), _default_embedding_function(), k=args.k)
    print(f"\n{'predictor':<22}{'MAE':>8}{'MdAE':>8}")
    for name, m in results.items():
        print(f"{name:<22}{m['mae']:>8.2f}{m['mdae']:>8.2f}")


if __name__ == "__main__":
    main()
