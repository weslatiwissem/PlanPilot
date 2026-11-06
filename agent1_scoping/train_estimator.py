"""M3 experiment: can the story text + the team's own scale beat the median?

    python -m agent1_scoping.train_estimator

Idea: the team's median points sets the scale; LightGBM learns, from the story text
(embedding), a multiplicative adjustment around it:
    points ~= team_median * exp(model(text))
It is trained with an L1 objective (median-optimal, robust to the 100-point outliers),
early-stopped on the val split and judged on the test split, against the same
baselines as evaluate_estimates.py. Embeddings are cached in .cache/ (first run is slow).

Two checks keep the claim honest: an ablation model WITHOUT the text embedding (does the
text matter, or only team scale + text length?) and paired bootstrap 95% intervals on the
MAE differences.
"""
import argparse
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional

# chromadb (via rag) must load before pandas: see rag.py
from agent1_scoping.rag import _default_embedding_function  # isort: skip
from agent1_scoping.index_stories import DEFAULT_DIR, load_stories  # isort: skip

import lightgbm as lgb  # isort: skip
import numpy as np  # isort: skip
import pandas as pd  # isort: skip

CACHE = ".cache/story_embeddings.npz"


def _texts(df: pd.DataFrame) -> List[str]:
    return [f"{t}. {d}".strip() for t, d in zip(df["title"], df["description"])]


def embed_cached(texts: List[str], embed: Callable, cache: Optional[str] = CACHE, batch: int = 512) -> np.ndarray:
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["emb"]
    chunks = []
    for i in range(0, len(texts), batch):
        chunks.append(np.asarray(embed(texts[i : i + batch]), dtype=np.float32))
        print(f"embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    emb = np.vstack(chunks)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=key, emb=emb)
    return emb


def make_features(df: pd.DataFrame, emb: np.ndarray, team_median: pd.Series, global_median: float):
    med = df["project"].map(team_median).fillna(global_median).to_numpy(dtype=float)
    extra = np.column_stack(
        [
            np.log1p(df["title"].str.len().to_numpy(dtype=float)),
            np.log1p(df["description"].str.len().to_numpy(dtype=float)),
            np.log(med),
        ]
    )
    return np.hstack([emb, extra]), med


def _new_model(seed: int):
    return lgb.LGBMRegressor(
        objective="l1", n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed, verbose=-1,
    )


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Mean(err_a - err_b) with a 95% bootstrap interval. Negative = A is better."""
    diff = err_a - err_b
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(train, val, test, emb_train, emb_val, emb_test, seed: int = 42) -> Dict:
    """Train on train, early-stop on val, evaluate on test. Returns metrics per predictor."""
    gm = float(train["storypoint"].median())
    tm = train.groupby("project")["storypoint"].median()

    Xtr, mtr = make_features(train, emb_train, tm, gm)
    Xva, mva = make_features(val, emb_val, tm, gm)
    Xte, mte = make_features(test, emb_test, tm, gm)
    ytr = np.log(train["storypoint"].to_numpy(dtype=float) / mtr)
    yva = np.log(val["storypoint"].to_numpy(dtype=float) / mva)

    d = emb_train.shape[1]  # embedding width: columns after it are scale/length features
    model = _new_model(seed)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    ablation = _new_model(seed)  # same model, text embedding removed
    ablation.fit(Xtr[:, d:], ytr, eval_set=[(Xva[:, d:], yva)], callbacks=[lgb.early_stopping(50, verbose=False)])

    y = test["storypoint"].to_numpy(dtype=float)
    preds = {
        "global_median": np.full(len(y), gm),
        "project_median": mte,
        "lightgbm_no_text": mte * np.exp(ablation.predict(Xte[:, d:])),
        "lightgbm_team_scaled": mte * np.exp(model.predict(Xte)),
    }
    metrics = {n: {"mae": float(np.abs(p - y).mean()), "mdae": float(np.median(np.abs(p - y)))} for n, p in preds.items()}

    wins = 0
    projects = test["project"].to_numpy()
    names = sorted(set(projects))
    for pr in names:
        m = projects == pr
        wins += np.abs(preds["lightgbm_team_scaled"][m] - y[m]).mean() < np.abs(preds["project_median"][m] - y[m]).mean()
    err = {n: np.abs(p - y) for n, p in preds.items()}
    bootstrap = {
        "text model vs project median": paired_bootstrap(err["lightgbm_team_scaled"], err["project_median"]),
        "no-text model vs project median": paired_bootstrap(err["lightgbm_no_text"], err["project_median"]),
        "text model vs no-text model": paired_bootstrap(err["lightgbm_team_scaled"], err["lightgbm_no_text"]),
    }
    return {
        "metrics": metrics,
        "bootstrap": bootstrap,
        "best_iteration": int(model.best_iteration_ or 0),
        "projects_won": int(wins),
        "projects": len(names),
        "artifacts": {"model": model, "team_medians": tm.to_dict(), "global_median": gm},
    }


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--save", action="store_true", help="save the trained model to models/estimator")
    args = p.parse_args(argv)

    df = load_stories(args.data_dir, train_only=False)
    marks = df["split_mark"].astype(str).str.lower()
    parts = {s: df[marks.str.startswith(s)].reset_index(drop=True) for s in ("train", "val", "test")}
    print({s: len(d) for s, d in parts.items()})

    texts = sum((_texts(parts[s]) for s in ("train", "val", "test")), [])
    emb = embed_cached(texts, _default_embedding_function(), args.cache)
    n_tr, n_va = len(parts["train"]), len(parts["val"])

    res = run(
        parts["train"], parts["val"], parts["test"],
        emb[:n_tr], emb[n_tr : n_tr + n_va], emb[n_tr + n_va :],
    )
    print(f"\nbest iteration: {res['best_iteration']}")
    print(f"{'predictor':<24}{'MAE':>8}{'MdAE':>8}")
    for name, m in res["metrics"].items():
        print(f"{name:<24}{m['mae']:>8.2f}{m['mdae']:>8.2f}")
    print(f"\nLightGBM beats the project median in {res['projects_won']}/{res['projects']} projects (MAE).")
    print("\nMAE difference, mean [95% bootstrap interval] (negative = first is better):")
    for name, (mean, lo, hi) in res["bootstrap"].items():
        print(f"  {name:<34}{mean:>7.3f}  [{lo:.3f}, {hi:.3f}]")

    if args.save:
        from agent1_scoping.estimator import ARTIFACT_DIR, save_estimator

        a = res["artifacts"]
        save_estimator(a["model"], a["team_medians"], a["global_median"], emb.shape[1])
        print(f"\nsaved estimator to {ARTIFACT_DIR} (track it with: dvc add {ARTIFACT_DIR})")


if __name__ == "__main__":
    main()
