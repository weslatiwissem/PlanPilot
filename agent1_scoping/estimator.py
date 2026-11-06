"""M3 at inference time: rescale story points to ONE team's own scale.

Train + save:   python -m agent1_scoping.train_estimator --save
Artifacts:      models/estimator/model.txt, models/estimator/meta.json  (track with DVC)
Use:            calibrate_scope(scope, team_id)  -> story points in that team's units

team_id is the name of a team in the training data (e.g. "mesos", "moodle").
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# chromadb (via rag) must load before pandas / lightgbm: see rag.py
from agent1_scoping.rag import _default_embedding_function  # isort: skip

import lightgbm as lgb  # isort: skip
import numpy as np  # isort: skip
import pandas as pd  # isort: skip

from orchestrator.schemas import Scope  # isort: skip

log = logging.getLogger(__name__)
ARTIFACT_DIR = os.getenv("ESTIMATOR_DIR", "models/estimator")
_loaded: Dict[str, Tuple[lgb.Booster, dict]] = {}


def save_estimator(model, team_medians: dict, global_median: float, embed_dim: int, path: str = ARTIFACT_DIR) -> None:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(d / "model.txt"))  # saves the best iteration
    meta = {
        "team_medians": {k: float(v) for k, v in team_medians.items()},
        "global_median": float(global_median),
        "embed_dim": int(embed_dim),
        "embed_model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _loaded.pop(path, None)


def load_estimator(path: str = ARTIFACT_DIR) -> Tuple[lgb.Booster, dict]:
    if path not in _loaded:
        d = Path(path)
        if not (d / "model.txt").exists():
            raise FileNotFoundError(
                f"No estimator in {path}. Train it with: python -m agent1_scoping.train_estimator --save"
            )
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        _loaded[path] = (lgb.Booster(model_file=str(d / "model.txt")), meta)
    return _loaded[path]


def known_teams(path: str = ARTIFACT_DIR) -> List[str]:
    return sorted(load_estimator(path)[1]["team_medians"])


def predict_points(
    stories: List[Tuple[str, str]],
    team_id: str,
    embed: Optional[Callable] = None,
    path: str = ARTIFACT_DIR,
) -> np.ndarray:
    """Predicted points (float) for (title, description) pairs, in the team's own units."""
    from agent1_scoping.train_estimator import make_features

    booster, meta = load_estimator(path)
    medians = pd.Series(meta["team_medians"])
    if team_id not in medians.index:
        raise KeyError(f"Unknown team '{team_id}'. Known teams: {', '.join(sorted(medians.index))}")

    df = pd.DataFrame(
        {"title": [t for t, _ in stories], "description": [d for _, d in stories], "project": team_id}
    )
    embed = embed or _default_embedding_function()
    emb = np.asarray(embed([f"{t}. {d}".strip() for t, d in stories]), dtype=np.float32)
    X, med = make_features(df, emb, medians, meta["global_median"])
    return med * np.exp(booster.predict(X))


def calibrate_scope(scope: Scope, team_id: str, embed: Optional[Callable] = None, path: str = ARTIFACT_DIR) -> Scope:
    """Replace each story's estimate with the team-scaled prediction (whole points, >= 1)."""
    stories = [s for epic in scope.epics for s in epic.stories]
    if not stories:
        return scope
    points = predict_points([(s.title, s.description) for s in stories], team_id, embed, path)
    for story, p in zip(stories, points):
        story.estimate_points = max(1, int(round(float(p))))
    return scope
