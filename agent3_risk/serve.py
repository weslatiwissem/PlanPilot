"""Serve the trained sprint-risk models inside Agent 3.

forecast_sprints(state) -> one Forecast per planned sprint: the chance that the sprint delivers less than it commits.

  * The history model when the user gave past sprints, the current-plan model otherwise (config["serve"] names them,
    and may switch the second one off). Trained by train_plan.py on planning-time rows.
  * The planned sprint is turned into the planning-time snapshot (nothing done yet, no scope change) and passed
    through the SAME feature code as the training data (sprint_data), so features cannot drift apart.
  * `planday` (sprint length) comes from state["sprint_days"] if given, else the training median; `no_admin` is
    always the training median (eval_snapshots measured the cost of that).
  * No model files, or any problem: no forecast and a log line. The rule-based risks never depend on this.

Heavy libraries (pandas, lightgbm) are imported inside the functions: on Windows chromadb must be imported before
pandas (see agent1_scoping/rag.py), and the app imports this module before Agent 1 loads chromadb.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.schemas import Forecast, OpsResult, PastSprint, Scope, SprintPlan

log = logging.getLogger(__name__)
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models" / "risk_plan"  # python -m agent3_risk.train_plan --save
FINISHED_COLS = ["boardid", "sprintid", "pos", "snap", "commit", "final_done", "no_teammember", "no_admin",
                 "no_issue_starttime", "no_issue_added", "no_issue_removed"]
_cache: Dict[str, Optional[dict]] = {}


def clear_cache() -> None:
    _cache.clear()


def model_dir() -> Path:
    return Path(os.getenv("RISK_MODEL_DIR") or DEFAULT_DIR)


def served_name(cfg: dict, has_history: bool) -> Optional[str]:
    """Which saved model answers: config["serve"] says it; None means "no forecast in this situation"."""
    names = cfg.get("serve") or {"history": "B2_plus", "no_history": "B1_plus"}  # models from train_final
    return names["history" if has_history else "no_history"]


def load_models() -> Optional[dict]:
    """The saved models and their config, or None if none were trained yet."""
    d = model_dir()
    key = str(d)
    if key not in _cache:
        if not (d / "config.json").exists():
            log.warning("no risk model in %s (train one: python -m agent3_risk.train_plan --save)", d)
            _cache[key] = None
        else:
            import lightgbm as lgb

            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            cfg["boosters"] = {n: lgb.Booster(model_file=str(d / spec["file"])) for n, spec in cfg["models"].items()}
            _cache[key] = cfg
    return _cache[key]


def build_features(sprints: List[dict], past: List[dict], team: int, planday: float, no_admin: float, history: int):
    """Planned sprints (+ finished ones) in the layout of the training data -> (X1, names1, X2, names2)."""
    import pandas as pd

    from agent3_risk import sprint_data as sd

    finished = pd.DataFrame(
        [dict(boardid=0, sprintid=100 + k, pos=k, snap=80, commit=p["committed"], final_done=p["delivered"],
              no_teammember=team, no_admin=no_admin, no_issue_starttime=0, no_issue_added=0, no_issue_removed=0)
         for k, p in enumerate(past)],
        columns=FINISHED_COLS,
    )
    current = pd.DataFrame([
        dict(boardid=0, sprintid=1000 + i, pos=len(past), snap=0, commit=sp["points"], planday=planday,
             no_issue_starttime=sp["n"], vel_starttime=sp["points"], no_issue_added=0, vel_added=0,
             no_issue_removed=0, vel_removed=0, no_issuetodo=sp["n"], vel_todo=sp["points"],
             no_issueinprogress=0, vel_inprogress=0, no_issuedone=0, vel_done=0, no_admin=no_admin,
             no_teammember=team)
        for i, sp in enumerate(sprints)
    ])
    X1, names1 = sd.tabular_plus(current)
    if not past:
        return X1, names1, None, None
    X2, names2 = sd.tabular_history_plus(finished, current, history)
    return X1, names1, X2, names2


def forecast_sprints(state: Dict[str, Any]) -> List[Forecast]:
    """Never raises: a problem is logged and gives no forecast."""
    try:
        cfg = load_models()
        if cfg is None:
            return []
        plan = SprintPlan.model_validate(state.get("sprint_plan") or {})
        ops = OpsResult.model_validate(state.get("ops_result") or {})
        scope = Scope.model_validate(state.get("scope") or {})
        points = {t.title: t.points for t in ops.tickets} or {
            s.title: s.estimate_points or 0 for e in scope.epics for s in e.stories}
        team = len(state.get("team_members") or []) or len({t.assignee for t in ops.tickets if t.assignee}) or 1
        sprints = [{"number": sp.number, "n": len(sp.items), "points": float(sum(points.get(t, 0) for t in sp.items))}
                   for sp in plan.sprints if sp.items]
        sprints = [sp for sp in sprints if sp["points"] > 0]
        if not sprints:
            return []
        past = [PastSprint.model_validate(p).model_dump() for p in (state.get("past_sprints") or [])]

        from agent3_risk import sprint_data as sd

        planday = float(min(state["sprint_days"], sd.PLANDAY_CAP)) if state.get("sprint_days") else cfg["impute"]["planday"]
        X1, names1, X2, names2 = build_features(sprints, past, team, planday, cfg["impute"]["no_admin"], cfg["history"])
        name = served_name(cfg, bool(past))
        if name is None:
            return []  # e.g. trained with --no-b1: no trustworthy forecast without past sprints
        X, names = (X2, names2) if past else (X1, names1)
        booster = cfg["boosters"][name]
        if names != cfg["models"][name]["features"] or booster.num_feature() != len(names):
            raise ValueError("the feature layout changed since the model was trained: retrain it")
        proba = booster.predict(X)
        return [
            Forecast(sprint=sp["number"], probability=float(p), reference_rate=cfg.get("base_rate"), model=name,
                     history_used=min(len(past), cfg["history"]))
            for sp, p in zip(sprints, proba)
        ]
    except Exception as e:
        log.warning("sprint forecast skipped: %s", e)
        return []
