"""Agent 3 (Person C): risk prediction and monitoring.

Two outputs, deliberately separate:
  risks      rule-based problems with a concrete cause (unassigned ticket, overloaded sprint, weak match ...)
  forecasts  trained-model probability, per sprint, of delivering less than committed (see serve.py)
"""
from agent3_risk.heuristics import assess
from agent3_risk.serve import forecast_sprints
from orchestrator.schemas import OpsResult, Scope, SprintPlan
from orchestrator.state import PlanState


def assess_risk(state: PlanState) -> PlanState:
    plan = SprintPlan.model_validate(state.get("sprint_plan") or {})
    scope = Scope.model_validate(state.get("scope") or {})
    ops = OpsResult.model_validate(state["ops_result"]) if state.get("ops_result") else None
    return {
        "risks": [r.model_dump() for r in assess(plan, scope, ops)],
        "forecasts": [f.model_dump() for f in forecast_sprints(state)],
    }
