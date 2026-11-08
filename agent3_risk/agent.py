"""Agent 3 (Person C): risk prediction and monitoring."""
from agent3_risk.heuristics import assess
from orchestrator.schemas import OpsResult, Scope, SprintPlan
from orchestrator.state import PlanState


def assess_risk(state: PlanState) -> PlanState:
    # TODO: replace `assess` with the trained model (M5) behind the same signature.
    plan = SprintPlan.model_validate(state.get("sprint_plan") or {})
    scope = Scope.model_validate(state.get("scope") or {})
    ops = OpsResult.model_validate(state["ops_result"]) if state.get("ops_result") else None
    return {"risks": [r.model_dump() for r in assess(plan, scope, ops)]}
