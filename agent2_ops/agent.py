"""Agent 2 (Person B): sprint planning + Jira/Trello connectors."""
from orchestrator.state import PlanState


def plan_sprints(state: PlanState) -> PlanState:
    # TODO: replace stub with M4 + connectors
    epics = state.get("scope", {}).get("epics", [])
    return {"sprint_plan": {"sprints": [{"number": 1, "items": [e["title"] for e in epics]}]}}
