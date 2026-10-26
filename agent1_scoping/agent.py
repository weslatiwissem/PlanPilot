"""Agent 1 (Person A): scoping, RAG, clarification dialog."""
from orchestrator.state import PlanState


def scope_project(state: PlanState) -> PlanState:
    # TODO: replace stub with M1/M2/M3 + RAG
    desc = state.get("project_description", "")
    return {"scope": {"epics": [{"title": "Placeholder epic", "source": desc[:80]}]}}
