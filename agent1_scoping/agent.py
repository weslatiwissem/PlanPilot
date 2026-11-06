"""Agent 1 (Person A): scoping, RAG, clarification dialog.

Clarification loop (stateless API): when questions come back, the client re-sends
/plan with the answers appended to the description under ANSWERS_MARKER. Once that
marker is present Agent 1 never asks again; it plans with explicit assumptions.

Optional state["team_id"]: if the trained estimator exists and knows the team, story
points are rescaled to that team's own history (see estimator.py); otherwise Gemini's
estimates are kept.
"""
import logging

from agent1_scoping.clarify import check_missing_info
from agent1_scoping.rag import retrieve_similar
from agent1_scoping.scope import generate_scope
from orchestrator.state import PlanState

log = logging.getLogger(__name__)
ANSWERS_MARKER = "## Clarifications"


def scope_project(state: PlanState) -> PlanState:
    desc = state.get("project_description", "").strip()
    if not desc:
        return {"clarifying_questions": ["Please describe the project you want to plan."]}

    try:
        if ANSWERS_MARKER not in desc:
            check = check_missing_info(desc)
            if check.questions:
                return {"clarifying_questions": check.questions}

        scope = generate_scope(desc, retrieve_similar(desc))

        team_id = state.get("team_id")
        if team_id:  # optional: rescale estimates to this team's own history (M3)
            try:
                from agent1_scoping.estimator import calibrate_scope  # lazy: heavy imports

                scope = calibrate_scope(scope, team_id)
            except Exception as e:  # keep Gemini's estimates rather than failing the plan
                log.warning("estimate calibration skipped for team %r: %s", team_id, e)
        return {"scope": scope.model_dump(), "clarifying_questions": []}
    except Exception as e:  # surface the problem instead of crashing the whole graph
        return {"errors": [f"agent1: {e}"], "clarifying_questions": []}
