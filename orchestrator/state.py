"""Shared state passed between the three agents (values are dicts; see schemas.py)."""
from typing import Any, Dict, List, TypedDict


class PlanState(TypedDict, total=False):
    project_description: str          # input from the user
    clarifying_questions: List[str]   # Agent 1: if non-empty, the pipeline stops and asks
    scope: Dict[str, Any]             # Agent 1 output  -> schemas.Scope
    sprint_plan: Dict[str, Any]       # Agent 2 output  -> schemas.SprintPlan
    risks: List[Dict[str, Any]]       # Agent 3 output  -> list[schemas.Risk]
    approved: bool                    # human approval gate (only /approve sets it True)
    errors: List[str]
    team_id: str
    # --- added for Agent 2 -------------------------------------------------------
    team_members: List[Dict[str, Any]]  # optional input -> list[schemas.Member]
    board_name: str                     # optional input: Trello board to create/reuse
    approved_by: str                    # who clicked approve (goes into the audit log)
    ops_result: Dict[str, Any]          # Agent 2 output -> schemas.OpsResult
