"""Shared state passed between the three agents."""
from typing import Any, Dict, List, TypedDict


class PlanState(TypedDict, total=False):
    project_description: str      # input from the user
    scope: Dict[str, Any]         # Agent 1 output: epics, stories, estimates
    sprint_plan: Dict[str, Any]   # Agent 2 output: sprints, assignments, tickets
    risks: List[Dict[str, Any]]   # Agent 3 output: risk flags and scores
    approved: bool                # human approval gate
    errors: List[str]
