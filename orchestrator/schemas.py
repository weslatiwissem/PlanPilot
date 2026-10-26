"""Data contracts between agents. Agree on these as a team before changing them.

Agents pass plain dicts through LangGraph state (see state.py). Each agent should
validate what it receives and emit `.model_dump()` of what it produces:

    scope = Scope.model_validate(state["scope"])
    return {"sprint_plan": SprintPlan(...).model_dump()}
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class Story(BaseModel):
    title: str
    description: str = ""
    estimate_points: Optional[int] = Field(default=None, ge=1)


class Epic(BaseModel):
    title: str
    stories: List[Story] = Field(default_factory=list)


class Scope(BaseModel):
    """Agent 1 output."""
    epics: List[Epic] = Field(default_factory=list)


class Sprint(BaseModel):
    number: int = Field(ge=1)
    items: List[str] = Field(default_factory=list)  # story titles
    capacity_points: Optional[int] = None


class SprintPlan(BaseModel):
    """Agent 2 output."""
    sprints: List[Sprint] = Field(default_factory=list)


class Risk(BaseModel):
    """Agent 3 output (list of these)."""
    item: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
