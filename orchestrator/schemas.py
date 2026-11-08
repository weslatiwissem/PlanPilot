"""Data contracts between agents. Agree on these as a team before changing them.

Agents pass plain dicts through LangGraph state (see state.py). Each agent should
validate what it receives and emit `.model_dump()` of what it produces:

    scope = Scope.model_validate(state["scope"])
    return {"sprint_plan": SprintPlan(...).model_dump()}

CHANGE vs. previous version: Scope / Sprint / SprintPlan / Risk are UNCHANGED.
Only Member, Ticket and OpsResult were ADDED (used by Agent 2, readable by Agent 3).
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
    """Agent 2 output (the plan itself)."""
    sprints: List[Sprint] = Field(default_factory=list)


# ---------------------------------------------------------------- Agent 2 additions
class Member(BaseModel):
    """A team member, used by the skill matcher (M4)."""
    name: str
    skills: List[str] = Field(default_factory=list)
    capacity_points: int = Field(default=10, ge=1)   # story points per sprint
    current_load: int = Field(default=0, ge=0)       # points already committed per sprint
    trello_username: Optional[str] = None            # optional: real Trello assignment


class Ticket(BaseModel):
    """One story as it becomes a Trello card."""
    key: str                          # idempotency key = hash(project id + story title)
    title: str
    epic: str
    points: int
    sprint: int
    assignee: Optional[str] = None    # None = nobody confident enough -> human decides
    confidence: float = 0.0           # similarity score of the chosen assignee (0..1)
    needs_human: bool = False         # low confidence -> routed to a human
    status: str = "proposed"          # proposed | created | already_exists
    card_id: Optional[str] = None
    url: Optional[str] = None


class OpsResult(BaseModel):
    """Agent 2 output (what was proposed / written). Agent 3 reads this."""
    written: bool = False             # True only after human approval + Trello write
    board_id: Optional[str] = None
    board_url: Optional[str] = None
    model_version: str = ""           # which skill matcher produced the assignments
    tickets: List[Ticket] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Risk(BaseModel):
    """Agent 3 output (list of these)."""
    item: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
