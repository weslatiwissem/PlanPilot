"""Data contracts between agents. Agree on these as a team before changing them.

Agents pass plain dicts through LangGraph state (see state.py). Each agent should
validate what it receives and emit `.model_dump()` of what it produces:

    scope = Scope.model_validate(state["scope"])
    return {"sprint_plan": SprintPlan(...).model_dump()}

CHANGE vs. previous version: Scope / Sprint / SprintPlan / Risk are UNCHANGED.
Only Member, Ticket and OpsResult were ADDED (used by Agent 2, readable by Agent 3).
Review flow (team lead edits every AI suggestion): Ticket got proposed_sprint / proposed_assignee / locked,
OpsResult got workload and review. All new fields have defaults, so older clients keep working.
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
    # --- review flow: what the AI suggested, so the lead's changes can be measured and learned from ---
    proposed_sprint: Optional[int] = None     # sprint Agent 2 suggested in stage 1 (never overwritten)
    proposed_assignee: Optional[str] = None   # assignee M4 suggested in stage 2 (never overwritten)
    locked: bool = False                      # the lead fixed this assignee: M4 must keep it when re-assigning


class Workload(BaseModel):
    """How much one teammate takes in one sprint (shown as a bar per person in the UI)."""
    sprint: int
    member: str                       # "Unassigned" collects the stories nobody holds yet
    points: int                       # story points this person holds in this sprint
    capacity_points: int              # what this person can still take (capacity - current load)
    utilisation_pct: int              # (points + current load) / capacity: above 100 = overloaded
    share_pct: int                    # share of this sprint's points: who carries the sprint


class ReviewStats(BaseModel):
    """How much the team lead kept of the AI's suggestions (a human-agreement metric for Agent 2)."""
    tickets: int = 0
    sprints_changed: int = 0          # stories the lead moved to another sprint than proposed
    assignees_changed: int = 0        # stories the lead gave to someone else than M4 proposed
    sprint_acceptance_rate: Optional[float] = None    # 1.0 = every proposed sprint was kept
    assignee_acceptance_rate: Optional[float] = None  # 1.0 = every proposed assignee was kept


class OpsResult(BaseModel):
    """Agent 2 output (what was proposed / written). Agent 3 reads this."""
    written: bool = False             # True only after human approval + Trello write
    board_id: Optional[str] = None
    board_url: Optional[str] = None
    model_version: str = ""           # which skill matcher produced the assignments
    tickets: List[Ticket] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    workload: List[Workload] = Field(default_factory=list)   # % per teammate and sprint
    load_balance: Optional[float] = None   # least / most utilised teammate, averaged over sprints (1.0 = even)
    review: Optional[ReviewStats] = None   # filled when the lead approves an edited plan


class Risk(BaseModel):
    """Agent 3 output (list of these)."""
    item: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class PastSprint(BaseModel):
    """A sprint the team has already finished (input of the sprint forecast)."""
    committed: float = Field(ge=0)    # story points the team committed to
    delivered: float = Field(ge=0)    # story points it actually finished


class Forecast(BaseModel):
    """Agent 3 output: chance that a planned sprint delivers less than it commits (trained model)."""
    sprint: int = Field(ge=1)
    probability: float = Field(ge=0.0, le=1.0)
    reference_rate: Optional[float] = None   # share of sprints that did so in the training data
    model: str                               # "B1_plus" (no history) | "B2_plus" (past sprints given)
    history_used: int = 0                    # how many past sprints the model saw
