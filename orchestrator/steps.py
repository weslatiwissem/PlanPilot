"""Step-by-step API for the wizard UI. The browser keeps the state; nothing is stored on the server.

    POST /steps/scope     description (+ answers to Agent 1's questions)  -> questions, or the scope
    POST /steps/sprints   the (edited) scope                              -> sprint proposal, tickets, risks, forecasts
    POST /steps/risks     the (edited) tickets                            -> risks + forecasts (call it after every edit)
    POST /steps/approve   the (edited) tickets                            -> cards written to Trello, risks, forecasts

Review flow (the team lead edits EVERY suggestion before anything is written):
    POST /steps/plan-sprints  the (edited) scope    -> STAGE 1: sprints only, no assignees. Lead edits sprints/points.
    POST /steps/assign        the lead's tickets    -> STAGE 2: M4 assigns people inside those sprints. Tickets
                                                       with locked=true keep their person; M4 redistributes the
                                                       rest around them. Lead edits / locks, and may call it again.
    POST /steps/judge         the tickets           -> LLM judge grades every assignment (1-5) and names its pick.
    POST /steps/approve       the final tickets     -> Trello. Send `model_version` from /assign so the audit log
                                                       keeps which model proposed the plan; the response's
                                                       ops_result.review says how much of the proposal was kept.
/steps/sprints (both stages at once) keeps working for scripts and older clients.

Optional on the last three: `past_sprints` (committed/delivered points of finished sprints, oldest first) and
`sprint_days`. With past sprints the forecast uses the history model, without them the current-plan model.

The editable object is the list of tickets (title, points, sprint, assignee). The sprint plan is always derived
from it, so a change made in the UI cannot be lost. When the user picks an assignee themselves, send
`confidence: 1.0` so the weak-match warning goes away. Nothing is written anywhere before /steps/approve, and
approving twice is safe (Agent 2 skips cards that already exist).

Requests and responses are typed, so /openapi.json describes them and the frontend can generate its types.
Errors have ONE shape, the one FastAPI already uses for invalid bodies:
    {"detail": [{"loc": ["body"], "msg": "...", "type": "value_error" | "agent_error"}]}
422 = the request does not make sense (fix it and resend), 502 = an agent or Trello failed.
The old /plan and /approve endpoints in main.py keep working for scripts.
"""
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent1_scoping.agent import ANSWERS_MARKER, scope_project
from agent2_ops.judge import JudgeReport, judge_tickets
from agent2_ops.agent import _load_members, assign_members, plan_sprints, propose_sprints, sprint_capacity
from agent2_ops.planner import edited_tickets, flatten_stories, load_balance, plan_from_tickets, workload
from agent3_risk.agent import assess_risk
from orchestrator.schemas import (
    Forecast,
    Member,
    OpsResult,
    PastSprint,
    Risk,
    Scope,
    SprintPlan,
    Ticket,
    Workload,
)

router = APIRouter(prefix="/steps", tags=["steps"])
DEFAULT_BOARD = "PlanPilot Project"  # same default as Agent 2


# ------------------------------------------------------------------ requests
class Answer(BaseModel):
    question: str
    answer: str


class ScopeStepRequest(BaseModel):
    project_description: str
    clarifications: List[Answer] = Field(default_factory=list)  # answers to the questions returned earlier
    skip_clarification: bool = False  # "no more questions": Agent 1 plans with explicit assumptions
    team_id: Optional[str] = None


class ForecastInputs(BaseModel):
    """Optional inputs of the sprint forecast (agent3_risk/serve.py)."""
    past_sprints: Optional[List[PastSprint]] = None  # finished sprints, oldest first: unlocks the history model
    sprint_days: Optional[int] = Field(default=None, ge=1, le=60)  # length of a sprint in days


class SprintsStepRequest(ForecastInputs):
    scope: Scope
    board_name: Optional[str] = None
    team_members: Optional[List[Member]] = None


class TicketsStepRequest(ForecastInputs):
    scope: Scope
    tickets: List[Ticket]
    board_name: Optional[str] = None
    team_members: Optional[List[Member]] = None
    model_version: Optional[str] = None  # ops_result.model_version from /assign: kept in the audit log


class ApproveStepRequest(TicketsStepRequest):
    board_name: str = DEFAULT_BOARD
    approved_by: str = "unknown"


# ------------------------------------------------------------------ responses
class ScopeStepResponse(BaseModel):
    status: Literal["ok", "needs_clarification"]
    clarifying_questions: List[str] = Field(default_factory=list)
    scope: Optional[Scope] = None


class SprintsStepResponse(BaseModel):
    sprint_plan: SprintPlan
    ops_result: OpsResult
    risks: List[Risk] = Field(default_factory=list)  # rule-based problems with a concrete cause
    forecasts: List[Forecast] = Field(default_factory=list)  # trained model, one per sprint (empty without a model)


class RisksStepResponse(BaseModel):
    sprint_plan: SprintPlan
    risks: List[Risk] = Field(default_factory=list)
    forecasts: List[Forecast] = Field(default_factory=list)
    workload: List[Workload] = Field(default_factory=list)  # % per teammate and sprint, after the edits
    load_balance: Optional[float] = None


class SprintLoad(BaseModel):
    """Stage 1: how full each sprint is (no people yet)."""
    sprint: int
    points: int
    capacity_points: int
    utilisation_pct: int


class SprintStageResponse(BaseModel):
    sprint_plan: SprintPlan
    ops_result: OpsResult                       # tickets with a proposed sprint and no assignee
    sprint_load: List[SprintLoad] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)  # plan checks only (overloaded sprint, big stories)
    forecasts: List[Forecast] = Field(default_factory=list)


# ------------------------------------------------------------------ helpers
def _error(status: int, messages: Union[str, Sequence[str]], kind: str) -> HTTPException:
    items = [messages] if isinstance(messages, str) else list(messages)
    return HTTPException(status_code=status, detail=[{"loc": ["body"], "msg": m, "type": kind} for m in items])


def _members(req_members: Optional[List[Member]]) -> Optional[List[Dict[str, Any]]]:
    return [m.model_dump() for m in req_members] if req_members else None


def _description(req: ScopeStepRequest) -> str:
    """The description Agent 1 expects: answers appended under its marker, so the UI never needs to know it."""
    text = req.project_description.strip()
    lines = [f"- {a.question.strip()}: {a.answer.strip()}" for a in req.clarifications if a.answer.strip()]
    if req.skip_clarification and not lines:
        lines = ["- No further details: make reasonable assumptions."]
    if lines and text:
        if ANSWERS_MARKER not in text:
            text += f"\n\n{ANSWERS_MARKER}"
        text += "\n" + "\n".join(lines)
    return text


def _extra(req: ForecastInputs) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if req.past_sprints:
        out["past_sprints"] = [p.model_dump() for p in req.past_sprints]
    if req.sprint_days:
        out["sprint_days"] = req.sprint_days
    return out


def _assess(state: Dict[str, Any]) -> Dict[str, Any]:
    out = assess_risk(state)
    return {"risks": out.get("risks", []), "forecasts": out.get("forecasts", [])}


def _state_from_tickets(scope_dict, raw_tickets, board_name, members_raw, extra=None) -> Dict[str, Any]:
    """Validate the user's tickets against the scope and build the state Agent 3 reads. 422 if inconsistent."""
    try:
        scope = Scope.model_validate(scope_dict)
        stories = flatten_stories(scope)
        tickets = edited_tickets(raw_tickets, stories, board_name or DEFAULT_BOARD)
        members = _load_members({"team_members": members_raw} if members_raw else {})
    except ValueError as e:  # includes pydantic's ValidationError
        raise _error(422, str(e), "value_error")
    points = {t.title: t.points for t in tickets}
    for epic in scope.epics:  # the tickets are canonical: Agent 3 must see the points the user left
        for story in epic.stories:
            story.estimate_points = points[story.title]
    plan = plan_from_tickets(tickets, sprint_capacity(members))
    rows = workload(tickets, members)
    ops = OpsResult(written=False, model_version="human-edited", tickets=tickets, workload=rows,
                    load_balance=load_balance(rows))
    state: Dict[str, Any] = {
        "scope": scope.model_dump(),
        "sprint_plan": plan.model_dump(),
        "ops_result": ops.model_dump(),
    }
    if members_raw:
        state["team_members"] = members_raw
    state.update(extra or {})
    return state


# ------------------------------------------------------------------ endpoints
@router.post("/scope", response_model=ScopeStepResponse)
def step_scope(req: ScopeStepRequest):
    state: Dict[str, Any] = {"project_description": _description(req)}
    if req.team_id:
        state["team_id"] = req.team_id
    out = scope_project(state)
    if out.get("errors"):
        raise _error(502, out["errors"], "agent_error")
    if out.get("clarifying_questions"):
        return {"status": "needs_clarification", "clarifying_questions": out["clarifying_questions"], "scope": None}
    if not out.get("scope"):
        raise _error(502, "agent1 returned neither a scope nor questions", "agent_error")
    return {"status": "ok", "clarifying_questions": [], "scope": out["scope"]}


@router.post("/sprints", response_model=SprintsStepResponse)
def step_sprints(req: SprintsStepRequest):
    state: Dict[str, Any] = {"scope": req.scope.model_dump()}
    if req.board_name:
        state["board_name"] = req.board_name
    if _members(req.team_members):
        state["team_members"] = _members(req.team_members)
    out = plan_sprints(state)  # proposal only: `approved` is not set
    if out.get("errors"):
        raise _error(422, out["errors"], "value_error")
    state.update(out)
    state.update(_extra(req))
    return {"sprint_plan": state["sprint_plan"], "ops_result": state["ops_result"], **_assess(state)}


@router.post("/risks", response_model=RisksStepResponse)
def step_risks(req: TicketsStepRequest):
    state = _state_from_tickets(req.scope.model_dump(), [t.model_dump() for t in req.tickets],
                                req.board_name, _members(req.team_members), _extra(req))
    ops = state["ops_result"]
    return {"sprint_plan": state["sprint_plan"], "workload": ops["workload"], "load_balance": ops["load_balance"],
            **_assess(state)}


@router.post("/approve", response_model=SprintsStepResponse)
def step_approve(req: ApproveStepRequest):
    raw_tickets, members = [t.model_dump() for t in req.tickets], _members(req.team_members)
    checked = _state_from_tickets(req.scope.model_dump(), raw_tickets, req.board_name, members, _extra(req))  # 422 before any write
    state: Dict[str, Any] = {
        "scope": checked["scope"],
        "tickets": raw_tickets,
        "board_name": req.board_name,
        "approved": True,
        "approved_by": req.approved_by,
    }
    if req.model_version:
        state["model_version"] = req.model_version
    if members:
        state["team_members"] = members
    out = plan_sprints(state)  # keeps the user's tickets; writes the Trello cards
    if out.get("errors"):
        raise _error(502, out["errors"], "agent_error")
    state.update(out)
    state.update(_extra(req))
    return {"sprint_plan": state["sprint_plan"], "ops_result": state["ops_result"], **_assess(state)}


# ------------------------------------------------------------------ review flow (stage 1 -> 2 -> judge)
def _sprint_load(plan: SprintPlan, tickets: List[Dict[str, Any]]) -> List[SprintLoad]:
    points = {t["title"]: t["points"] for t in tickets}
    out = []
    for s in plan.sprints:
        used, cap = sum(points[x] for x in s.items), s.capacity_points or 1
        out.append(SprintLoad(sprint=s.number, points=used, capacity_points=cap, utilisation_pct=round(100 * used / cap)))
    return out


@router.post("/plan-sprints", response_model=SprintStageResponse)
def step_plan_sprints(req: SprintsStepRequest):
    """Stage 1: Agent 2 proposes sprints only. The lead may change any ticket's sprint or points."""
    state: Dict[str, Any] = {"scope": req.scope.model_dump()}
    if req.board_name:
        state["board_name"] = req.board_name
    if _members(req.team_members):
        state["team_members"] = _members(req.team_members)
    out = propose_sprints(state)
    if out.get("errors"):
        raise _error(422, out["errors"], "value_error")
    plan = SprintPlan.model_validate(out["sprint_plan"])
    tickets = out["ops_result"]["tickets"]
    risk_state = {"scope": state["scope"], "sprint_plan": out["sprint_plan"], **_extra(req)}  # no ops: plan checks only
    return {"sprint_plan": out["sprint_plan"], "ops_result": out["ops_result"],
            "sprint_load": _sprint_load(plan, tickets), **_assess(risk_state)}


@router.post("/assign", response_model=SprintsStepResponse)
def step_assign(req: TicketsStepRequest):
    """Stage 2: M4 assigns people inside the lead's sprints, keeping every ticket the lead locked."""
    state: Dict[str, Any] = {"scope": req.scope.model_dump(), "tickets": [t.model_dump() for t in req.tickets]}
    if req.board_name:
        state["board_name"] = req.board_name
    if _members(req.team_members):
        state["team_members"] = _members(req.team_members)
    out = assign_members(state)
    if out.get("errors"):
        raise _error(422, out["errors"], "value_error")
    state.update(out)
    state.update(_extra(req))
    return {"sprint_plan": state["sprint_plan"], "ops_result": state["ops_result"], **_assess(state)}


@router.post("/judge", response_model=JudgeReport)
def step_judge(req: TicketsStepRequest):
    """LLM-as-a-judge on the current tickets (needs GEMINI_API_KEY). Unassigned tickets are skipped."""
    state = _state_from_tickets(req.scope.model_dump(), [t.model_dump() for t in req.tickets],
                                req.board_name, _members(req.team_members))
    tickets = [Ticket.model_validate(t) for t in state["ops_result"]["tickets"]]
    stories = flatten_stories(Scope.model_validate(state["scope"]))
    members = _load_members(state)
    try:
        report = judge_tickets(tickets, stories, members)
    except Exception as e:  # key missing, quota, network
        raise _error(502, f"judge failed: {e}", "agent_error")
    return report
