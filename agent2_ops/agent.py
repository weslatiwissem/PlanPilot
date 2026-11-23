"""Agent 2 (Person B): sprint planning + assignment + Trello connector.

Runs AFTER Agent 1 in the LangGraph pipeline (orchestrator/graph.py).

    state["scope"]  --->  plan sprints  --->  match members (M4)  --->  [approval gate]
                                                                          |
                                                                          +--> write Trello cards

Two modes, controlled ONLY by state["approved"]:
  * approved missing/False (POST /plan)    -> PROPOSAL: nothing is written anywhere.
  * approved is True       (POST /approve) -> the same plan is written to Trello.
The approval gate is "hard": write_to_trello() itself refuses to run without approval.

Review flow for the team lead (used by the /steps API, every AI suggestion can be edited):
  1. propose_sprints()  AI packs stories into sprints, no assignees      -> lead edits sprints / points
  2. assign_members()   M4 assigns INSIDE the lead's sprints; tickets the
                        lead marked `locked` keep their person and count
                        as load, M4 redistributes the rest around them   -> lead edits / locks, may re-run 2
  3. plan_sprints(approved=True) writes exactly the reviewed tickets to Trello and logs every change
     the lead made (proposed vs final) to the audit log: the feedback for retraining M4.

Output in state:
  sprint_plan -> schemas.SprintPlan   (same contract as before)
  ops_result  -> schemas.OpsResult    (tickets, assignees, board id/url; Agent 3 reads this)
"""
import json
import logging
import os
from typing import Dict, List, Optional

from agent2_ops.audit import audit
from agent2_ops.matcher import assign
from agent2_ops.planner import (
    edited_tickets,
    flatten_stories,
    load_balance,
    pack_sprints,
    plan_from_tickets,
    review_stats,
    sprint_tickets,
    ticket_key,
    workload,
)
from agent2_ops.trello import TrelloClient
from orchestrator.schemas import Member, OpsResult, Scope, Ticket
from orchestrator.state import PlanState

log = logging.getLogger(__name__)
TEAM_FILE = os.path.join(os.path.dirname(__file__), "team.json")


def _load_members(state: PlanState) -> List[Member]:
    """Team from the request if given, otherwise from agent2_ops/team.json."""
    raw = state.get("team_members")
    if not raw:
        with open(TEAM_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    return [Member.model_validate(m) for m in raw]


def sprint_capacity(members: List[Member]) -> int:
    """Points per sprint: the SPRINT_CAPACITY override, else the team's combined capacity."""
    return max(int(os.getenv("SPRINT_CAPACITY", sum(m.capacity_points for m in members))), 1)


def _card_text(t: Ticket, description: str) -> str:
    who = t.assignee or "NEEDS HUMAN ASSIGNMENT (low confidence)"
    return (
        f"{description}\n\nEpic: {t.epic}\nStory points: {t.points}\nAssignee: {who}"
        f"\n\n---\nplanpilot-key: {t.key}"  # <- idempotency marker, do not edit
    )


def _log_review(tickets: List[Ticket], review, who: str, version: str) -> None:
    """Audit every change the lead made to the AI's proposal (the feedback M4 can be retrained on)."""
    for t in tickets:
        if t.proposed_sprint is not None and t.sprint != t.proposed_sprint:
            audit("override_sprint", who, t.key, version, title=t.title, proposed=t.proposed_sprint, final=t.sprint)
        if t.proposed_assignee is not None and t.assignee != t.proposed_assignee:
            audit("override_assignee", who, t.key, version, title=t.title,
                  proposed=t.proposed_assignee, final=t.assignee)
    audit("review_summary", who, "plan", version, **review.model_dump())


def write_to_trello(
    tickets: List[Ticket],
    descriptions: Dict[str, str],
    members: List[Member],
    board_name: str,
    approved: bool,
    approved_by: str,
    model_version: str,
    client: Optional[TrelloClient] = None,
) -> OpsResult:
    """Create board / sprint lists / cards. Safe to call again: existing cards are skipped."""
    if approved is not True:  # the hard approval gate
        raise PermissionError("Agent 2 may not write to Trello without an explicit approval.")

    client = client or TrelloClient.from_env()
    result = OpsResult(written=True, model_version=model_version, tickets=tickets)

    board = client.get_or_create_board(board_name)
    result.board_id, result.board_url = board["id"], board.get("url")
    audit("get_or_create_board", approved_by, board_name, model_version, board_id=board["id"])

    # one list per sprint
    list_ids = {}
    for n in sorted({t.sprint for t in tickets}):
        list_ids[n] = client.ensure_list(board["id"], f"Sprint {n}")
        audit("ensure_list", approved_by, f"Sprint {n}", model_version)

    # Trello user ids for members that have a trello_username (optional)
    member_ids = {}
    for m in members:
        if m.trello_username:
            mid = client.member_id_on_board(m.trello_username, board["id"])
            if mid:
                member_ids[m.name] = mid
            else:
                result.warnings.append(f"Could not resolve Trello user '{m.trello_username}'.")

    existing = client.existing_cards(board["id"])
    for t in tickets:
        if t.key in existing:  # idempotency: already created on an earlier run
            card = existing[t.key]
            t.status, t.card_id, t.url = "already_exists", card["id"], card.get("shortUrl")
            audit("create_card", approved_by, t.key, model_version, status="skipped_duplicate")
            continue
        ids = [member_ids[t.assignee]] if t.assignee in member_ids else []
        card = client.create_card(
            list_ids[t.sprint], f"{t.title} [{t.points} pts]", _card_text(t, descriptions[t.title]), ids
        )
        t.status, t.card_id, t.url = "created", card["id"], card.get("shortUrl")
        audit("create_card", approved_by, t.key, model_version, title=t.title, assignee=t.assignee)
    return result


def assign_tickets(tickets: List[Ticket], stories: List[Dict], members: List[Member]) -> str:
    """Fill assignee / proposed_assignee / confidence, sprint by sprint, around the lead's locked tickets.

    Locked tickets keep their person; their points are added to that person's load for the sprint, so the
    Hungarian step spreads the remaining stories over whoever still has room. Returns the matcher version.
    """
    by_title = {s["title"]: s for s in stories}
    known = {m.name for m in members}
    bad = sorted({t.assignee for t in tickets if t.locked and t.assignee not in known})
    if bad:
        raise ValueError(f"locked tickets name people who are not in the team: {', '.join(map(str, bad))}")
    version = "none"
    for n in sorted({t.sprint for t in tickets}):
        in_sprint = [t for t in tickets if t.sprint == n]
        fixed = [t for t in in_sprint if t.locked and t.assignee]
        free = [t for t in in_sprint if not (t.locked and t.assignee)]
        held = {m.name: sum(t.points for t in fixed if t.assignee == m.name) for m in members}
        team = [m.model_copy(update={"current_load": m.current_load + held[m.name]}) for m in members]
        for t in fixed:
            t.confidence, t.needs_human = 1.0, False  # a human chose this person
        if not free:
            continue
        stories_now = [{**by_title[t.title], "points": t.points} for t in free]  # the lead's points count
        part, version = assign(stories_now, team)
        for t in free:
            who, conf = part[t.title]
            t.assignee = t.proposed_assignee = who
            t.confidence, t.needs_human = round(conf, 3), who is None
    return version


def _summary(tickets: List[Ticket], members: List[Member], **fields) -> OpsResult:
    rows = workload(tickets, members)
    return OpsResult(tickets=tickets, workload=rows, load_balance=load_balance(rows), **fields)


def propose_sprints(state: PlanState) -> PlanState:
    """Review stage 1: sprints only (no assignees). Never writes anything. Problems go to state["errors"]."""
    try:
        scope = Scope.model_validate(state.get("scope") or {})
        stories = flatten_stories(scope)
        if not stories:
            raise ValueError("scope has no stories to plan")
        members = _load_members(state)
        capacity = sprint_capacity(members)
        tickets = sprint_tickets(stories, capacity, state.get("board_name") or "PlanPilot Project")
        return {
            "sprint_plan": plan_from_tickets(tickets, capacity).model_dump(),
            "ops_result": OpsResult(written=False, model_version="sprint-packer", tickets=tickets).model_dump(),
        }
    except Exception as e:
        log.exception("agent2 propose_sprints failed")
        return {"errors": [f"agent2: {e}"]}


def assign_members(state: PlanState) -> PlanState:
    """Review stage 2: M4 assigns people inside the lead's (edited) sprints. Never writes anything."""
    try:
        scope = Scope.model_validate(state.get("scope") or {})
        stories = flatten_stories(scope)
        members = _load_members(state)
        project_id = state.get("board_name") or "PlanPilot Project"
        tickets = edited_tickets(state.get("tickets") or [], stories, project_id)  # validates against the scope
        version = assign_tickets(tickets, stories, members)
        ops = _summary(tickets, members, written=False, model_version=version)
        return {
            "sprint_plan": plan_from_tickets(tickets, sprint_capacity(members)).model_dump(),
            "ops_result": ops.model_dump(),
        }
    except Exception as e:
        log.exception("agent2 assign_members failed")
        return {"errors": [f"agent2: {e}"]}


def plan_sprints(state: PlanState) -> PlanState:
    """LangGraph node. Never raises: problems are returned in state["errors"]."""
    try:
        scope = Scope.model_validate(state.get("scope") or {})  # validate Agent 1's output
        stories = flatten_stories(scope)
        if not stories:
            raise ValueError("scope has no stories to plan")
        members = _load_members(state)

        project_id = state.get("board_name") or "PlanPilot Project"
        capacity = sprint_capacity(members)

        if state.get("tickets"):
            # a plan edited by the user (step API): keep sprints, points and assignees exactly as left
            tickets = edited_tickets(state["tickets"], stories, project_id)
            plan = plan_from_tickets(tickets, capacity)
            # keep WHICH model made the proposal (the UI sends it back), and say that a human edited it
            base = state.get("model_version")
            version = f"{base} (human-edited)" if base else "human-edited"
        else:
            # 1) sprints: greedy fill up to the capacity, 2) assignment (M4), balanced per sprint
            plan, sprint_of = pack_sprints(stories, capacity)
            tickets = [
                Ticket(key=ticket_key(project_id, s["title"]), title=s["title"], epic=s["epic"],
                       points=s["points"], sprint=sprint_of[s["title"]], proposed_sprint=sprint_of[s["title"]])
                for s in stories
            ]
            version = assign_tickets(tickets, stories, members)
        out = {"sprint_plan": plan.model_dump()}

        # 4) approval gate
        if state.get("approved") is True:
            review = review_stats(tickets)
            _log_review(tickets, review, state.get("approved_by", "unknown"), version)
            ops = write_to_trello(
                tickets,
                {s["title"]: s["description"] for s in stories},
                members,
                project_id,
                True,
                state.get("approved_by", "unknown"),
                version,
            )
            ops.review = review
        else:
            ops = OpsResult(written=False, model_version=version, tickets=tickets)
        ops.workload = workload(tickets, members)
        ops.load_balance = load_balance(ops.workload)
        out["ops_result"] = ops.model_dump()
        return out
    except Exception as e:  # surface the problem instead of crashing the whole graph
        log.exception("agent2 failed")
        return {"errors": [f"agent2: {e}"]}
