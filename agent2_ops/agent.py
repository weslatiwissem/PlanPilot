"""Agent 2 (Person B): sprint planning + assignment + Trello connector.

Runs AFTER Agent 1 in the LangGraph pipeline (orchestrator/graph.py).

    state["scope"]  --->  plan sprints  --->  match members (M4)  --->  [approval gate]
                                                                          |
                                                                          +--> write Trello cards

Two modes, controlled ONLY by state["approved"]:
  * approved missing/False (POST /plan)    -> PROPOSAL: nothing is written anywhere.
  * approved is True       (POST /approve) -> the same plan is written to Trello.
The approval gate is "hard": write_to_trello() itself refuses to run without approval.

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
from agent2_ops.planner import flatten_stories, pack_sprints, ticket_key
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


def _card_text(t: Ticket, description: str) -> str:
    who = t.assignee or "NEEDS HUMAN ASSIGNMENT (low confidence)"
    return (
        f"{description}\n\nEpic: {t.epic}\nStory points: {t.points}\nAssignee: {who}"
        f"\n\n---\nplanpilot-key: {t.key}"  # <- idempotency marker, do not edit
    )


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


def plan_sprints(state: PlanState) -> PlanState:
    """LangGraph node. Never raises: problems are returned in state["errors"]."""
    try:
        scope = Scope.model_validate(state.get("scope") or {})  # validate Agent 1's output
        stories = flatten_stories(scope)
        if not stories:
            raise ValueError("scope has no stories to plan")
        members = _load_members(state)

        # 1) sprints: capacity = env override, else the team's combined capacity
        capacity = int(os.getenv("SPRINT_CAPACITY", sum(m.capacity_points for m in members)))
        capacity = max(capacity, 1)
        plan, sprint_of = pack_sprints(stories, capacity)

        # 2) assignment (M4), balanced per sprint
        by_title = {s["title"]: s for s in stories}
        assignments, version = {}, "none"
        for sprint in plan.sprints:
            part, version = assign([by_title[t] for t in sprint.items], members)
            assignments.update(part)

        # 3) tickets (what will become Trello cards)
        project_id = state.get("board_name") or "PlanPilot Project"
        tickets = []
        for s in stories:
            who, conf = assignments[s["title"]]
            tickets.append(
                Ticket(
                    key=ticket_key(project_id, s["title"]),
                    title=s["title"],
                    epic=s["epic"],
                    points=s["points"],
                    sprint=sprint_of[s["title"]],
                    assignee=who,
                    confidence=round(conf, 3),
                    needs_human=who is None,
                )
            )
        out = {"sprint_plan": plan.model_dump()}

        # 4) approval gate
        if state.get("approved") is True:
            ops = write_to_trello(
                tickets,
                {s["title"]: s["description"] for s in stories},
                members,
                project_id,
                True,
                state.get("approved_by", "unknown"),
                version,
            )
        else:
            ops = OpsResult(written=False, model_version=version, tickets=tickets)
        out["ops_result"] = ops.model_dump()
        return out
    except Exception as e:  # surface the problem instead of crashing the whole graph
        log.exception("agent2 failed")
        return {"errors": [f"agent2: {e}"]}
