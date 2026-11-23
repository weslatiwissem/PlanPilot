"""Sprint packing: Scope (from Agent 1) -> list of stories grouped into sprints.

Deliberately simple and deterministic: stories are taken in the order Agent 1
produced them (epics are MVP-first) and added to the current sprint until the
sprint's point capacity is full, then a new sprint starts.

Also here (pure functions, no I/O): the workload % per teammate, the load-balance index and the
review statistics (how much of the AI's proposal the team lead kept).
"""
import hashlib
from typing import Dict, List, Optional

from orchestrator.schemas import Member, ReviewStats, Scope, Sprint, SprintPlan, Ticket, Workload

DEFAULT_POINTS = 3  # used if a story has no estimate


def flatten_stories(scope: Scope) -> List[Dict]:
    """Scope -> flat list of dicts: title, description, epic, points."""
    stories = []
    for epic in scope.epics:
        for s in epic.stories:
            stories.append(
                {
                    "title": s.title,
                    "description": s.description,
                    "epic": epic.title,
                    "points": s.estimate_points or DEFAULT_POINTS,
                }
            )
    return stories


def pack_sprints(stories: List[Dict], capacity: int):
    """Greedy fill. Returns (SprintPlan, {story title: sprint number})."""
    sprints, sprint_of = [], {}
    items, used = [], 0
    for s in stories:
        # open a new sprint if this story does not fit (a too-big story gets its own)
        if items and used + s["points"] > capacity:
            sprints.append(Sprint(number=len(sprints) + 1, items=items, capacity_points=capacity))
            items, used = [], 0
        items.append(s["title"])
        used += s["points"]
        sprint_of[s["title"]] = len(sprints) + 1
    if items:
        sprints.append(Sprint(number=len(sprints) + 1, items=items, capacity_points=capacity))
    return SprintPlan(sprints=sprints), sprint_of


def ticket_key(project_id: str, story_title: str) -> str:
    """Idempotency key: same project + same story => same key => no duplicate cards."""
    return hashlib.sha1(f"{project_id}|{story_title}".encode("utf-8")).hexdigest()[:16]


def check_tickets(tickets: List[Ticket], stories: List[Dict]) -> None:
    """Edited tickets must match the scope one to one. Raises ValueError with a message for the user."""
    titles = [t.title for t in tickets]
    duplicated = sorted({x for x in titles if titles.count(x) > 1})
    if duplicated:
        raise ValueError(f"more than one ticket for: {', '.join(duplicated)}")
    expected = [s["title"] for s in stories]
    missing = [x for x in expected if x not in titles]
    unknown = [x for x in titles if x not in expected]
    if missing:
        raise ValueError(f"no ticket for: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"ticket does not match any story: {', '.join(unknown)}")
    bad = [t.title for t in tickets if t.sprint < 1 or t.points < 1]
    if bad:
        raise ValueError(f"sprint and points must be at least 1: {', '.join(bad)}")


def edited_tickets(raw: List[Dict], stories: List[Dict], project_id: str) -> List[Ticket]:
    """Tickets as the user left them (sprint, points, assignee), made consistent with the scope.

    Keys, epics and states are recomputed here and never trusted from the client. A ticket needs a human
    exactly when nobody is assigned.
    """
    tickets = [Ticket.model_validate(t) for t in raw]
    check_tickets(tickets, stories)
    epic_of = {s["title"]: s["epic"] for s in stories}
    order = {s["title"]: i for i, s in enumerate(stories)}
    for t in tickets:
        t.key = ticket_key(project_id, t.title)
        t.epic = epic_of[t.title]
        t.needs_human = t.assignee is None
        t.status, t.card_id, t.url = "proposed", None, None
    return sorted(tickets, key=lambda t: order[t.title])


def plan_from_tickets(tickets: List[Ticket], capacity: int) -> SprintPlan:
    """The plan implied by the tickets: one sprint per sprint number used, stories in scope order."""
    by_sprint: Dict[int, List[str]] = {}
    for t in tickets:
        by_sprint.setdefault(t.sprint, []).append(t.title)
    return SprintPlan(
        sprints=[Sprint(number=n, items=items, capacity_points=capacity) for n, items in sorted(by_sprint.items())]
    )


def sprint_tickets(stories: List[Dict], capacity: int, project_id: str) -> List[Ticket]:
    """Review stage 1: tickets with a proposed sprint and NO assignee yet (the lead edits the sprints first)."""
    _, sprint_of = pack_sprints(stories, capacity)
    return [
        Ticket(key=ticket_key(project_id, s["title"]), title=s["title"], epic=s["epic"], points=s["points"],
               sprint=sprint_of[s["title"]], proposed_sprint=sprint_of[s["title"]], needs_human=True)
        for s in stories
    ]


UNASSIGNED = "Unassigned"


def workload(tickets: List[Ticket], members: List[Member]) -> List[Workload]:
    """Per sprint and teammate: points held, % of their capacity used, % of the sprint they carry."""
    rows: List[Workload] = []
    for n in sorted({t.sprint for t in tickets}):
        in_sprint = [t for t in tickets if t.sprint == n]
        total = sum(t.points for t in in_sprint) or 1
        for m in members:
            pts = sum(t.points for t in in_sprint if t.assignee == m.name)
            rows.append(Workload(
                sprint=n, member=m.name, points=pts,
                capacity_points=max(m.capacity_points - m.current_load, 0),
                utilisation_pct=round(100 * (pts + m.current_load) / m.capacity_points),
                share_pct=round(100 * pts / total),
            ))
        known = {m.name for m in members}
        free = sum(t.points for t in in_sprint if t.assignee not in known)  # nobody, or a name not in the team
        if free:
            rows.append(Workload(sprint=n, member=UNASSIGNED, points=free, capacity_points=0,
                                 utilisation_pct=0, share_pct=round(100 * free / total)))
    return rows


def load_balance(rows: List[Workload]) -> Optional[float]:
    """Least-utilised / most-utilised teammate in each sprint, averaged. 1.0 = perfectly even, 0 = someone idle."""
    scores = []
    for n in sorted({r.sprint for r in rows}):
        used = [r.utilisation_pct for r in rows if r.sprint == n and r.member != UNASSIGNED]
        if used and max(used) > 0:
            scores.append(min(used) / max(used))
    return round(sum(scores) / len(scores), 3) if scores else None


def review_stats(tickets: List[Ticket]) -> ReviewStats:
    """Compare the lead's final tickets with what the AI proposed (only tickets that carry a proposal count)."""
    with_sprint = [t for t in tickets if t.proposed_sprint is not None]
    with_assignee = [t for t in tickets if t.proposed_assignee is not None]
    moved = sum(t.sprint != t.proposed_sprint for t in with_sprint)
    changed = sum(t.assignee != t.proposed_assignee for t in with_assignee)
    return ReviewStats(
        tickets=len(tickets), sprints_changed=moved, assignees_changed=changed,
        sprint_acceptance_rate=round(1 - moved / len(with_sprint), 3) if with_sprint else None,
        assignee_acceptance_rate=round(1 - changed / len(with_assignee), 3) if with_assignee else None,
    )
