"""Sprint packing: Scope (from Agent 1) -> list of stories grouped into sprints.

Deliberately simple and deterministic: stories are taken in the order Agent 1
produced them (epics are MVP-first) and added to the current sprint until the
sprint's point capacity is full, then a new sprint starts.
"""
import hashlib
from typing import Dict, List

from orchestrator.schemas import Scope, Sprint, SprintPlan

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
