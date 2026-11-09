"""Rule-based risk checks (baseline for M5).

Same contract the trained model will use later:
    assess(plan, scope, ops=None) -> list[Risk]   (sorted, highest score first)
Scores are 0..1; higher = riskier. `ops` is Agent 2's OpsResult (tickets with assignee
and match confidence); without it only the plan/scope checks run.
"""
from collections import defaultdict
from typing import List, Optional

from orchestrator.schemas import OpsResult, Risk, Scope, SprintPlan

MAX_ITEMS_NO_CAPACITY = 8    # items in a sprint before it counts as overloaded (no capacity set)
LARGE_STORY_POINTS = 8       # stories at/above this should probably be split
TIGHT_UTILISATION = 0.9      # planned/capacity at or above this = no slack
LOW_CONFIDENCE = 0.2         # Agent 2 match confidence below this = weak assignment
CONCENTRATION_SHARE = 0.5    # one person holding more than this share of a sprint's points
MIN_POINTS_FOR_CONCENTRATION = 10  # ignore tiny sprints: someone always holds most of 7 points


def _points_by_title(scope: Scope) -> dict:
    return {s.title: s.estimate_points for e in scope.epics for s in e.stories}


def _sprint_risks(plan: SprintPlan, scope: Scope) -> List[Risk]:
    points = _points_by_title(scope)
    risks: List[Risk] = []

    for sprint in plan.sprints:
        label = f"Sprint {sprint.number}"
        if not sprint.items:
            risks.append(Risk(item=label, score=0.3, reason="Sprint has no items."))
            continue

        estimates = [points.get(title) for title in sprint.items]
        total = sum(p for p in estimates if p is not None)

        if sprint.capacity_points:
            ratio = total / sprint.capacity_points
            if ratio > 1:
                risks.append(Risk(
                    item=label, score=round(min(1.0, 0.5 + (ratio - 1)), 2),
                    reason=f"Planned {total} points against a capacity of {sprint.capacity_points}.",
                ))
            elif ratio >= TIGHT_UTILISATION:
                risks.append(Risk(
                    item=label, score=round(min(0.5, 0.3 + (ratio - TIGHT_UTILISATION) * 2), 2),
                    reason=(f"{total} of {sprint.capacity_points} points planned ({ratio:.0%}); "
                            "no slack for surprises."),
                ))
        elif len(sprint.items) > MAX_ITEMS_NO_CAPACITY:
            risks.append(Risk(
                item=label, score=0.6,
                reason=f"{len(sprint.items)} items and no capacity set; likely overloaded.",
            ))

        missing = sum(1 for p in estimates if p is None)
        if missing:
            share = missing / len(sprint.items)
            risks.append(Risk(
                item=label, score=round(0.2 + 0.4 * share, 2),
                reason=f"{missing} of {len(sprint.items)} items have no estimate.",
            ))

        for title, p in zip(sprint.items, estimates):
            if p is not None and p >= LARGE_STORY_POINTS:
                risks.append(Risk(item=title, score=0.5, reason=f"{p} points; consider splitting."))
    return risks


def _ops_risks(ops: OpsResult) -> List[Risk]:
    risks: List[Risk] = []

    for t in ops.tickets:
        big = t.points >= 5
        if t.assignee is None or t.needs_human:
            risks.append(Risk(
                item=t.title, score=0.6 if big else 0.5,
                reason=f"No confident assignee (match confidence {t.confidence:.2f}); "
                       "a human must pick an owner.",
            ))
        elif t.confidence < LOW_CONFIDENCE:
            risks.append(Risk(
                item=t.title, score=0.4 if big else 0.3,
                reason=f"Weak skill match for {t.assignee} (confidence {t.confidence:.2f}).",
            ))

    load = defaultdict(lambda: defaultdict(int))  # sprint -> person -> points
    for t in ops.tickets:
        if t.assignee:
            load[t.sprint][t.assignee] += t.points
    for sprint, people in load.items():
        total = sum(people.values())
        person, pts = max(people.items(), key=lambda kv: kv[1])
        if total >= MIN_POINTS_FOR_CONCENTRATION and pts / total > CONCENTRATION_SHARE:
            risks.append(Risk(
                item=f"Sprint {sprint}", score=0.4,
                reason=f"{person} holds {pts / total:.0%} of the assigned points; "
                       "the sprint depends on one person.",
            ))

    for w in ops.warnings:
        risks.append(Risk(item="Agent 2 warning", score=0.4, reason=w))
    return risks


def assess(plan: SprintPlan, scope: Scope, ops: Optional[OpsResult] = None) -> List[Risk]:
    if not plan.sprints:
        return [Risk(item="plan", score=0.9, reason="No sprints were planned.")]
    risks = _sprint_risks(plan, scope)
    if ops is not None:
        risks += _ops_risks(ops)
    return sorted(risks, key=lambda r: -r.score)
