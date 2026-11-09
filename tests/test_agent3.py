import pytest

from agent3_risk.agent import assess_risk
from agent3_risk.heuristics import assess
from orchestrator.schemas import Epic, Risk, Scope, Sprint, SprintPlan, Story


def make_scope(**points):
    """make_scope(a=3, b=None) -> one epic with stories 'a' and 'b'."""
    return Scope(epics=[Epic(title="E", stories=[
        Story(title=t, estimate_points=p) for t, p in points.items()
    ])])


@pytest.fixture
def healthy():
    return (
        SprintPlan(sprints=[Sprint(number=1, items=["a", "b"], capacity_points=10)]),
        make_scope(a=3, b=4),
    )


def test_empty_plan_is_flagged():
    risks = assess(SprintPlan(), Scope())
    assert len(risks) == 1 and risks[0].score >= 0.9


def test_healthy_plan_has_no_risks(healthy):
    assert assess(*healthy) == []


def test_overloaded_sprint_is_flagged():
    plan = SprintPlan(sprints=[Sprint(number=1, items=["a", "b"], capacity_points=5)])
    risks = assess(plan, make_scope(a=5, b=5))
    assert risks[0].item == "Sprint 1" and risks[0].score > 0.5
    assert "capacity" in risks[0].reason


def test_unestimated_stories_are_flagged():
    plan = SprintPlan(sprints=[Sprint(number=1, items=["a", "b"], capacity_points=10)])
    risks = assess(plan, make_scope(a=3, b=None))
    assert any("no estimate" in r.reason for r in risks)


def test_large_story_is_flagged():
    plan = SprintPlan(sprints=[Sprint(number=1, items=["big"], capacity_points=20)])
    risks = assess(plan, make_scope(big=13))
    assert any(r.item == "big" for r in risks)


def test_many_items_without_capacity_is_flagged():
    titles = [f"s{i}" for i in range(9)]
    plan = SprintPlan(sprints=[Sprint(number=1, items=titles)])
    risks = assess(plan, make_scope(**{t: 1 for t in titles}))
    assert any("no capacity" in r.reason for r in risks)


def test_risks_sorted_highest_first():
    plan = SprintPlan(sprints=[Sprint(number=1, items=["a", "big"], capacity_points=5)])
    risks = assess(plan, make_scope(a=None, big=13))
    scores = [r.score for r in risks]
    assert scores == sorted(scores, reverse=True)


def test_agent_returns_valid_risk_dicts():
    state = {
        "sprint_plan": {"sprints": [{"number": 1, "items": ["a"], "capacity_points": 1}]},
        "scope": make_scope(a=5).model_dump(),
    }
    out = assess_risk(state)
    assert out["risks"] and all(Risk.model_validate(r) for r in out["risks"])


def test_agent_handles_missing_plan():
    out = assess_risk({})
    assert out["risks"][0]["item"] == "plan"


# ---------------------------------------------------------------- real Agent 1+2 output
import json
from pathlib import Path

from orchestrator.schemas import OpsResult, Ticket

FIXTURE = Path(__file__).parent / "fixtures" / "sample_plan_response.json"


def test_real_plan_response():
    """A real /plan response: 2 sprints (27/28 and 20/28 points), one unassigned ticket."""
    state = json.loads(FIXTURE.read_text())
    risks = [Risk.model_validate(r) for r in assess_risk(state)["risks"]]
    by_item = {r.item: r for r in risks}

    # the ticket Agent 2 could not assign is the top risk
    assert risks[0].item == "Scheduled Email Reminder Worker"
    assert "human" in risks[0].reason
    # sprint 1 is 96% full: flagged as tight, but not as overloaded
    assert "no slack" in by_item["Sprint 1"].reason
    assert all("capacity of" not in r.reason for r in risks)
    # weakest assigned match
    assert "Weak skill match" in by_item["Task Completion Toggle"].reason
    # sprint 2 (20/28) and the other sprint-1 matches are fine
    assert "Sprint 2" not in by_item
    assert len(risks) == 3


def _ops(*tickets, warnings=()):
    return OpsResult(tickets=list(tickets), warnings=list(warnings))


def _t(title, sprint=1, points=3, assignee="Ana", confidence=0.5, needs_human=False):
    return Ticket(key=title, title=title, epic="E", points=points, sprint=sprint,
                  assignee=assignee, confidence=confidence, needs_human=needs_human)


PLAN = SprintPlan(sprints=[Sprint(number=1, items=["x"], capacity_points=50)])
SCOPE = make_scope(x=3)


def test_unassigned_ticket_flagged_higher_when_large():
    small = assess(PLAN, SCOPE, _ops(_t("s", assignee=None, points=2)))
    big = assess(PLAN, SCOPE, _ops(_t("b", assignee=None, points=5)))
    assert small[0].score == 0.5 and big[0].score == 0.6


def test_needs_human_flag_counts_even_with_assignee():
    risks = assess(PLAN, SCOPE, _ops(_t("n", needs_human=True)))
    assert risks and "human" in risks[0].reason


def test_confident_assignments_are_not_flagged():
    ops = _ops(_t("a", assignee="Ana"), _t("b", assignee="Ben"))
    assert assess(PLAN, SCOPE, ops) == []


def test_concentration_flagged():
    ops = _ops(_t("a", points=8, assignee="Ana"), _t("b", points=2, assignee="Ben"))
    risks = assess(PLAN, SCOPE, ops)
    assert any("depends on one person" in r.reason and "Ana" in r.reason for r in risks)


def test_agent2_warnings_pass_through():
    risks = assess(PLAN, SCOPE, _ops(warnings=["Trello board not found"]))
    assert risks[0].item == "Agent 2 warning" and "Trello" in risks[0].reason


def test_ops_checks_skipped_without_ops_result():
    assert assess(PLAN, SCOPE) == []


FIXTURE_2 = Path(__file__).parent / "fixtures" / "sample_plan_response_2.json"


def test_second_real_response_has_no_false_concentration_alarm():
    """Sprint 1 is 28/28, Sprint 2 only 7/28. A 7-point sprint is not 'one-person risk'."""
    state = json.loads(FIXTURE_2.read_text())
    risks = [Risk.model_validate(r) for r in assess_risk(state)["risks"]]
    items = {r.item for r in risks}
    assert items == {
        "Sprint 1",
        "Notification Data Model and Backend Trigger",
        "Scheduled Background Reminder Worker",
        "Task Details and Management Backend",
        "Email Delivery and Error Logging",
    }
    assert all("depends on one person" not in r.reason for r in risks)


def test_small_sprint_concentration_not_flagged():
    ops = _ops(_t("a", points=5, assignee="Ana"), _t("b", points=2, assignee="Ben"))
    assert assess(PLAN, SCOPE, ops) == []
