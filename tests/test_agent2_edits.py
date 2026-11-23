import pytest

from agent2_ops import agent as a2
from agent2_ops.planner import ticket_key
from orchestrator.schemas import OpsResult

SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "d1", "estimate_points": 3},
        {"title": "Auth API", "description": "d2", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [{"title": "Task list", "description": "d3", "estimate_points": 2}]},
]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SPRINT_CAPACITY", raising=False)


def _edited():
    return [  # deliberately stale or wrong flags: the server must not trust them
        {"key": "x", "title": "Login UI", "epic": "?", "points": 3, "sprint": 2, "assignee": "Bob",
         "confidence": 1.0, "needs_human": True, "status": "created", "card_id": "c1"},
        {"key": "y", "title": "Auth API", "epic": "Auth", "points": 5, "sprint": 1, "assignee": None,
         "confidence": 0.1, "needs_human": False},
        {"key": "z", "title": "Task list", "epic": "Tasks", "points": 2, "sprint": 1, "assignee": "Alice",
         "confidence": 0.4, "needs_human": False},
    ]


def _no_matcher(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the matcher must not run for an edited plan")

    monkeypatch.setattr(a2, "assign", boom)


def test_edited_plan_is_kept_exactly_and_nothing_is_written(monkeypatch):
    _no_matcher(monkeypatch)
    out = a2.plan_sprints({"scope": SCOPE, "board_name": "P", "tickets": _edited()})
    assert "errors" not in out
    sprints = {s["number"]: s["items"] for s in out["sprint_plan"]["sprints"]}
    assert sprints == {1: ["Auth API", "Task list"], 2: ["Login UI"]}
    assert all(s["capacity_points"] == 28 for s in out["sprint_plan"]["sprints"])  # 10 + 10 + 8 from team.json
    ops = out["ops_result"]
    assert ops["written"] is False and ops["model_version"] == "human-edited"
    by = {t["title"]: t for t in ops["tickets"]}
    assert by["Login UI"]["assignee"] == "Bob" and by["Login UI"]["sprint"] == 2
    assert by["Login UI"]["needs_human"] is False and by["Auth API"]["needs_human"] is True  # recomputed
    assert by["Login UI"]["key"] == ticket_key("P", "Login UI")  # recomputed, not trusted
    assert by["Login UI"]["epic"] == "Auth" and by["Login UI"]["status"] == "proposed" and by["Login UI"]["card_id"] is None
    assert [t["title"] for t in ops["tickets"]] == ["Login UI", "Auth API", "Task list"]  # scope order


def test_approving_an_edited_plan_writes_those_tickets(monkeypatch):
    _no_matcher(monkeypatch)
    seen = {}

    def fake_write(tickets, descriptions, members, board_name, approved, approved_by, model_version, client=None):
        seen.update(tickets=tickets, descriptions=descriptions, board=board_name, approved=approved,
                    by=approved_by, version=model_version)
        return OpsResult(written=True, model_version=model_version, tickets=tickets)

    monkeypatch.setattr(a2, "write_to_trello", fake_write)
    out = a2.plan_sprints({"scope": SCOPE, "board_name": "P", "tickets": _edited(), "approved": True, "approved_by": "wissem"})
    assert out["ops_result"]["written"] is True
    assert seen["approved"] is True and seen["by"] == "wissem" and seen["board"] == "P"
    assert seen["version"] == "human-edited"
    assert {(t.title, t.sprint, t.assignee) for t in seen["tickets"]} == {
        ("Login UI", 2, "Bob"), ("Auth API", 1, None), ("Task list", 1, "Alice")}
    assert seen["descriptions"]["Auth API"] == "d2"  # card text still comes from the scope


@pytest.mark.parametrize("break_it, message", [
    (lambda t: t[:-1], "no ticket for: Task list"),
    (lambda t: t + [{**t[0], "title": "Ghost"}], "does not match any story: Ghost"),
    (lambda t: t + [dict(t[0])], "more than one ticket for: Login UI"),
    (lambda t: [{**t[0], "sprint": 0}] + t[1:], "at least 1: Login UI"),
    (lambda t: [{**t[0], "points": 0}] + t[1:], "at least 1: Login UI"),
])
def test_inconsistent_tickets_are_rejected(monkeypatch, break_it, message):
    _no_matcher(monkeypatch)
    out = a2.plan_sprints({"scope": SCOPE, "tickets": break_it(_edited())})
    assert message in out["errors"][0]
    assert "ops_result" not in out


def test_without_tickets_the_original_behaviour_is_unchanged(monkeypatch):
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: ("Alice", 0.9) for s in stories}, "stub"))
    out = a2.plan_sprints({"scope": SCOPE})
    assert [s["items"] for s in out["sprint_plan"]["sprints"]] == [["Login UI", "Auth API", "Task list"]]  # 10 points < 28
    ops = out["ops_result"]
    assert ops["model_version"] == "stub" and ops["written"] is False
    assert all(t["assignee"] == "Alice" and t["confidence"] == 0.9 and not t["needs_human"] for t in ops["tickets"])
    assert out["sprint_plan"]["sprints"][0]["capacity_points"] == 28


def test_capacity_override_still_works(monkeypatch):
    monkeypatch.setenv("SPRINT_CAPACITY", "6")
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: (None, 0.0) for s in stories}, "stub"))
    out = a2.plan_sprints({"scope": SCOPE})
    assert [s["items"] for s in out["sprint_plan"]["sprints"]] == [["Login UI"], ["Auth API"], ["Task list"]]
    assert all(t["needs_human"] for t in out["ops_result"]["tickets"])
