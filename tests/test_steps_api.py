import pytest
from fastapi.testclient import TestClient

from agent2_ops import agent as a2
from orchestrator import steps
from orchestrator.main import app
from orchestrator.schemas import OpsResult

client = TestClient(app)
SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "d1", "estimate_points": 3},
        {"title": "Auth API", "description": "d2", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [{"title": "Task list", "description": "d3", "estimate_points": 2}]},
]}


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.delenv("SPRINT_CAPACITY", raising=False)
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: ("Alice", 0.9) for s in stories}, "stub"))


def _tickets(**changes):
    base = {
        "Login UI": dict(title="Login UI", epic="Auth", points=3, sprint=1, assignee="Bob", confidence=0.8, key="k1"),
        "Auth API": dict(title="Auth API", epic="Auth", points=5, sprint=1, assignee="Alice", confidence=0.8, key="k2"),
        "Task list": dict(title="Task list", epic="Tasks", points=2, sprint=1, assignee="Alice", confidence=0.8, key="k3"),
    }
    for title, edit in changes.items():
        base[title.replace("_", " ")].update(edit)
    return list(base.values())


# ------------------------------------------------------------------ /steps/scope
def test_scope_returns_questions_then_scope(monkeypatch):
    seen = []

    def fake(state):
        seen.append(state["project_description"])
        return {"clarifying_questions": ["Which platform?"]} if len(seen) == 1 else {"scope": SCOPE, "clarifying_questions": []}

    monkeypatch.setattr(steps, "scope_project", fake)
    r = client.post("/steps/scope", json={"project_description": "A todo app"})
    assert r.status_code == 200 and r.json() == {"status": "needs_clarification", "clarifying_questions": ["Which platform?"], "scope": None}
    r = client.post("/steps/scope", json={"project_description": "A todo app",
                                          "clarifications": [{"question": "Which platform?", "answer": "Web"}]})
    assert r.json()["status"] == "ok" and r.json()["scope"] == SCOPE
    assert seen[1] == "A todo app\n\n## Clarifications\n- Which platform?: Web"  # Agent 1's marker, added for the UI


def test_skip_clarification_adds_the_marker(monkeypatch):
    seen = []
    monkeypatch.setattr(steps, "scope_project", lambda s: seen.append(s) or {"scope": SCOPE})
    client.post("/steps/scope", json={"project_description": "A todo app", "skip_clarification": True})
    assert "## Clarifications" in seen[0]["project_description"] and "assumptions" in seen[0]["project_description"]


def test_agent1_failure_is_a_502(monkeypatch):
    monkeypatch.setattr(steps, "scope_project", lambda s: {"errors": ["agent1: API key not valid"]})
    r = client.post("/steps/scope", json={"project_description": "x"})
    assert r.status_code == 502 and "API key" in r.json()["detail"][0]["msg"]
    assert r.json()["detail"][0]["type"] == "agent_error"


# ------------------------------------------------------------------ /steps/sprints
def test_sprints_returns_proposal_tickets_and_risks():
    r = client.post("/steps/sprints", json={"scope": SCOPE, "board_name": "P"})
    assert r.status_code == 200
    body = r.json()
    assert body["sprint_plan"]["sprints"][0]["items"] == ["Login UI", "Auth API", "Task list"]
    assert body["ops_result"]["written"] is False and len(body["ops_result"]["tickets"]) == 3
    assert isinstance(body["risks"], list)


def test_sprints_with_an_empty_scope_is_a_422():
    r = client.post("/steps/sprints", json={"scope": {"epics": []}})
    assert r.status_code == 422 and "no stories" in r.json()["detail"][0]["msg"]
    assert client.post("/steps/sprints", json={"scope": {"epics": "nonsense"}}).status_code == 422


# ------------------------------------------------------------------ /steps/risks
def test_risks_follow_the_users_edits():
    ok = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets()}).json()
    assert not any("No confident assignee" in r["reason"] for r in ok["risks"])

    unassigned = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(Task_list={"assignee": None})}).json()
    assert any(r["item"] == "Task list" and "No confident assignee" in r["reason"] for r in unassigned["risks"])

    # points are canonical: editing a ticket to 40 points overloads the 28-point sprint
    heavy = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(Auth_API={"points": 40})}).json()
    assert any(r["item"] == "Sprint 1" and "capacity of 28" in r["reason"] for r in heavy["risks"])

    # moving a story to a new sprint changes the plan
    moved = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(Task_list={"sprint": 2})}).json()
    assert {s["number"]: s["items"] for s in moved["sprint_plan"]["sprints"]} == {1: ["Login UI", "Auth API"], 2: ["Task list"]}


def test_choosing_an_assignee_yourself_clears_the_weak_match_warning():
    weak = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(Task_list={"confidence": 0.05})}).json()
    assert any(r["item"] == "Task list" and "Weak skill match" in r["reason"] for r in weak["risks"])
    chosen = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets(Task_list={"confidence": 1.0})}).json()
    assert not any(r["item"] == "Task list" for r in chosen["risks"])


def test_risks_rejects_tickets_that_do_not_match_the_scope():
    r = client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets()[:2]})
    assert r.status_code == 422 and "no ticket for: Task list" in r.json()["detail"][0]["msg"]


# ------------------------------------------------------------------ /steps/approve
def test_approve_writes_the_edited_plan(monkeypatch):
    seen = {}

    def fake_write(tickets, descriptions, members, board_name, approved, approved_by, model_version, client=None):
        seen.update(tickets=tickets, board=board_name, approved=approved, by=approved_by)
        return OpsResult(written=True, board_url="https://trello.example/b/1", model_version=model_version, tickets=tickets)

    monkeypatch.setattr(a2, "write_to_trello", fake_write)
    r = client.post("/steps/approve", json={"scope": SCOPE, "tickets": _tickets(Task_list={"sprint": 2, "assignee": "Chloe"}),
                                            "board_name": "Demo", "approved_by": "wissem"})
    assert r.status_code == 200
    body = r.json()
    assert body["ops_result"]["written"] is True and body["ops_result"]["board_url"].startswith("https://")
    assert seen["approved"] is True and seen["by"] == "wissem" and seen["board"] == "Demo"
    assert {(t.title, t.sprint, t.assignee) for t in seen["tickets"]} >= {("Task list", 2, "Chloe")}
    assert {s["number"] for s in body["sprint_plan"]["sprints"]} == {1, 2}
    assert "risks" in body


def test_approve_validates_before_writing(monkeypatch):
    monkeypatch.setattr(a2, "write_to_trello", lambda *a, **k: pytest.fail("must not write"))
    r = client.post("/steps/approve", json={"scope": SCOPE, "tickets": _tickets()[:1]})
    assert r.status_code == 422


def test_trello_failure_is_a_502(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Trello said no")

    monkeypatch.setattr(a2, "write_to_trello", boom)
    r = client.post("/steps/approve", json={"scope": SCOPE, "tickets": _tickets()})
    assert r.status_code == 502 and "Trello said no" in r.json()["detail"][0]["msg"]


# ------------------------------------------------------------------ browser access + old endpoints
def test_cors_allows_the_vite_dev_server():
    r = client.options("/steps/risks", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "POST",
                                                "Access-Control-Request-Headers": "content-type"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_old_endpoints_still_exist():
    assert client.get("/health").json() == {"status": "ok"}
    assert {"/plan", "/approve"} <= {route.path for route in app.routes}


# ------------------------------------------------------------------ typed contract for the frontend
def test_openapi_describes_every_step_response():
    schema = client.get("/openapi.json").json()
    expected = {"/steps/scope": "ScopeStepResponse", "/steps/sprints": "SprintsStepResponse",
                "/steps/risks": "RisksStepResponse", "/steps/approve": "SprintsStepResponse"}
    for path, model in expected.items():
        out = schema["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert out["$ref"].endswith(model), path
    assert {"Ticket", "Scope", "SprintPlan", "OpsResult", "Risk"} <= set(schema["components"]["schemas"])


def test_a_ticket_with_missing_fields_is_rejected_in_fastapis_error_shape():
    r = client.post("/steps/risks", json={"scope": SCOPE, "tickets": [{"title": "Login UI"}]})
    assert r.status_code == 422
    assert any("tickets" in item["loc"] for item in r.json()["detail"])


def test_every_error_has_the_same_shape(monkeypatch):
    monkeypatch.setattr(steps, "scope_project", lambda s: {"errors": ["agent1: boom"]})
    errors = [
        client.post("/steps/scope", json={"project_description": "x"}),                      # 502
        client.post("/steps/risks", json={"scope": SCOPE, "tickets": _tickets()[:1]}),       # 422, our check
        client.post("/steps/risks", json={"scope": SCOPE, "tickets": [{"title": "x"}]}),     # 422, FastAPI's check
    ]
    for r in errors:
        assert r.status_code in (422, 502)
        assert all({"loc", "msg", "type"} <= set(item) for item in r.json()["detail"])
