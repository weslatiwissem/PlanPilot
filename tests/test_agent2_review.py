"""Review flow: lead edits sprints -> M4 assigns inside them -> lead edits/locks -> approve. Plus metrics."""
import json

import pytest
from fastapi.testclient import TestClient

from agent2_ops import agent as a2
from agent2_ops import judge as judge_mod
from agent2_ops.evaluate_matcher import evaluate, load_scopes
from agent2_ops.planner import load_balance, review_stats, workload
from orchestrator.main import app
from orchestrator.schemas import Member, OpsResult, Ticket

client = TestClient(app)
TEAM = [
    {"name": "Alice", "skills": ["backend", "api"], "capacity_points": 10},
    {"name": "Bob", "skills": ["frontend", "ui"], "capacity_points": 10},
]
SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "ui form", "estimate_points": 3},
        {"title": "Auth API", "description": "backend api", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [
        {"title": "Task API", "description": "backend api", "estimate_points": 5},
        {"title": "Task list", "description": "ui list", "estimate_points": 2}]},
]}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("SPRINT_CAPACITY", raising=False)
    monkeypatch.setenv("SKILL_MATCHER", "keywords")           # no model download in tests
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))


def _stage1():
    r = client.post("/steps/plan-sprints", json={"scope": SCOPE, "team_members": TEAM, "board_name": "P"})
    assert r.status_code == 200, r.text
    return r.json()


def _assign(tickets):
    r = client.post("/steps/assign", json={"scope": SCOPE, "tickets": tickets, "team_members": TEAM, "board_name": "P"})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ stage 1: sprints only
def test_stage1_proposes_sprints_without_people():
    body = _stage1()
    tickets = body["ops_result"]["tickets"]
    assert all(t["assignee"] is None and t["proposed_sprint"] == t["sprint"] for t in tickets)
    assert body["sprint_load"][0]["capacity_points"] == 20 and body["sprint_load"][0]["points"] == 15
    assert not any("No confident assignee" in r["reason"] for r in body["risks"])  # people come later


# ------------------------------------------------------------------ stage 2: M4 follows the lead's sprints
def test_assignment_follows_the_leads_sprint_edits():
    tickets = _stage1()["ops_result"]["tickets"]
    for t in tickets:
        if t["title"] == "Task API":
            t["sprint"] = 2                                   # the lead moves a story
    ops = _assign(tickets)["ops_result"]
    by = {t["title"]: t for t in ops["tickets"]}
    assert by["Task API"]["sprint"] == 2 and by["Task API"]["proposed_sprint"] == 1  # edit kept, proposal remembered
    assert by["Auth API"]["assignee"] == "Alice" and by["Login UI"]["assignee"] == "Bob"
    assert all(t["proposed_assignee"] == t["assignee"] for t in ops["tickets"])
    rows = {(w["sprint"], w["member"]): w for w in ops["workload"]}
    assert rows[(2, "Alice")]["points"] == 5 and rows[(2, "Alice")]["utilisation_pct"] == 50
    assert rows[(2, "Alice")]["share_pct"] == 100 and ops["load_balance"] is not None


def test_locked_tickets_are_kept_and_the_rest_is_redistributed():
    tickets = _assign(_stage1()["ops_result"]["tickets"])["ops_result"]["tickets"]
    for t in tickets:  # the lead gives BOTH backend stories to Bob and locks them (10 pts = his full capacity)
        if t["title"] in ("Auth API", "Task API"):
            t.update(assignee="Bob", locked=True)
    again = {t["title"]: t for t in _assign(tickets)["ops_result"]["tickets"]}
    assert again["Auth API"]["assignee"] == "Bob" and again["Auth API"]["confidence"] == 1.0
    # Bob is now full, so M4 takes the UI work off him. Nobody else has UI skills, so it goes to a human
    # instead of being forced on Alice.
    for title in ("Login UI", "Task list"):
        assert again[title]["assignee"] != "Bob" and again[title]["needs_human"] is True


def test_locking_someone_outside_the_team_is_a_422():
    tickets = _stage1()["ops_result"]["tickets"]
    tickets[0].update(assignee="Zed", locked=True)
    r = client.post("/steps/assign", json={"scope": SCOPE, "tickets": tickets, "team_members": TEAM})
    assert r.status_code == 422 and "Zed" in r.json()["detail"][0]["msg"]


# ------------------------------------------------------------------ approve: review stats + audit + version
def test_approve_measures_and_logs_the_leads_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(a2, "write_to_trello", lambda tickets, *a, **k: OpsResult(written=True, tickets=tickets,
                                                                                 model_version=a[5]))
    stage2 = _assign(_stage1()["ops_result"]["tickets"])
    tickets = stage2["ops_result"]["tickets"]
    tickets[0].update(assignee="Alice", confidence=1.0)      # one assignee changed
    tickets[3].update(sprint=2)                              # one sprint changed
    r = client.post("/steps/approve", json={"scope": SCOPE, "tickets": tickets, "team_members": TEAM, "board_name": "P",
                                            "approved_by": "lead", "model_version": stage2["ops_result"]["model_version"]})
    assert r.status_code == 200, r.text
    ops = r.json()["ops_result"]
    assert ops["review"]["assignees_changed"] == 1 and ops["review"]["assignee_acceptance_rate"] == 0.75
    assert ops["review"]["sprints_changed"] == 1
    assert ops["model_version"] == "keywords-v1 (human-edited)"
    log = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    actions = [e["action"] for e in log]
    assert "override_assignee" in actions and "override_sprint" in actions and "review_summary" in actions


# ------------------------------------------------------------------ pure functions
def test_workload_balance_and_review_math():
    members = [Member(name="A", capacity_points=10), Member(name="B", capacity_points=10, current_load=5)]
    tickets = [Ticket(key="1", title="x", epic="e", points=5, sprint=1, assignee="A", proposed_assignee="B"),
               Ticket(key="2", title="y", epic="e", points=3, sprint=1, assignee=None, proposed_sprint=2)]
    rows = {r.member: r for r in workload(tickets, members)}
    assert rows["A"].utilisation_pct == 50 and rows["A"].share_pct == 62
    assert rows["B"].utilisation_pct == 50 and rows["B"].capacity_points == 5   # current load counts
    assert rows["Unassigned"].points == 3
    assert load_balance(list(rows.values())) == 1.0
    stats = review_stats(tickets)
    assert stats.assignee_acceptance_rate == 0.0 and stats.sprint_acceptance_rate == 0.0


# ------------------------------------------------------------------ judge + offline evaluation
def _fake_judge(prompt, schema, **kw):
    """Gives 5 when the assignee's name appears in a backend line for Alice / ui line for Bob, else 2."""
    verdicts = []
    for line in prompt.split("Assignments to review")[1].splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and parts[0].isdigit():
            best = "Alice" if "backend" in parts[1] else "Bob"
            verdicts.append({"id": int(parts[0]), "score": 5 if parts[2] == best else 2, "best_member": best,
                             "reason": "r"})
    return schema.model_validate({"verdicts": verdicts})


def test_judge_endpoint_grades_every_assigned_ticket(monkeypatch):
    monkeypatch.setattr(judge_mod, "generate_structured", _fake_judge)
    tickets = _assign(_stage1()["ops_result"]["tickets"])["ops_result"]["tickets"]
    tickets[0]["assignee"] = "Alice"  # Login UI to the backend person: the judge should disagree
    r = client.post("/steps/judge", json={"scope": SCOPE, "tickets": tickets, "team_members": TEAM})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["judged"] == 4 and body["agreement_rate"] == 0.75 and body["mean_score"] == 4.25
    bad = next(v for v in body["verdicts"] if v["title"] == "Login UI")
    assert bad["agrees"] is False and bad["best_member"] == "Bob"


def test_offline_evaluation_compares_strategies(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_mod, "generate_structured", _fake_judge)
    # "m4" would download the embedding model: stand it in with the keyword matcher for the test
    from agent2_ops import matcher
    monkeypatch.setattr(matcher, "_embedding_sims", matcher._keyword_sims)
    (tmp_path / "s.json").write_text(json.dumps({"scope": SCOPE}))
    members = [Member.model_validate(m) for m in TEAM]
    res = evaluate(load_scopes(str(tmp_path)), members, use_judge=True)
    assert set(res) == {"m4", "keywords", "round_robin", "random"}
    assert res["m4"]["metrics"]["judge_mean_score"] == 5.0 and res["m4"]["metrics"]["coverage"] == 1.0
    assert res["m4"]["metrics"]["judge_mean_score"] >= res["round_robin"]["metrics"]["judge_mean_score"]


def test_matcher_never_overloads_someone_who_has_a_teammate_with_room(monkeypatch):
    from agent2_ops import matcher
    team = [Member(name="Alice", skills=["api"], capacity_points=10), Member(name="Bob", skills=["api", "ui"],
                                                                            capacity_points=10)]
    stories = [{"title": f"API {i}", "epic": "e", "description": "api", "points": 5} for i in range(3)]
    monkeypatch.setattr(matcher, "_keyword_sims", lambda texts, members: __import__("numpy").array(
        [[1.0, 0.6]] * len(texts)))  # Alice is the better match for all three
    result, _ = matcher.assign(stories, team)
    load = {"Alice": 0, "Bob": 0}
    for who, _conf in result.values():
        load[who] += 5
    assert load == {"Alice": 10, "Bob": 5}  # 15 points, Alice's capacity is 10: the third story goes to Bob
