"""Agent 2 (ops & assignment): the whole agent in one test file. No network: Trello is faked, no model download.

Sections
  1. proposal, skill matching, capacity, Trello writes (approval gate, idempotency, rate-limit retries)
  2. plans edited by the lead are kept exactly as left
  3. review flow: sprints -> lead edits -> assignment inside them -> locks -> approve (+ workload %, review stats),
     LLM-as-a-judge and the offline evaluation against baselines
  4. backlog edited by the lead (added / renamed / removed epics and stories)
"""
import json

import pytest
from fastapi.testclient import TestClient

from agent2_ops import agent as a2
from agent2_ops import agent as agent2
from agent2_ops import judge as judge_mod
from agent2_ops.agent import plan_sprints, write_to_trello
from agent2_ops.evaluate_matcher import evaluate, load_scopes
from agent2_ops.planner import load_balance, pack_sprints, review_stats, ticket_key, workload
from orchestrator.main import app
from orchestrator.schemas import Member, OpsResult, Ticket

client = TestClient(app)


@pytest.fixture(autouse=True)
def agent2_env(monkeypatch, tmp_path):
    """Self-contained: keyword matcher (no download), audit log in a temp folder, no real board, default capacity."""
    monkeypatch.setenv("SKILL_MATCHER", "keywords")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("TRELLO_BOARD_ID", raising=False)
    monkeypatch.delenv("SPRINT_CAPACITY", raising=False)




# ====================================================================================================
# 1. proposal, matching, capacity, Trello writes
# ====================================================================================================
SCOPE = {
    "epics": [
        {
            "title": "Accounts",
            "stories": [
                {"title": "Login API", "description": "backend api for authentication", "estimate_points": 5},
                {"title": "Login page", "description": "frontend ui with react", "estimate_points": 5},
            ],
        },
        {
            "title": "Quality",
            "stories": [
                {"title": "Write tests", "description": "testing and qa", "estimate_points": 8},
                {"title": "Mystery task", "description": "zzz", "estimate_points": 3},
            ],
        },
    ]
}


class FakeTrello:
    """In-memory stand-in for TrelloClient."""

    def __init__(self):
        self.cards = {}  # id -> dict
        self.lists = {}

    def get_or_create_board(self, name):
        return {"id": "board1", "url": "https://trello.com/b/board1"}

    def ensure_list(self, board_id, name):
        return self.lists.setdefault(name, f"list-{name}")

    def existing_cards(self, board_id):
        from agent2_ops.trello import KEY_RE

        return {KEY_RE.search(c["desc"]).group(1): c for c in self.cards.values()}

    def create_card(self, list_id, name, desc, member_ids):
        cid = f"card{len(self.cards) + 1}"
        self.cards[cid] = {"id": cid, "desc": desc, "shortUrl": f"https://trello.com/c/{cid}"}
        return self.cards[cid]

    def member_id_on_board(self, username, board_id):
        return None


def test_proposal_is_valid_and_writes_nothing(monkeypatch):
    # if Agent 2 touched Trello in proposal mode this would blow up
    monkeypatch.setattr(agent2, "TrelloClient", lambda *a, **k: pytest.fail("Trello used without approval"))
    out = plan_sprints({"scope": SCOPE})
    assert "errors" not in out
    assert out["ops_result"]["written"] is False
    titles = [i for s in out["sprint_plan"]["sprints"] for i in s["items"]]
    assert sorted(titles) == sorted(["Login API", "Login page", "Write tests", "Mystery task"])


def test_skill_matching_and_low_confidence_goes_to_human():
    tickets = {t["title"]: t for t in plan_sprints({"scope": SCOPE})["ops_result"]["tickets"]}
    assert tickets["Login API"]["assignee"] == "Alice"
    assert tickets["Login page"]["assignee"] == "Bob"
    assert tickets["Write tests"]["assignee"] == "Chloe"
    assert tickets["Mystery task"]["assignee"] is None and tickets["Mystery task"]["needs_human"]


def test_sprints_respect_capacity():
    stories = [{"title": f"s{i}", "points": 5} for i in range(5)]
    plan, sprint_of = pack_sprints(stories, capacity=10)
    assert [len(s.items) for s in plan.sprints] == [2, 2, 1]
    assert sprint_of["s4"] == 3


def test_write_is_refused_without_approval():
    t = Ticket(key="a" * 16, title="x", epic="e", points=1, sprint=1)
    with pytest.raises(PermissionError):
        write_to_trello([t], {"x": ""}, [], "b", approved=False, approved_by="me",
                        model_version="v", client=FakeTrello())


def test_approved_write_is_idempotent(monkeypatch, tmp_path):
    fake = FakeTrello()
    monkeypatch.setattr(agent2.TrelloClient, "from_env", classmethod(lambda cls: fake))
    state = {"scope": SCOPE, "approved": True, "approved_by": "tester", "board_name": "B"}

    first = plan_sprints(state)["ops_result"]
    assert first["written"] and len(fake.cards) == 4
    assert {t["status"] for t in first["tickets"]} == {"created"}

    second = plan_sprints(state)["ops_result"]  # retry: must not duplicate
    assert len(fake.cards) == 4
    assert {t["status"] for t in second["tickets"]} == {"already_exists"}
    assert (tmp_path / "audit.jsonl").read_text().count("create_card") == 8  # 4 created + 4 skipped


def test_api_plan_then_approve(monkeypatch):
    fake = FakeTrello()
    monkeypatch.setattr(agent2.TrelloClient, "from_env", classmethod(lambda cls: fake))
    client = TestClient(app)
    plan = client.post("/plan", json={"project_description": "Build a todo app"}).json()
    assert plan["ops_result"]["written"] is False and not fake.cards
    done = client.post("/approve", json={"scope": plan["scope"], "approved_by": "me"}).json()
    assert done["ops_result"]["written"] is True and len(fake.cards) == 1


def test_bad_scope_returns_error_not_crash():
    assert "agent2" in plan_sprints({"scope": {"epics": []}})["errors"][0]


def test_trello_client_retries_on_rate_limit(monkeypatch):
    """429 twice, then success -> the client must wait (exponential) and retry."""
    from agent2_ops import trello

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body, self.ok = code, body, code < 400
            self.text = ""

        def json(self):
            return self._body

    replies = [Resp(429), Resp(429), Resp(200, {"id": "abc"})]
    sleeps = []
    monkeypatch.setattr(trello.requests, "request", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(trello.time, "sleep", sleeps.append)

    assert trello.TrelloClient("k", "t")._call("GET", "/members/me") == {"id": "abc"}
    assert sleeps == [1, 2]


# ====================================================================================================
# 2. plans edited by the lead
# ====================================================================================================
EDIT_SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "d1", "estimate_points": 3},
        {"title": "Auth API", "description": "d2", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [{"title": "Task list", "description": "d3", "estimate_points": 2}]},
]}




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
    out = a2.plan_sprints({"scope": EDIT_SCOPE, "board_name": "P", "tickets": _edited()})
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
    out = a2.plan_sprints({"scope": EDIT_SCOPE, "board_name": "P", "tickets": _edited(), "approved": True, "approved_by": "wissem"})
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
    out = a2.plan_sprints({"scope": EDIT_SCOPE, "tickets": break_it(_edited())})
    assert message in out["errors"][0]
    assert "ops_result" not in out


def test_without_tickets_the_original_behaviour_is_unchanged(monkeypatch):
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: ("Alice", 0.9) for s in stories}, "stub"))
    out = a2.plan_sprints({"scope": EDIT_SCOPE})
    assert [s["items"] for s in out["sprint_plan"]["sprints"]] == [["Login UI", "Auth API", "Task list"]]  # 10 points < 28
    ops = out["ops_result"]
    assert ops["model_version"] == "stub" and ops["written"] is False
    assert all(t["assignee"] == "Alice" and t["confidence"] == 0.9 and not t["needs_human"] for t in ops["tickets"])
    assert out["sprint_plan"]["sprints"][0]["capacity_points"] == 28


def test_capacity_override_still_works(monkeypatch):
    monkeypatch.setenv("SPRINT_CAPACITY", "6")
    monkeypatch.setattr(a2, "assign", lambda stories, members: ({s["title"]: (None, 0.0) for s in stories}, "stub"))
    out = a2.plan_sprints({"scope": EDIT_SCOPE})
    assert [s["items"] for s in out["sprint_plan"]["sprints"]] == [["Login UI"], ["Auth API"], ["Task list"]]
    assert all(t["needs_human"] for t in out["ops_result"]["tickets"])


# ====================================================================================================
# 3. review flow, workload %, LLM judge, offline evaluation
# ====================================================================================================
REVIEW_TEAM = [
    {"name": "Alice", "skills": ["backend", "api"], "capacity_points": 10},
    {"name": "Bob", "skills": ["frontend", "ui"], "capacity_points": 10},
]
REVIEW_SCOPE = {"epics": [
    {"title": "Auth", "stories": [
        {"title": "Login UI", "description": "ui form", "estimate_points": 3},
        {"title": "Auth API", "description": "backend api", "estimate_points": 5}]},
    {"title": "Tasks", "stories": [
        {"title": "Task API", "description": "backend api", "estimate_points": 5},
        {"title": "Task list", "description": "ui list", "estimate_points": 2}]},
]}




def _stage1():
    r = client.post("/steps/plan-sprints", json={"scope": REVIEW_SCOPE, "team_members": REVIEW_TEAM, "board_name": "P"})
    assert r.status_code == 200, r.text
    return r.json()


def _assign(tickets):
    r = client.post("/steps/assign", json={"scope": REVIEW_SCOPE, "tickets": tickets, "team_members": REVIEW_TEAM, "board_name": "P"})
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
    r = client.post("/steps/assign", json={"scope": REVIEW_SCOPE, "tickets": tickets, "team_members": REVIEW_TEAM})
    assert r.status_code == 422 and "Zed" in r.json()["detail"][0]["msg"]


# ------------------------------------------------------------------ approve: review stats + audit + version
def test_approve_measures_and_logs_the_leads_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(a2, "write_to_trello", lambda tickets, *a, **k: OpsResult(written=True, tickets=tickets,
                                                                                 model_version=a[5]))
    stage2 = _assign(_stage1()["ops_result"]["tickets"])
    tickets = stage2["ops_result"]["tickets"]
    tickets[0].update(assignee="Alice", confidence=1.0)      # one assignee changed
    tickets[3].update(sprint=2)                              # one sprint changed
    r = client.post("/steps/approve", json={"scope": REVIEW_SCOPE, "tickets": tickets, "team_members": REVIEW_TEAM, "board_name": "P",
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
    r = client.post("/steps/judge", json={"scope": REVIEW_SCOPE, "tickets": tickets, "team_members": REVIEW_TEAM})
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
    (tmp_path / "s.json").write_text(json.dumps({"scope": REVIEW_SCOPE}))
    members = [Member.model_validate(m) for m in REVIEW_TEAM]
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


# ====================================================================================================
# 4. backlog edited by the lead
# ====================================================================================================




def _plan(epics):
    return client.post("/steps/plan-sprints", json={"scope": {"epics": epics}})


def test_an_edited_backlog_with_a_new_epic_is_planned():
    r = _plan([
        {"title": "Auth", "stories": [{"title": "Login", "estimate_points": 3}]},
        {"title": "Added by the lead", "stories": [{"title": "Export to PDF", "description": "d", "estimate_points": 5}]},
    ])
    assert r.status_code == 200, r.text
    tickets = {t["title"]: t for t in r.json()["ops_result"]["tickets"]}
    assert tickets["Export to PDF"]["epic"] == "Added by the lead" and tickets["Export to PDF"]["points"] == 5


@pytest.mark.parametrize("epics, message", [
    ([{"title": "A", "stories": [{"title": "Login"}]}, {"title": "B", "stories": [{"title": "login "}]}], "unique"),
    ([{"title": "A", "stories": [{"title": "  "}]}], "no title"),
    ([{"title": " ", "stories": [{"title": "Login"}]}], "epic needs a title"),
])
def test_a_backlog_that_would_break_planning_is_a_422(epics, message):
    r = _plan(epics)
    assert r.status_code == 422 and message in r.json()["detail"][0]["msg"]
