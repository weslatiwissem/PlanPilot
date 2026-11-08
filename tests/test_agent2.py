"""Agent 2 tests: no network, Trello is replaced by an in-memory fake."""
import pytest
from fastapi.testclient import TestClient

from agent2_ops import agent as agent2
from agent2_ops.agent import plan_sprints, write_to_trello
from agent2_ops.planner import pack_sprints
from orchestrator.main import app
from orchestrator.schemas import Ticket

@pytest.fixture(autouse=True)
def agent2_env(monkeypatch, tmp_path):
    """Self-contained: no model download, audit log in a temp folder, no real board."""
    monkeypatch.setenv("SKILL_MATCHER", "keywords")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("TRELLO_BOARD_ID", raising=False)


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
