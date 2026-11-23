"""Login, roles and team accounts (orchestrator/db.py + orchestrator/auth.py)."""
import pytest
from fastapi.testclient import TestClient

from orchestrator import db
from orchestrator.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANPILOT_AUTH", "1")                  # this file tests the checks themselves
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "users.db"))
    monkeypatch.setenv("ADMIN_EMAIL", "Admin@Example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    db.init()


def _login(email="admin@example.com", password="secret123"):
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_admin_is_created_once_from_env(monkeypatch):
    users = db.list_users()
    assert [(u["email"], u["role"], u["in_team"]) for u in users] == [("admin@example.com", "admin", False)]
    monkeypatch.setenv("ADMIN_EMAIL", "other@example.com")
    db.init()                                                  # not empty any more: nothing new
    assert len(db.list_users()) == 1


def test_passwords_are_hashed_and_checked():
    with db.connect() as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert "secret123" not in stored and db.check_password("secret123", stored)
    r = client.post("/auth/login", json={"email": "admin@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_admin_adds_the_team_and_members_cannot():
    admin = _login()
    r = client.post("/users", headers=admin, json={"email": "lead@x.com", "name": "Lea", "password": "leadpass",
                                                   "role": "lead", "skills": ["backend", "api"]})
    assert r.status_code == 200 and r.json()["skills"] == ["backend", "api"]
    client.post("/users", headers=admin, json={"email": "bo@x.com", "name": "Bo", "password": "bopass1"})
    member = _login("bo@x.com", "bopass1")
    assert client.get("/users", headers=member).status_code == 200            # can see the team
    assert client.post("/users", headers=member, json={"email": "z@x.com", "name": "Z",
                                                       "password": "zzzzzz"}).status_code == 403
    dup = client.post("/users", headers=admin, json={"email": "BO@x.com", "name": "B", "password": "bopass1"})
    assert dup.status_code == 422 and "already exists" in dup.json()["detail"][0]["msg"]


def test_planning_needs_a_lead_or_admin():
    admin = _login()
    client.post("/users", headers=admin, json={"email": "bo@x.com", "name": "Bo", "password": "bopass1"})
    body = {"scope": {"epics": [{"title": "E", "stories": [{"title": "S", "estimate_points": 3}]}]}}
    assert client.post("/steps/plan-sprints", json=body).status_code == 401            # not logged in
    assert client.post("/steps/plan-sprints", json=body, headers=_login("bo@x.com", "bopass1")).status_code == 403
    assert client.post("/steps/plan-sprints", json=body, headers=admin).status_code == 200
    assert client.post("/approve", json={"scope": {}}).status_code == 401


def test_change_password_and_logout():
    h = _login()
    assert client.post("/auth/password", headers=h, json={"current_password": "nope", "new_password": "xxxxxx"}
                       ).status_code == 422
    assert client.post("/auth/password", headers=h, json={"current_password": "secret123",
                                                          "new_password": "newpass9"}).status_code == 200
    assert client.get("/auth/me", headers=h).json()["email"] == "admin@example.com"
    client.post("/auth/logout", headers=h)
    assert client.get("/auth/me", headers=h).status_code == 401
    _login(password="newpass9")


def test_the_only_admin_cannot_be_demoted_or_delete_themself():
    h = _login()
    me = client.get("/auth/me", headers=h).json()
    assert client.put(f"/users/{me['id']}", headers=h, json={"role": "member"}).status_code == 422
    assert client.delete(f"/users/{me['id']}", headers=h).status_code == 422
