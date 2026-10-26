from fastapi.testclient import TestClient

from orchestrator.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_plan_endpoint():
    r = client.post("/plan", json={"project_description": "Build a todo app"})
    assert r.status_code == 200
    assert "sprint_plan" in r.json()
