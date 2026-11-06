from agent1_scoping.agent import ANSWERS_MARKER, scope_project
from agent1_scoping.models import ClarificationCheck, LLMEpic, LLMScope, LLMStory
from orchestrator.schemas import Scope


def test_returns_valid_scope():
    out = scope_project({"project_description": "Build a todo app"})
    Scope.model_validate(out["scope"])
    assert out["clarifying_questions"] == []


def test_asks_questions_when_info_missing(monkeypatch):
    monkeypatch.setattr(
        "agent1_scoping.clarify.generate_structured",
        lambda *a, **k: ClarificationCheck(missing_info=["target users"], questions=["Who are the users?"]),
    )
    out = scope_project({"project_description": "Build an app"})
    assert out["clarifying_questions"] == ["Who are the users?"]
    assert "scope" not in out


def test_never_asks_again_once_answers_are_included(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("clarification must be skipped")

    monkeypatch.setattr("agent1_scoping.agent.check_missing_info", boom)
    out = scope_project({"project_description": f"Build an app\n\n{ANSWERS_MARKER}\nQ: users?\nA: students"})
    assert "scope" in out


def test_story_titles_are_unique_and_points_snapped(monkeypatch):
    story = lambda: LLMStory(title="Login", description="", estimate_points=6)
    monkeypatch.setattr(
        "agent1_scoping.scope.generate_structured",
        lambda *a, **k: LLMScope(epics=[LLMEpic(title="A", stories=[story(), story()])]),
    )
    out = scope_project({"project_description": "x"})
    titles = [s["title"] for e in out["scope"]["epics"] for s in e["stories"]]
    assert len(set(titles)) == 2
    assert {s["estimate_points"] for e in out["scope"]["epics"] for s in e["stories"]} <= {5, 8}


def test_llm_failure_is_reported_not_raised(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr("agent1_scoping.clarify.generate_structured", fail)
    out = scope_project({"project_description": "x"})
    assert "quota" in out["errors"][0]
