import os
import tempfile

import pytest

# The agents' tests call the API without logging in: switch the login checks off (tests/test_auth.py
# switches them back on) and keep the users database out of the project folder.
os.environ.setdefault("PLANPILOT_AUTH", "0")
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.gettempdir(), "planpilot_test.db"))

from agent1_scoping.models import ClarificationCheck, LLMEpic, LLMScope, LLMStory


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Never call Gemini or load the embedding model in tests."""

    def fake(prompt, schema, system="", **kwargs):
        if schema is ClarificationCheck:
            return ClarificationCheck(missing_info=[], questions=[])
        if schema is LLMScope:
            return LLMScope(
                epics=[
                    LLMEpic(
                        title="Core",
                        stories=[LLMStory(title="Create task", description="", estimate_points=3)],
                    )
                ]
            )
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("agent1_scoping.clarify.generate_structured", fake)
    monkeypatch.setattr("agent1_scoping.scope.generate_structured", fake)
    monkeypatch.setattr("agent1_scoping.agent.retrieve_similar", lambda *a, **k: [])
