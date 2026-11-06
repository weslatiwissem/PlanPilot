import pytest
from pydantic import BaseModel

from agent1_scoping import llm


class Out(BaseModel):
    x: int


class FakeModels:
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def generate_content(self, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("Server disconnected without sending a response.")

        class R:
            parsed = Out(x=1)
            text = '{"x": 1}'

        return R()


def _patch(monkeypatch, fails):
    models = FakeModels(fails)
    client = type("C", (), {"models": models})()
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return models


def test_retries_dropped_connection(monkeypatch):
    models = _patch(monkeypatch, fails=2)
    assert llm.generate_structured("p", Out).x == 1
    assert models.calls == 3


def test_gives_up_with_clear_message(monkeypatch):
    _patch(monkeypatch, fails=99)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        llm.generate_structured("p", Out)
