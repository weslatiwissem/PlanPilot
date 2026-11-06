"""LLM-facing output models for Agent 1.

These are deliberately simple (no defaults, no constraints) so Gemini's JSON-schema
mode handles them reliably. Everything is converted to the team contract
(orchestrator/schemas.py) before leaving the agent.
"""
from typing import List

from pydantic import BaseModel, field_validator

FIBONACCI_POINTS = (1, 2, 3, 5, 8, 13)


class ClarificationCheck(BaseModel):
    missing_info: List[str]  # checklist items that are NOT covered by the description
    questions: List[str]  # at most a few questions; empty list = enough information


class LLMStory(BaseModel):
    title: str
    description: str
    estimate_points: int

    @field_validator("estimate_points")
    @classmethod
    def snap_to_fibonacci(cls, v: int) -> int:
        return min(FIBONACCI_POINTS, key=lambda f: abs(f - v))


class LLMEpic(BaseModel):
    title: str
    stories: List[LLMStory]


class LLMScope(BaseModel):
    epics: List[LLMEpic]
