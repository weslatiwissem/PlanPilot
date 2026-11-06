"""Plan generation (the 'M2' step): description (+ similar projects) -> Scope."""
from typing import List, Optional

from agent1_scoping.llm import generate_structured
from agent1_scoping.models import LLMScope
from agent1_scoping.prompts import SCOPE_PROMPT, SIMILAR_BLOCK, SYSTEM
from orchestrator.schemas import Scope


def _format_similar(similar: Optional[List[str]]) -> str:
    if not similar:
        return ""
    return SIMILAR_BLOCK.format(items="\n".join(f"- {s}" for s in similar))


def _make_titles_unique(scope: Scope) -> Scope:
    """Agent 2 references stories by title, so titles must be unique."""
    seen = set()
    for epic in scope.epics:
        for story in epic.stories:
            title, n = story.title.strip(), 2
            while title.lower() in seen:
                title = f"{story.title.strip()} ({n})"
                n += 1
            seen.add(title.lower())
            story.title = title
    return scope


def generate_scope(description: str, similar: Optional[List[str]] = None) -> Scope:
    prompt = SCOPE_PROMPT.format(description=description, similar=_format_similar(similar))
    out = generate_structured(prompt, LLMScope, system=SYSTEM)
    scope = Scope.model_validate(out.model_dump())  # enforce the team contract
    if not scope.epics:
        raise ValueError("Gemini returned an empty scope")
    return _make_titles_unique(scope)
