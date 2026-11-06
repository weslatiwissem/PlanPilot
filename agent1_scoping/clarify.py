"""Missing-information detection (the 'M1' step)."""
from agent1_scoping.llm import generate_structured
from agent1_scoping.models import ClarificationCheck
from agent1_scoping.prompts import CHECKLIST, CLARIFY_PROMPT, SYSTEM

MAX_QUESTIONS = 4


def check_missing_info(description: str) -> ClarificationCheck:
    prompt = CLARIFY_PROMPT.format(
        description=description,
        checklist="\n".join(f"- {c}" for c in CHECKLIST),
    )
    result = generate_structured(prompt, ClarificationCheck, system=SYSTEM)
    result.questions = [q.strip() for q in result.questions if q.strip()][:MAX_QUESTIONS]
    return result
