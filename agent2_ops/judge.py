"""LLM-as-a-judge for Agent 2: an independent model grades who was given which story.

Why: M4's confidence is its OWN cosine similarity, so it cannot tell us whether M4 is right. The judge is a
different model (Gemini) that reads each story and the skills of EVERY teammate, then says how good the
chosen person is (1-5) and whom it would have picked. It never sees which strategy made the assignment, so
the same judge can grade M4 and the baselines fairly (see evaluate_matcher.py).

    from agent2_ops.judge import judge_tickets
    report = judge_tickets(tickets, stories, members)   # needs GEMINI_API_KEY

Settings (.env, optional): JUDGE_MODEL (default gemini-3.8-flash: a stronger model than the one judged).
"""
import os
from typing import Dict, List

from pydantic import BaseModel, Field

from agent1_scoping.llm import generate_structured
from orchestrator.schemas import Member, Ticket

BATCH = 20  # stories per Gemini call (one call per batch, not per story: fewer requests, no rate limits)

SYSTEM = "You are a senior engineering manager reviewing task assignments. Be strict and consistent."

PROMPT = """Team (name: skills, free capacity in story points this sprint):
{team}

Assignments to review (one per line: id | story | assigned to):
{items}

For EACH id, return:
- score: 1-5, how well the assigned person fits the story (5 = the best person on this team for it,
  3 = acceptable, 1 = clearly the wrong person). Judge skills first, then whether the person has room.
- best_member: the teammate YOU would assign (may be the same person).
- reason: one short sentence.
Return exactly one verdict per id."""


class Verdict(BaseModel):
    id: int
    score: int
    best_member: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[Verdict]


class TicketVerdict(BaseModel):
    title: str
    assignee: str
    score: int = Field(ge=1, le=5)
    best_member: str
    agrees: bool          # the judge would have picked the same person
    reason: str


class JudgeReport(BaseModel):
    model: str
    judged: int = 0                   # assigned tickets that were graded (unassigned ones are skipped)
    mean_score: float | None = None   # 1-5
    agreement_rate: float | None = None   # share of tickets where the judge picks the same person
    good_rate: float | None = None    # share of tickets scored 4 or 5
    verdicts: List[TicketVerdict] = Field(default_factory=list)


def _team_text(members: List[Member]) -> str:
    return "\n".join(f"- {m.name}: {', '.join(m.skills) or 'no skills listed'} "
                     f"(free capacity {max(m.capacity_points - m.current_load, 0)})" for m in members)


def judge_tickets(tickets: List[Ticket], stories: List[Dict], members: List[Member]) -> JudgeReport:
    model = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
    text = {s["title"]: f'{s["epic"]} / {s["title"]}: {s["description"]}' for s in stories}
    todo = [t for t in tickets if t.assignee]
    names = {m.name for m in members}
    out: List[TicketVerdict] = []
    for start in range(0, len(todo), BATCH):
        part = todo[start:start + BATCH]
        items = "\n".join(f"{i} | {text.get(t.title, t.title)} | {t.assignee}" for i, t in enumerate(part))
        res = generate_structured(PROMPT.format(team=_team_text(members), items=items), Verdicts,
                                  system=SYSTEM, temperature=0.0, model=model)
        by_id = {v.id: v for v in res.verdicts}
        for i, t in enumerate(part):
            v = by_id.get(i)
            if v is None:  # the judge skipped one: do not invent a grade
                continue
            best = v.best_member if v.best_member in names else t.assignee
            out.append(TicketVerdict(title=t.title, assignee=t.assignee, score=min(max(v.score, 1), 5),
                                     best_member=best, agrees=best == t.assignee, reason=v.reason))
    if not out:
        return JudgeReport(model=model)
    n = len(out)
    return JudgeReport(
        model=model, judged=n,
        mean_score=round(sum(v.score for v in out) / n, 3),
        agreement_rate=round(sum(v.agrees for v in out) / n, 3),
        good_rate=round(sum(v.score >= 4 for v in out) / n, 3),
        verdicts=out,
    )
