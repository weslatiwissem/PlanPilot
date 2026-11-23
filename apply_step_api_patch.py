"""Apply the step-API changes to Agent 2, the shared state and main.py.

Run once from the repo root:   python apply_step_api_patch.py

It edits agent2_ops/planner.py, agent2_ops/agent.py, orchestrator/state.py and orchestrator/main.py with
exact-text replacements. If any anchor is missing (a file changed since this was written) or the patch is
already applied, it stops and changes NOTHING. Review with `git diff`; undo with `git checkout -- <file>`.

What changes, in one line each:
  planner.py  + check_tickets / edited_tickets / plan_from_tickets (the plan is derived from the user's tickets)
  agent.py    + sprint_capacity(); plan_sprints keeps tickets passed in state["tickets"] instead of re-planning
  state.py    + tickets
  main.py     + CORS for the browser UI, + the /steps router
"""
import sys
from pathlib import Path


def read(path):
    raw = Path(path).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write(path, text, crlf):
    Path(path).write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def sub(text, old, new, what, optional=False):
    n = text.count(old)
    if n != 1:
        if optional and n == 0:
            return text
        sys.exit(f"STOP: expected exactly one match for {what}, found {n}. Nothing was changed.")
    return text.replace(old, new)


PLANNER_ADDITIONS = '''

def check_tickets(tickets: List[Ticket], stories: List[Dict]) -> None:
    """Edited tickets must match the scope one to one. Raises ValueError with a message for the user."""
    titles = [t.title for t in tickets]
    duplicated = sorted({x for x in titles if titles.count(x) > 1})
    if duplicated:
        raise ValueError(f"more than one ticket for: {', '.join(duplicated)}")
    expected = [s["title"] for s in stories]
    missing = [x for x in expected if x not in titles]
    unknown = [x for x in titles if x not in expected]
    if missing:
        raise ValueError(f"no ticket for: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"ticket does not match any story: {', '.join(unknown)}")
    bad = [t.title for t in tickets if t.sprint < 1 or t.points < 1]
    if bad:
        raise ValueError(f"sprint and points must be at least 1: {', '.join(bad)}")


def edited_tickets(raw: List[Dict], stories: List[Dict], project_id: str) -> List[Ticket]:
    """Tickets as the user left them (sprint, points, assignee), made consistent with the scope.

    Keys, epics and states are recomputed here and never trusted from the client. A ticket needs a human
    exactly when nobody is assigned.
    """
    tickets = [Ticket.model_validate(t) for t in raw]
    check_tickets(tickets, stories)
    epic_of = {s["title"]: s["epic"] for s in stories}
    order = {s["title"]: i for i, s in enumerate(stories)}
    for t in tickets:
        t.key = ticket_key(project_id, t.title)
        t.epic = epic_of[t.title]
        t.needs_human = t.assignee is None
        t.status, t.card_id, t.url = "proposed", None, None
    return sorted(tickets, key=lambda t: order[t.title])


def plan_from_tickets(tickets: List[Ticket], capacity: int) -> SprintPlan:
    """The plan implied by the tickets: one sprint per sprint number used, stories in scope order."""
    by_sprint: Dict[int, List[str]] = {}
    for t in tickets:
        by_sprint.setdefault(t.sprint, []).append(t.title)
    return SprintPlan(
        sprints=[Sprint(number=n, items=items, capacity_points=capacity) for n, items in sorted(by_sprint.items())]
    )
'''

SPRINT_CAPACITY = '''def sprint_capacity(members: List[Member]) -> int:
    """Points per sprint: the SPRINT_CAPACITY override, else the team's combined capacity."""
    return max(int(os.getenv("SPRINT_CAPACITY", sum(m.capacity_points for m in members))), 1)


'''

OLD_CAPACITY = (
    "        # 1) sprints: capacity = env override, else the team's combined capacity\n"
    '        capacity = int(os.getenv("SPRINT_CAPACITY", sum(m.capacity_points for m in members)))\n'
    "        capacity = max(capacity, 1)\n"
    "        plan, sprint_of = pack_sprints(stories, capacity)\n"
)
PROJECT_LINE = '        project_id = state.get("board_name") or "PlanPilot Project"\n'
OUT_LINE = '        out = {"sprint_plan": plan.model_dump()}\n'

# ---------------------------------------------------------------- read everything first
planner, planner_crlf = read("agent2_ops/planner.py")
agent, agent_crlf = read("agent2_ops/agent.py")
state, state_crlf = read("orchestrator/state.py")
main, main_crlf = read("orchestrator/main.py")
if "def edited_tickets" in planner or "def sprint_capacity" in agent:
    sys.exit("STOP: the step-API patch seems to be applied already. Nothing was changed.")

# ---------------------------------------------------------------- planner.py
planner = sub(planner, "from orchestrator.schemas import Scope, Sprint, SprintPlan\n",
              "from orchestrator.schemas import Scope, Sprint, SprintPlan, Ticket\n", "planner.py schemas import")
planner = planner.rstrip("\n") + "\n" + PLANNER_ADDITIONS

# ---------------------------------------------------------------- agent.py
agent = sub(agent, "from agent2_ops.planner import flatten_stories, pack_sprints, ticket_key\n",
            "from agent2_ops.planner import (\n    edited_tickets,\n    flatten_stories,\n    pack_sprints,\n"
            "    plan_from_tickets,\n    ticket_key,\n)\n", "agent.py planner import")
agent = sub(agent, "def _card_text(", SPRINT_CAPACITY + "def _card_text(", "agent.py _card_text")
start = agent.find(OLD_CAPACITY)
end = agent.find(OUT_LINE)
if start < 0 or end < start or agent.count(OLD_CAPACITY) != 1 or agent.count(OUT_LINE) != 1:
    sys.exit("STOP: plan_sprints does not look like the version this patch was written for. Nothing was changed.")
region = agent[start:end]
region = region.replace(OLD_CAPACITY, "        # 1) sprints: greedy fill up to the capacity\n        plan, sprint_of = pack_sprints(stories, capacity)\n")
region = sub(region, PROJECT_LINE, "", "project_id line inside plan_sprints")
indented = "".join(("    " + ln if ln.strip() else ln) for ln in region.splitlines(keepends=True))
header = (
    PROJECT_LINE
    + "        capacity = sprint_capacity(members)\n\n"
    "        if state.get(\"tickets\"):\n"
    "            # a plan edited by the user (step API): keep sprints, points and assignees exactly as left\n"
    "            tickets = edited_tickets(state[\"tickets\"], stories, project_id)\n"
    "            plan, version = plan_from_tickets(tickets, capacity), \"human-edited\"\n"
    "        else:\n"
)
agent = agent[:start] + header + indented + agent[end:]

# ---------------------------------------------------------------- state.py
lines = state.splitlines(keepends=True)
idx = [i for i, ln in enumerate(lines) if ln.startswith("    ops_result:")]
if len(idx) != 1:
    sys.exit("STOP: could not find the ops_result line in state.py. Nothing was changed.")
lines.insert(idx[0] + 1, "    # --- added for the step API ----------------------------------------------------\n"
                         "    tickets: List[Dict[str, Any]]       # optional input: user-edited tickets -> list[schemas.Ticket]\n")
state = "".join(lines)

# ---------------------------------------------------------------- main.py
main = sub(main, "from typing import Any, Dict, List, Optional\n", "import os\nfrom typing import Any, Dict, List, Optional\n", "main.py typing import")
main = sub(main, "from fastapi import FastAPI\n", "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n", "main.py fastapi import")
main = sub(main, "from orchestrator.graph import build_graph\n",
           "from orchestrator.graph import build_graph\nfrom orchestrator.steps import router as steps_router\n", "main.py graph import")
main = sub(main, 'app = FastAPI(title="PlanPilot")\n',
           'app = FastAPI(title="PlanPilot")\n'
           "app.add_middleware(  # lets the browser UI (Vite dev server by default) call this API\n"
           "    CORSMiddleware,\n"
           '    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")],\n'
           '    allow_methods=["*"],\n'
           '    allow_headers=["*"],\n'
           ")\n"
           "app.include_router(steps_router)  # step-by-step endpoints for the wizard UI (see orchestrator/steps.py)\n",
           "main.py app creation")
main = sub(main, "then runs Agent 3.  <- the only way Agent 2 is allowed to write.\n",
           "then runs Agent 3.  <- the only way Agent 2 is allowed to write.\n"
           "/steps/*       step-by-step API for the UI: scope -> sprints -> risks -> approve (orchestrator/steps.py)\n",
           "main.py docstring", optional=True)

# ---------------------------------------------------------------- write (only reached if every anchor matched)
write("agent2_ops/planner.py", planner, planner_crlf)
write("agent2_ops/agent.py", agent, agent_crlf)
write("orchestrator/state.py", state, state_crlf)
write("orchestrator/main.py", main, main_crlf)
print("patched: agent2_ops/planner.py, agent2_ops/agent.py, orchestrator/state.py, orchestrator/main.py")
print("next: put orchestrator/steps.py in place, then run: ruff check .  and  pytest")
