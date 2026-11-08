"""Demo of the full flow against a running API.

    uvicorn orchestrator.main:app --port 8001 --reload   (terminal 1; 8000 is Chroma)
    python -m agent2_ops.demo "Build a todo app"         (terminal 2)

Steps: POST /plan -> (answer Agent 1's questions if it has any) -> show the proposal
       -> ask the human -> POST /approve.
"""
import sys

import requests

API = "http://localhost:8001"
BOARD = "PlanPilot Demo"
MARKER = "## Clarifications"  # once present, Agent 1 stops asking (see agent1_scoping/agent.py)

desc = sys.argv[1] if len(sys.argv) > 1 else "A todo app for students with a web UI, user login and reminders."

while True:
    state = requests.post(
        f"{API}/plan", json={"project_description": desc, "board_name": BOARD}, timeout=300
    ).json()
    questions = state.get("clarifying_questions")
    if not questions:
        break
    print("\nAgent 1 needs more info (press Enter to skip a question):")
    answers = [input(f"- {q}\n  > ").strip() or "no preference" for q in questions]
    desc += f"\n\n{MARKER}\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))

if state.get("errors"):
    sys.exit(f"Errors: {state['errors']}")

print("\nPROPOSED PLAN (nothing written yet)")
for t in state["ops_result"]["tickets"]:
    who = t["assignee"] or "-> HUMAN"
    print(f'  Sprint {t["sprint"]} | {t["title"]} ({t["points"]} pts) | {who} (conf {t["confidence"]})')

if input("\nApprove and create Trello cards? [y/N] ").lower() != "y":
    sys.exit("Not approved - nothing written.")

done = requests.post(
    f"{API}/approve",
    json={"scope": state["scope"], "board_name": BOARD, "approved_by": "demo-user"},
    timeout=300,
).json()
if done.get("errors"):
    sys.exit(f"Errors: {done['errors']}")
print("\nBoard:", done["ops_result"]["board_url"])
for t in done["ops_result"]["tickets"]:
    print(f'  {t["status"]:15} {t["title"]}')
