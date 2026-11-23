"""Walk through the step API the way the UI will, with an edit in the middle.

    uvicorn orchestrator.main:app --port 8002          # in another terminal
    python smoke_steps.py                              # scope -> sprints -> edit -> risks (nothing is written)
    python smoke_steps.py --approve "PlanPilot Test"   # ...then write the cards to that Trello board
    python smoke_steps.py --past 10:8,12:6 --sprint-days 14   # give past sprints (committed:delivered, oldest first)

No hand-written JSON needed. Use a throwaway board name for --approve.
"""
import argparse
import json
import sys

import requests

DESCRIPTION = "A todo app with web UI, login and reminders for a small startup, team of 3 over 6 weeks."


def call(base: str, path: str, body: dict) -> dict:
    try:
        r = requests.post(f"{base}{path}", json=body, timeout=300)
    except requests.exceptions.ConnectionError:
        sys.exit(f"\nCannot reach the server at {base}.\nStart it in another terminal:\n"
                 "    uvicorn orchestrator.main:app --port 8002\n"
                 "then look at THAT terminal: an error at startup (a crash) is shown there, not here.")
    except requests.exceptions.ReadTimeout:
        sys.exit(f"\n{path} took longer than 300 seconds. Check the server terminal for errors.")
    if r.status_code != 200:
        print(f"\n{path} -> HTTP {r.status_code}\n{json.dumps(r.json(), indent=2)}")
        sys.exit(1)
    return r.json()


def show_forecasts(forecasts: list) -> None:
    if not forecasts:
        print("   no forecast (no trained model found: python -m agent3_risk.train_final --save)")
    for f in forecasts:
        ref = f"typical {f['reference_rate']:.0%}" if f.get("reference_rate") is not None else "typical n/a"
        print(f"   forecast sprint {f['sprint']}: {f['probability']:.0%} chance of delivering less than committed "
              f"({ref}; {f['model']}, {f['history_used']} past sprints)")


def show_plan(tickets: list, risks: list) -> None:
    for n in sorted({t["sprint"] for t in tickets}):
        mine = [t for t in tickets if t["sprint"] == n]
        who = sorted({t["assignee"] or "NOBODY" for t in mine})
        print(f"   sprint {n}: {len(mine)} stories, {sum(t['points'] for t in mine)} points, people: {', '.join(who)}")
    print(f"   {len(risks)} risks" + (":" if risks else ""))
    for r in risks[:6]:
        print(f"     {r['score']:.2f}  {r['item']}: {r['reason']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8002")
    p.add_argument("--description", default=DESCRIPTION)
    p.add_argument("--approve", metavar="BOARD", help="also write the cards to this Trello board")
    p.add_argument("--past", default="", help="finished sprints as committed:delivered, oldest first, e.g. 10:8,12:6")
    p.add_argument("--sprint-days", type=int, default=None)
    args = p.parse_args()
    board = args.approve or "PlanPilot Smoke Test"
    extra = {}
    if args.past:
        extra["past_sprints"] = [{"committed": float(a), "delivered": float(b)}
                                 for a, b in (x.split(":") for x in args.past.split(","))]
    if args.sprint_days:
        extra["sprint_days"] = args.sprint_days

    out = call(args.base, "/steps/scope", {"project_description": args.description, "skip_clarification": True})
    if out["status"] != "ok":
        print("1. scope: Agent 1 still has questions:", out["clarifying_questions"])
        sys.exit(1)
    scope = out["scope"]
    stories = [s for e in scope["epics"] for s in e["stories"]]
    print(f"1. scope: ok, {len(scope['epics'])} epics, {len(stories)} stories")

    out = call(args.base, "/steps/sprints", {"scope": scope, "board_name": board, **extra})
    tickets = out["ops_result"]["tickets"]
    print(f"2. sprints (proposal, nothing written: written={out['ops_result']['written']}):")
    show_plan(tickets, out["risks"])
    show_forecasts(out["forecasts"])

    last = tickets[-1]
    target = max(t["sprint"] for t in tickets) + 1 if len({t["sprint"] for t in tickets}) == 1 else 1
    old = (last["sprint"], last["assignee"])
    last.update(sprint=target, assignee="Bob", confidence=1.0)  # what the UI sends when the user edits
    print(f"3. edit: '{last['title']}' moved from sprint {old[0]} to sprint {target}, assignee {old[1]} -> Bob")

    out = call(args.base, "/steps/risks", {"scope": scope, "tickets": tickets, "board_name": board, **extra})
    print("4. risks after the edit (the edit must show up here):")
    plan = {s["number"]: s["items"] for s in out["sprint_plan"]["sprints"]}
    assert last["title"] in plan[target], "BUG: the edited story is not in its new sprint"
    show_plan(tickets, out["risks"])
    show_forecasts(out["forecasts"])

    if args.approve:
        out = call(args.base, "/steps/approve", {"scope": scope, "tickets": tickets, "board_name": board,
                                                 "approved_by": "smoke_steps", **extra})
        ops = out["ops_result"]
        print(f"5. approve: written={ops['written']}, board {ops.get('board_url')}")
        for t in ops["tickets"]:
            print(f"   {t['status']:<14} sprint {t['sprint']}  {t['assignee'] or 'NOBODY':<6} {t['title']}")
        moved = next(t for t in ops["tickets"] if t["title"] == last["title"])
        assert (moved["sprint"], moved["assignee"]) == (target, "Bob"), "BUG: the edit was lost at approval"
        print("   OK: the edited sprint and assignee are what was written")
    else:
        print("5. approve: skipped (add --approve \"Throwaway Board\" to write to Trello)")


if __name__ == "__main__":
    main()
