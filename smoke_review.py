"""Walk Agent 2's review flow the way the team lead will, against a running server.

    uvicorn orchestrator.main:app --host 127.0.0.1 --port 8765        # terminal 1
    python smoke_review.py                                             # terminal 2: stages 1-2, nothing written
    python smoke_review.py --judge                                     # + LLM judge (needs GEMINI_API_KEY)
    python smoke_review.py --approve "PlanPilot Test"                  # + write the cards to that Trello board

Starts from a saved Agent 1 scope (no Gemini needed): --scope to use another file.
What it does, printing the result of each stage:
  1. /steps/plan-sprints   AI proposes sprints              -> the lead moves the LAST story to a new sprint
  2. /steps/assign         M4 assigns inside those sprints  -> the lead gives the first story to someone else, locks it
  3. /steps/assign again   M4 redistributes around the lock
  4. /steps/judge          (optional) LLM grades every assignment
  5. /steps/approve        (optional) Trello + how much of the AI's proposal the lead kept
"""
import argparse
import json
import sys

import requests


def call(base, path, body):
    try:
        r = requests.post(f"{base}{path}", json=body, timeout=600)
    except requests.exceptions.ConnectionError:
        sys.exit(f"Cannot reach {base}. Start the server first:\n"
                 "    uvicorn orchestrator.main:app --host 127.0.0.1 --port 8765")
    if r.status_code != 200:
        sys.exit(f"{path} -> HTTP {r.status_code}\n{json.dumps(r.json(), indent=2)}")
    return r.json()


def show_workload(ops):
    print("   workload (points | % of capacity | % of the sprint):")
    for w in ops["workload"]:
        flag = "  <-- over capacity" if w["utilisation_pct"] > 100 else ""
        print(f"     sprint {w['sprint']}  {w['member']:<11} {w['points']:>3} pts | {w['utilisation_pct']:>4}% "
              f"| {w['share_pct']:>3}%{flag}")
    print(f"   load balance: {ops['load_balance']}  (1.0 = even)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8765")
    p.add_argument("--scope", default="tests/fixtures/sample_plan_response.json")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--approve", metavar="BOARD", help="write to this Trello board (use a throwaway name)")
    args = p.parse_args()

    data = json.load(open(args.scope, encoding="utf-8"))
    scope = data.get("scope", data)
    board = args.approve or "PlanPilot Review Demo"

    print("\n1) /steps/plan-sprints: the AI proposes sprints (no people yet)")
    s1 = call(args.base, "/steps/plan-sprints", {"scope": scope, "board_name": board})
    for load in s1["sprint_load"]:
        print(f"   sprint {load['sprint']}: {load['points']}/{load['capacity_points']} pts ({load['utilisation_pct']}%)")
    tickets = s1["ops_result"]["tickets"]
    last = tickets[-1]
    last["sprint"] = max(t["sprint"] for t in tickets) + 1
    print(f"   LEAD EDIT: moves '{last['title']}' to sprint {last['sprint']}")

    print("\n2) /steps/assign: M4 assigns people inside the lead's sprints")
    s2 = call(args.base, "/steps/assign", {"scope": scope, "tickets": tickets, "board_name": board})
    ops = s2["ops_result"]
    print(f"   model: {ops['model_version']}")
    for t in ops["tickets"]:
        print(f"   sprint {t['sprint']}  {t['title'][:45]:<45} -> {t['assignee'] or 'NEEDS HUMAN'} "
              f"(confidence {t['confidence']:.2f})")
    show_workload(ops)

    tickets = ops["tickets"]
    first = tickets[0]
    team = sorted({w["member"] for w in ops["workload"] if w["member"] != "Unassigned"})
    other = next(m for m in team if m != first["assignee"])
    first.update(assignee=other, locked=True, confidence=1.0)
    print(f"\n   LEAD EDIT: gives '{first['title']}' to {other} and locks it")

    print("\n3) /steps/assign again: M4 redistributes the rest around the lock")
    s3 = call(args.base, "/steps/assign", {"scope": scope, "tickets": tickets, "board_name": board})
    ops = s3["ops_result"]
    for t in ops["tickets"]:
        moved = "" if t["assignee"] == next(x for x in tickets if x["title"] == t["title"])["assignee"] else "  (changed)"
        print(f"   sprint {t['sprint']}  {t['title'][:45]:<45} -> {t['assignee'] or 'NEEDS HUMAN'}"
              f"{' [locked]' if t['locked'] else ''}{moved}")
    show_workload(ops)
    print(f"   risks: {[r['reason'] for r in s3['risks']][:3]}")

    if args.judge:
        print("\n4) /steps/judge: LLM-as-a-judge")
        j = call(args.base, "/steps/judge", {"scope": scope, "tickets": ops["tickets"], "board_name": board})
        print(f"   judge {j['model']}: mean {j['mean_score']}/5, agrees {j['agreement_rate']:.0%}, "
              f"good (4-5) {j['good_rate']:.0%}")
        for v in j["verdicts"]:
            if not v["agrees"]:
                print(f"   - {v['title'][:40]}: {v['assignee']} -> judge prefers {v['best_member']} ({v['reason']})")

    if args.approve:
        print(f"\n5) /steps/approve: writing to Trello board '{board}'")
        s5 = call(args.base, "/steps/approve", {"scope": scope, "tickets": ops["tickets"], "board_name": board,
                                                "approved_by": "smoke-test", "model_version": ops["model_version"]})
        r = s5["ops_result"]
        print(f"   board: {r['board_url']}\n   review: {r['review']}\n   model logged: {r['model_version']}")
    else:
        print("\nNothing was written. Add --approve \"Board name\" to send it to Trello.")


if __name__ == "__main__":
    main()
