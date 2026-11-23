"""Evaluate M4 (who gets which story) against baselines, with an LLM judge. Logs to MLflow.

    python -m agent2_ops.evaluate_matcher --no-judge          # quick: coverage + balance only, no API key
    python -m agent2_ops.evaluate_matcher                     # + LLM judge (needs GEMINI_API_KEY)
    python -m agent2_ops.evaluate_matcher --mlflow            # ... and log every strategy as an MLflow run
    python -m agent2_ops.evaluate_matcher --scopes my_scopes/ --team my_team.json

Strategies, all given the SAME sprints (Agent 2's packing) and graded by the SAME judge, which never learns
which strategy it is grading:
    m4           sentence-transformers similarity + Hungarian + load balancing (what Agent 2 uses)
    keywords     word overlap + Hungarian (M4's fallback)
    round_robin  stories handed out in turn (a team without any tool)
    random       random teammate (lower bound)

Metrics per strategy:
    judge_mean_score   1-5, how well the assignee fits (LLM judge)
    judge_agreement    share of stories where the judge would pick the same person
    judge_good_rate    share of stories scored 4-5
    coverage           share of stories that got an assignee (M4 leaves weak matches to a human)
    load_balance       least / most utilised teammate per sprint (1.0 = even)

Input scopes: every JSON file in --scopes holding either a scope ({"epics": ...}) or an API response with a
"scope" key. The default folder has two real Agent 1 outputs; add more (e.g. synthetic ones) for a bigger test.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

from agent2_ops.agent import TEAM_FILE, assign_tickets, sprint_capacity
from agent2_ops.planner import flatten_stories, load_balance, sprint_tickets, workload
from orchestrator.schemas import Member, Scope, Ticket

STRATEGIES = ["m4", "keywords", "round_robin", "random"]
EXPERIMENT = "planpilot-m4-matcher"


def load_scopes(folder: str) -> List[Scope]:
    scopes = []
    for f in sorted(Path(folder).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        data = data.get("scope", data)
        if isinstance(data, dict) and data.get("epics"):
            scopes.append(Scope.model_validate(data))
    if not scopes:
        raise SystemExit(f"No scope JSON files found in {folder}")
    return scopes


def run_strategy(name: str, tickets: List[Ticket], stories: List[Dict], members: List[Member], seed: int) -> str:
    """Fill the tickets' assignees with one strategy. Returns a version label."""
    if name in ("m4", "keywords"):
        os.environ["SKILL_MATCHER"] = "embeddings" if name == "m4" else "keywords"
        return assign_tickets(tickets, stories, members)
    rng = random.Random(seed)
    for i, t in enumerate(tickets):
        m = members[i % len(members)] if name == "round_robin" else rng.choice(members)
        t.assignee = t.proposed_assignee = m.name
    return name


def evaluate(scopes: List[Scope], members: List[Member], use_judge: bool, seed: int = 42) -> Dict[str, Dict]:
    results = {}
    original = os.environ.get("SKILL_MATCHER")
    for name in STRATEGIES:
        all_tickets, verdicts, balances, version = [], [], [], ""
        for i, scope in enumerate(scopes):
            stories = flatten_stories(scope)
            tickets = sprint_tickets(stories, sprint_capacity(members), f"eval-{i}")
            version = run_strategy(name, tickets, stories, members, seed + i)
            all_tickets += tickets
            balance = load_balance(workload(tickets, members))
            if balance is not None:
                balances.append(balance)
            if use_judge:
                from agent2_ops.judge import judge_tickets  # lazy: needs the Gemini SDK

                verdicts += judge_tickets(tickets, stories, members).verdicts
        n = len(all_tickets)
        m = {
            "coverage": round(sum(t.assignee is not None for t in all_tickets) / n, 3),
            "load_balance": round(sum(balances) / len(balances), 3) if balances else 0.0,
            "n_stories": n,
        }
        if verdicts:
            k = len(verdicts)
            m.update(judge_mean_score=round(sum(v.score for v in verdicts) / k, 3),
                     judge_agreement=round(sum(v.agrees for v in verdicts) / k, 3),
                     judge_good_rate=round(sum(v.score >= 4 for v in verdicts) / k, 3))
        results[name] = {"metrics": m, "version": version}
    if original is None:
        os.environ.pop("SKILL_MATCHER", None)
    else:
        os.environ["SKILL_MATCHER"] = original
    return results


def log_to_mlflow(results: Dict[str, Dict], n_scopes: int, use_judge: bool) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    for name, r in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"strategy": name, "matcher_version": r["version"], "n_scopes": n_scopes,
                               "judge_model": os.getenv("JUDGE_MODEL", "gemini-3.8-flash") if use_judge else "none"})
            mlflow.log_metrics(r["metrics"])
    print(f"\nLogged {len(results)} runs to MLflow experiment '{EXPERIMENT}'.")


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--scopes", default="tests/fixtures", help="folder of scope JSON files")
    p.add_argument("--team", default=TEAM_FILE, help="team JSON file (same format as agent2_ops/team.json)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (no API key needed)")
    p.add_argument("--mlflow", action="store_true", help="log every strategy as an MLflow run")
    args = p.parse_args(argv)

    scopes = load_scopes(args.scopes)
    members = [Member.model_validate(m) for m in json.loads(Path(args.team).read_text(encoding="utf-8"))]
    results = evaluate(scopes, members, use_judge=not args.no_judge)

    cols = ["judge_mean_score", "judge_agreement", "judge_good_rate", "coverage", "load_balance"]
    print(f"\n{len(scopes)} scopes, {results['m4']['metrics']['n_stories']} stories, team of {len(members)}\n")
    print(f"{'strategy':<13}" + "".join(f"{c:>18}" for c in cols))
    for name, r in results.items():
        print(f"{name:<13}" + "".join(f"{r['metrics'].get(c, float('nan')):>18.3f}" for c in cols))
    if args.mlflow:
        log_to_mlflow(results, len(scopes), not args.no_judge)


if __name__ == "__main__":
    main()
