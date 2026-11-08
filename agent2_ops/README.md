# Agent 2 - Ops & Assignment

Runs **after Agent 1**. Takes `state["scope"]` and produces `state["sprint_plan"]` + `state["ops_result"]`.

| step | file | what it does |
|------|------|--------------|
| 1 | `planner.py` | packs stories into sprints (greedy, by point capacity) |
| 2 | `matcher.py` | **M4**: story <-> member similarity (sentence-transformers) + Hungarian algorithm + load balancing; low confidence => no assignee (human decides) |
| 3 | `agent.py`   | **approval gate**, writes board / sprint lists / cards to Trello |
| - | `trello.py`  | Trello REST calls, exponential backoff (429 / 5xx), call counters |
| - | `audit.py`   | one JSON line per write -> `agent2_ops/audit_log.jsonl` |
| - | `team.json`  | **sample/fallback team only**. The real team is sent in the request (`team_members`) |
| - | `demo.py`    | CLI: plan -> answer Agent 1 questions -> review -> assign leftovers -> approve |

## Two modes (decided only by `state["approved"]`)
* `POST /plan`    -> **proposal**. Nothing is written to Trello.
* `POST /approve` -> human approval event. The same plan is written to Trello.
  Body: `scope` (from /plan, may be edited), `board_name` (same as /plan), `approved_by`,
  optional `team_members` (same as /plan).

## Trello credentials (needed only for /approve)
1. https://trello.com/power-ups/admin -> **New** -> fill the form with anything valid.
2. Open the Power-Up -> **API Key** tab -> **Generate a new API Key** -> copy it.
3. Open (with your key): `https://trello.com/1/authorize?expiration=never&scope=read,write&response_type=token&name=PlanPilot&key=YOUR_KEY`
   -> **Allow** -> copy the token.
4. `.env`: `TRELLO_API_KEY=...` and `TRELLO_API_TOKEN=...`
5. Test: `curl "https://api.trello.com/1/members/me?key=KEY&token=TOKEN"`

## Output for Agent 3 (`state["ops_result"]`, schema `OpsResult` in `orchestrator/schemas.py`)
* `board_id`, `board_url`, `written` (True only after /approve)
* `tickets[]`: `key, title, epic, points, sprint, assignee, confidence, needs_human, status, card_id, url`
* `state["sprint_plan"]` keeps the original contract (`sprints[].number/items/capacity_points`).

## Guarantees (from the proposal)
* No silent writes: every write is in `audit_log.jsonl` (who / what / when / agent / model version).
  Note: failed Trello writes surface as `errors` in state but are not currently logged to the audit file.
* Hard approval gate: `write_to_trello()` refuses to run unless approved; only `/approve` sets it.
* Idempotent: card key = hash(board name + story title), stored in the card description
  (`planpilot-key: ...`). Re-running /approve never creates duplicates.
* Plan validated with Pydantic (`Scope`) before use.
* Low-confidence assignments are routed to a human (`needs_human = true`).
* Rate limits: exponential backoff (1s, 2s, 4s, 8s, 16s).

## Status vs. the proposal - what is NOT done (be honest in the report)
| Item (proposal) | Status |
|---|---|
| Jira connector | **Not done.** Trello only (on purpose, to keep it simple). `trello.py` is the template for a `jira.py`. |
| M4 contrastive fine-tuning | **Not done.** Pre-fine-tune embeddings are used (this is the proposal's own baseline). Needs training pairs (task, correct assignee). Plug the result in with `SKILL_MODEL=<path>`. |
| M4 baselines in MLflow (random / round-robin / pre-fine-tune) | **Not done.** Needs a labelled set of (task, assignee) pairs, which does not exist yet. |
| Write batching | **Not done.** A plan is ~10-40 cards; backoff is enough at this scale. |
| Real "current load" from Trello | **Not done.** Load comes from `current_load` in the team input. |
| Review UI | **Not done** (Person B's week-7 item). `/docs` and `demo.py` stand in for it. |
| Feedback capture into PostgreSQL | **Not done.** Overrides go to the audit log file only. |
| Grafana dashboard / Prometheus export | **Not done.** No `ops_result.metrics` field exists yet; wiring is week 8. |
| Dockerfile for the API | **Not done.** CI (GitHub Actions) exists and runs the tests. |
| Authentication on /approve | **Not done.** Anyone who can reach the API can approve. Fine for the course demo, not for production. |
| Tests against real Trello / real embedding model | Trello is tested with a fake; the real calls were verified manually. |
