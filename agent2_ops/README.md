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
| - | `judge.py`   | **LLM-as-a-judge**: Gemini grades every assignment (1-5) and names who it would pick |
| - | `evaluate_matcher.py` | M4 vs keywords / round-robin / random, same judge, logged to MLflow |

## Which model? (M4)
`sentence-transformers/all-MiniLM-L6-v2`, pretrained (= the proposal's "pre-fine-tune" baseline), plus the
Hungarian algorithm (SciPy) with a load penalty and a capacity rule. Falls back to keyword overlap if the model
cannot load. Every response says which one ran: `ops_result.model_version`.

## Review flow: the team lead edits every AI suggestion (`/steps/*`)
```
1. POST /steps/plan-sprints  AI: stories -> sprints (no people)         lead: move stories, change points
2. POST /steps/assign        M4: people INSIDE the lead's sprints       lead: change people, lock some
   (call again)              M4: keeps locked tickets, redistributes the rest around them
3. POST /steps/judge         optional: LLM judge grades the assignment
4. POST /steps/approve       Trello cards + review stats + every change in the audit log
```
* `proposed_sprint` / `proposed_assignee` remember what the AI suggested; the lead's version is `sprint` / `assignee`.
* `locked: true` = "the lead decided this one": M4 never changes it, and its points count as that person's load.
* `ops_result.workload`: per sprint and teammate, `points`, `utilisation_pct` (% of their capacity, >100 = overloaded)
  and `share_pct` (% of the sprint they carry). `ops_result.load_balance`: 1.0 = even.
* M4 never puts anyone over capacity while a teammate who fits still has room; if nobody fits, the story goes
  to the lead (`needs_human`).
* Send `model_version` (from `/assign`) back to `/approve` so the audit log keeps which model proposed the plan.

## Metrics (not the confidence score)
| metric | where | meaning |
|---|---|---|
| judge mean score (1-5), agreement, good rate | `/steps/judge`, `evaluate_matcher.py` | independent LLM grades who got what |
| human acceptance rate (sprints, assignees) | `ops_result.review` after `/approve`, audit log | share of the AI's proposal the lead kept |
| coverage | `evaluate_matcher.py` | share of stories M4 dared to assign |
| load balance | every response, `evaluate_matcher.py` | least / most utilised teammate |
Baselines on the same sprints and judge: keywords, round-robin, random (`python -m agent2_ops.evaluate_matcher --mlflow`).

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
| M4 baselines in MLflow (random / round-robin / pre-fine-tune) | **Done** with an LLM judge instead of labels: `evaluate_matcher.py --mlflow`. A labelled set would still be better. |
| Write batching | **Not done.** A plan is ~10-40 cards; backoff is enough at this scale. |
| Real "current load" from Trello | **Not done.** Load comes from `current_load` in the team input. |
| Review UI | **API done** (`/steps/plan-sprints`, `/steps/assign`, `/steps/judge`); the Streamlit screens are next. `smoke_review.py` walks the flow. |
| Feedback capture into PostgreSQL | **Not done.** Every override (proposed vs final sprint/assignee) goes to the audit log file. |
| Grafana dashboard / Prometheus export | **Not done.** No `ops_result.metrics` field exists yet; wiring is week 8. |
| Dockerfile for the API | **Not done.** CI (GitHub Actions) exists and runs the tests. |
| Authentication on /approve | **Not done.** Anyone who can reach the API can approve. Fine for the course demo, not for production. |
| Tests against real Trello / real embedding model | Trello is tested with a fake; the real calls were verified manually. |
