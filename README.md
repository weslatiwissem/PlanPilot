# PlanPilot

A self-learning AI copilot that turns a project description into an approved,
tracked Jira/Trello sprint plan.

**Short on time? Read `TEAMMATE_SETUP.md` — a copy-paste checklist with every command and every secret you need.**

This README is the onboarding doc. If you're a teammate setting up for the
first time, follow **Setup (step by step)** below top to bottom. It should
take ~20-30 minutes the first time, and ~5 minutes on any laptop after that.

---

## 0. Why the project is structured this way

Three people need to work off the exact same environment without "works on
my machine" problems. So:

- **Git** holds all code, configs, prompts, and Airflow DAGs. This is the
  only thing you clone.
- **Docker Compose** holds anything stateful (Postgres, MLflow tracking
  server, Chroma vector DB). Nobody installs these natively — you just run
  `docker compose up`.
- **DVC** holds datasets and trained model artifacts, pointed at one shared
  cloud remote. Git only stores small pointer files, not the actual data.
- **A single `requirements.txt`** (pinned versions) is the one source of
  truth for Python packages. No "pip install whatever" — if you need a new
  package, add it to the file and commit it.
- **`.env`** holds secrets (API keys, tokens). It is never committed — each
  person has their own copy, created from `.env.example`.

Folder ownership (matches the team split in the proposal):

```
planpilot/
├── agent1_scoping/      # Person A: M1, M2, M3, RAG, clarification dialog
├── agent2_ops/          # Person B: M4 skill matcher, sprint planning, Trello connector (see agent2_ops/README.md)
├── agent3_risk/         # Person C: M5, M6, Airflow DAGs, monitoring
├── orchestrator/        # shared: LangGraph state machine + FastAPI gateway
├── data/                # DVC-tracked, not committed to Git directly
├── dags/                # Airflow DAGs (added in week 8)
├── notebooks/           # exploration, not production code
├── docker-compose.yml   # Postgres + MLflow + Chroma
├── requirements.txt
├── .env.example
└── .dvc/                # created by `dvc init`
```

---

## 1. What you need installed (one-time, per laptop)

Works the same on Windows, macOS, and Linux. Install in this order:

### Step 1 — Git
- **Windows:** download from https://git-scm.com/download/win and run the installer (defaults are fine).
- **macOS:** run `git --version` in Terminal — it will prompt you to install Xcode Command Line Tools if missing. Or `brew install git`.
- **Linux:** `sudo apt install git` (Debian/Ubuntu) or your distro's equivalent.
- Verify: `git --version`

### Step 2 — Docker Desktop (gives you Docker + Docker Compose together)
- Download from https://www.docker.com/products/docker-desktop/
- Install, then **open Docker Desktop and leave it running** whenever you work on the project — Compose needs the Docker engine active.
- Windows: enable WSL2 backend if prompted (the installer does this automatically).
- Verify: `docker --version` and `docker compose version`

### Step 3 — Python 3.11
- Download from https://www.python.org/downloads/ (pick 3.11.x — not 3.12/3.13 yet, some ML libraries lag behind on new Python versions).
- **Windows:** during install, check "Add Python to PATH".
- **macOS/Linux:** consider `pyenv` if you need multiple Python versions later, but a plain install is fine to start.
- Verify: `python3 --version` (or `python --version` on Windows)

### Step 4 — DVC
Installed via pip in the next section (inside the virtual environment), so nothing to do here yet.

### Step 5 — A GitHub account + repo access
Whoever creates the repo adds the other two as collaborators (or you create it under a shared GitHub org). Everyone needs push access.

### Step 6 (optional but recommended) — VS Code
https://code.visualstudio.com/ + the Python and Docker extensions. Makes step 7 easier but isn't required.

---

## 2. First-time project setup (do this once per laptop)

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/planpilot.git
cd planpilot

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the env template and fill in your own secrets
cp .env.example .env
# open .env and fill in: GEMINI_API_KEY, TRELLO_API_KEY, TRELLO_API_TOKEN (Agent 1 + Agent 2 need these;
# JIRA_* is not used yet). See TEAMMATE_SETUP.md for how to get them.
# (ask whoever set up the Jira/Trello sandbox for these — do NOT commit .env)

# 5. Start the shared services (Postgres + MLflow + Chroma)
docker compose up -d

# 6. Confirm MLflow is reachable
# open http://localhost:5000 in your browser — you should see the MLflow UI

# 7. Set up DVC and pull the data
dvc pull
# first time only, whoever hasn't been added to the shared Google Drive
# folder should ask the DVC remote owner for access — see DVC section below

# 8. Run the test suite to confirm everything works
pytest
```

If `pytest` passes and `http://localhost:5000` loads, you're fully set up.

---

## 3. Daily workflow

```bash
source .venv/bin/activate     # every new terminal session
docker compose up -d          # if services aren't already running
git pull                      # get teammates' latest code
dvc pull                      # get any new/updated data or model files
```

When you're done for the session:

```bash
git add . && git commit -m "..." && git push
dvc push                      # if you added/changed any data or model artifacts
docker compose down           # optional — stops local services, saves battery
```

---

## 4. DVC remote setup (data & model versioning)

We use **Google Drive** as the DVC remote — free, and everyone already has
an account.

**One person (do this once for the team):**
```bash
dvc remote add -d storage gdrive://<shared-folder-id>
git add .dvc/config
git commit -m "configure DVC remote"
git push
```
Create a shared Google Drive folder, grab its ID from the URL, and share
edit access with the other two teammates' Google accounts.

**Everyone else:** just run `dvc pull` after cloning — the config is already
in Git. The first `dvc pull` will open a browser tab asking you to
authenticate with Google; do that once and it's cached.

When you add or change a dataset or trained model:
```bash
dvc add data/raw/some_dataset.csv
git add data/raw/some_dataset.csv.dvc data/.gitignore
git commit -m "add raw dataset"
dvc push
git push
```

---

## 5. MLflow, Airflow, Chroma — where they live

- **MLflow** (tracking + model registry) runs via `docker-compose.yml`,
  backed by Postgres, with artifacts stored in a local volume for now.
  UI: http://localhost:5000
- **Chroma** (RAG vector store for Agent 1) also runs via
  `docker-compose.yml`. Connect to it at `http://localhost:8000` from
  Python using the `chromadb` client.
- **Airflow** is intentionally *not* in `docker-compose.yml` yet — per the
  timeline it's introduced in week 8. When we get there, we'll add it as
  its own `docker-compose.airflow.yml` (Airflow's official Compose file is
  heavy, so keeping it separate means people who aren't touching DAGs don't
  need to run it day to day).

---

## 6. GPU note (for M2 fine-tuning and M5 training)

Most laptops won't have a usable GPU for fine-tuning the 3B-parameter SLM.
Per the proposal's risk mitigation: do QLoRA fine-tuning on **Google Colab
or Kaggle** (both give free GPU hours), pull the resulting model artifact
down via DVC, and everything else (LightGBM, DistilBERT inference, the
sequence model) runs fine on CPU locally.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `docker compose up` fails with a port conflict | Something else is using port 5000 or 8000 — stop it, or change the port mapping in `docker-compose.yml` |
| `dvc pull` asks to authenticate every time | Re-run `dvc remote modify storage gdrive_use_service_account false` then re-auth once |
| `POST /plan` returns `clarifying_questions` and no sprint plan | Not a bug: Agent 1 wants more detail. Use `python -m agent2_ops.demo` (it asks you the questions) or add the answers under a `## Clarifications` heading and re-send |
| `/approve` returns `agent2: Trello ... 401` | Wrong/expired `TRELLO_API_KEY` or `TRELLO_API_TOKEN`. Test: `curl "https://api.trello.com/1/members/me?key=KEY&token=TOKEN"`; restart uvicorn after editing `.env` |
| Browser shows `{"detail":"Not Found"}` on the API address | Normal — there is no page at `/`. Open `http://localhost:8001/docs` instead |
| `pip install -r requirements.txt` fails on `torch` | Install the CPU-only build manually first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then re-run the requirements install |
| Teammate's code works, yours doesn't | Check `python --version` and `pip list` match — if not, recreate your `.venv` from `requirements.txt` |

* * *

## 8. Running the pipeline (Agent 1 → Agent 2 → Agent 3) and Agent 2 / Trello

Flow: `POST /plan` (Agent 1 scopes, Agent 2 **proposes** sprints + assignees, nothing is written) → a human reviews →
`POST /approve` (Agent 2 writes the board / sprint lists / cards to Trello, then Agent 3 runs).

```
# terminal 1 — the API (port 8001, because Chroma in docker uses 8000)
uvicorn orchestrator.main:app --port 8001 --reload

# terminal 2 — guided demo: answers Agent 1's questions, shows the proposal, asks approval, writes Trello
python -m agent2_ops.demo "A todo app with web UI, login and reminders"

# or click through the API in the browser
# http://localhost:8001/docs
```

Notes:
- Docker is **not** needed to run the API: Chroma runs from the local `.chroma/` folder by default (`CHROMA_USE_SERVER=0`).
  Docker is only for MLflow / Postgres / the Chroma server.
- Trello credentials: see `agent2_ops/README.md` and `TEAMMATE_SETUP.md`.
- Optional RAG index for Agent 1: `dvc pull`, then `python -m agent1_scoping.index_stories`. Without it, plans are generated without RAG.
- Agent 2 tests (offline, no keys needed): `pytest tests/test_agent2.py`.
- Every Trello write is logged in `agent2_ops/audit_log.jsonl` (git-ignored).
- Agent 3 reads `state["ops_result"]` (board id, ticket list with Trello card ids, metrics). Contract: `orchestrator/schemas.py`.
