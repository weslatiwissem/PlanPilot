"""FastAPI gateway. Run: uvicorn orchestrator.main:app --reload

POST /plan     scoping -> sprint PROPOSAL -> risk   (nothing is written to Trello)
POST /approve  human approval event: writes the (possibly edited) scope to Trello,
               then runs Agent 3.  <- the only way Agent 2 is allowed to write.
/steps/*       step-by-step API for the UI: scope -> sprints -> risks -> approve (orchestrator/steps.py)
"""
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent2_ops.agent import plan_sprints
from agent3_risk.agent import assess_risk
from orchestrator.graph import build_graph
from orchestrator.steps import router as steps_router

load_dotenv()
app = FastAPI(title="PlanPilot", separate_input_output_schemas=False)  # one schema per model in /openapi.json
app.add_middleware(  # lets the browser UI (Vite dev server by default) call this API
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(steps_router)  # step-by-step endpoints for the wizard UI (see orchestrator/steps.py)
graph = build_graph()


class PlanRequest(BaseModel):
    project_description: str
    team_id: Optional[str] = None
    board_name: Optional[str] = None                   # Trello board name (Agent 2)
    team_members: Optional[List[Dict[str, Any]]] = None  # default: agent2_ops/team.json


class ApproveRequest(BaseModel):
    scope: Dict[str, Any]          # the scope returned by /plan (human may have edited it)
    board_name: str = "PlanPilot Project"  # use the SAME name as in /plan
    approved_by: str = "unknown"   # goes into the audit log
    team_members: Optional[List[Dict[str, Any]]] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan")
def create_plan(req: PlanRequest):
    state = {"project_description": req.project_description}
    if req.team_id:
        state["team_id"] = req.team_id
    if req.board_name:
        state["board_name"] = req.board_name
    if req.team_members:
        state["team_members"] = req.team_members
    return graph.invoke(state)  # `approved` is never set here -> Agent 2 only proposes


@app.post("/approve")
def approve_plan(req: ApproveRequest):
    state = {
        "scope": req.scope,
        "board_name": req.board_name,
        "approved": True,
        "approved_by": req.approved_by,
    }
    if req.team_members:
        state["team_members"] = req.team_members
    state.update(plan_sprints(state))  # Agent 2: now allowed to write
    if not state.get("errors"):
        state.update(assess_risk(state))  # Agent 3 (stub for now)
    return state
