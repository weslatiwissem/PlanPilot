"""FastAPI gateway. Run: uvicorn orchestrator.main:app --reload"""
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.graph import build_graph
from typing import Optional

load_dotenv()
app = FastAPI(title="PlanPilot")
graph = build_graph()


class PlanRequest(BaseModel):
    project_description: str
    team_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan")
def create_plan(req: PlanRequest):
    state = {"project_description": req.project_description}
    if req.team_id:
        state["team_id"] = req.team_id
    return graph.invoke(state)
