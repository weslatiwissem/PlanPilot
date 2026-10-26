"""FastAPI gateway. Run: uvicorn orchestrator.main:app --reload"""
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.graph import build_graph

load_dotenv()
app = FastAPI(title="PlanPilot")
graph = build_graph()


class PlanRequest(BaseModel):
    project_description: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan")
def create_plan(req: PlanRequest):
    return graph.invoke({"project_description": req.project_description})
