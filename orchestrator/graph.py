"""LangGraph state machine: scoping -> sprint planning -> risk assessment."""
from langgraph.graph import END, START, StateGraph

from agent1_scoping.agent import scope_project
from agent2_ops.agent import plan_sprints
from agent3_risk.agent import assess_risk
from orchestrator.state import PlanState


def build_graph():
    g = StateGraph(PlanState)
    g.add_node("scoping", scope_project)
    g.add_node("sprint_planning", plan_sprints)
    g.add_node("risk", assess_risk)
    g.add_edge(START, "scoping")
    g.add_edge("scoping", "sprint_planning")
    g.add_edge("sprint_planning", "risk")
    g.add_edge("risk", END)
    return g.compile()
