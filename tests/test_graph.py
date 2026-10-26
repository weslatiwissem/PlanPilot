from orchestrator.graph import build_graph


def test_graph_runs_end_to_end():
    result = build_graph().invoke({"project_description": "Build a todo app"})
    assert "scope" in result
    assert "sprint_plan" in result
    assert "risks" in result
