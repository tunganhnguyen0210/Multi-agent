from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_runs_research_to_done() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    result = MultiAgentWorkflow().run(state)
    assert result.research_notes is not None
    assert result.analysis_notes is not None
    assert result.final_answer is not None
    assert result.route_history[-1] == "done"


def test_workflow_records_trace_events() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = MultiAgentWorkflow().run(state)
    trace_names = [event["name"] for event in result.trace]
    assert "supervisor_route" in trace_names
    assert "agent_completed" in trace_names
