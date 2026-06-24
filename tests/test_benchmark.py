from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_benchmark_returns_metrics() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        return MultiAgentWorkflow().run(state)

    state, metrics = run_benchmark("multi-agent", "Explain multi-agent systems", runner)
    assert state.final_answer is not None
    assert metrics.run_name == "multi-agent"
    assert metrics.latency_seconds >= 0
    assert "routes=" in metrics.notes
