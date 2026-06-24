from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_researcher_populates_sources_and_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    result = ResearcherAgent().run(state)
    assert result.sources
    assert result.research_notes is not None
    assert "[Source 1]" in result.research_notes


def test_analyst_handles_missing_research_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = AnalystAgent().run(state)
    assert result.analysis_notes is not None
    assert "Research step must run first" in result.analysis_notes
    assert result.errors


def test_writer_generates_final_answer_with_references() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Research GraphRAG state-of-the-art"),
        research_notes="Research notes:\n- [Source 1] Demo source.",
        analysis_notes="Key findings:\n- Demo finding.",
    )
    result = WriterAgent().run(state)
    assert result.final_answer is not None
    assert "References:" in result.final_answer
