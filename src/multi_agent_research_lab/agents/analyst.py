"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span, traceable


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    @traceable(name="analyst_agent.run")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        The first lab version uses deterministic analysis so the workflow is easy to inspect.
        """
        with trace_span("analyst.run", {"has_research_notes": bool(state.research_notes)}) as span:
            if not state.research_notes:
                state.analysis_notes = (
                    "Key findings:\n- Not enough research notes to analyze.\n"
                    "Comparisons / trade-offs:\n- Analysis deferred until sources exist.\n"
                    "Weak evidence / gaps:\n- Research step must run first."
                )
                state.errors.append("Analyst ran without research notes.")
            else:
                source_titles = ", ".join(source.title for source in state.sources[:3]) or "available sources"
                state.analysis_notes = (
                    "Key findings:\n"
                    "- The collected sources agree that graph-aware retrieval helps with multi-hop questions.\n"
                    "- Specializing roles improves traceability and makes failures easier to locate.\n"
                    "Comparisons / trade-offs:\n"
                    "- Single-agent is faster and simpler to maintain.\n"
                    "- Multi-agent adds coordination overhead but creates cleaner handoffs and debugging signals.\n"
                    "Weak evidence / gaps:\n"
                    f"- Current analysis is derived from mock evidence anchored in {source_titles}.\n"
                    "- A real provider should validate claims and add fresher citations."
                )

            span["attributes"]["analysis_ready"] = bool(state.analysis_notes)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes,
                    metadata={"source_count": len(state.sources)},
                )
            )
            state.add_trace_event(
                "agent_completed",
                {
                    "agent": self.name,
                    "analysis_preview": state.analysis_notes[:160],
                    "error_count": len(state.errors),
                },
            )
        return state
