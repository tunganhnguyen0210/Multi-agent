"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span, traceable
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    @traceable(name="researcher_agent.run")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Search and summarization stay here so later provider changes do not leak into the workflow.
        """
        with trace_span("researcher.run", {"query": state.request.query}) as span:
            documents = self.search_client.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            deduped: list[tuple[str, object]] = []
            seen_titles: set[str] = set()
            for document in documents:
                lowered = document.title.lower()
                if lowered in seen_titles:
                    continue
                seen_titles.add(lowered)
                deduped.append((lowered, document))

            state.sources = [document for _, document in deduped]
            if state.sources:
                lines = ["Research notes:"]
                for index, document in enumerate(state.sources, start=1):
                    lines.append(f"- [Source {index}] {document.title}: {document.snippet}")
                state.research_notes = "\n".join(lines)
            else:
                state.research_notes = "Research notes:\n- No supporting sources were found."
                state.errors.append("Researcher found no sources for the query.")

            span["attributes"]["source_count"] = len(state.sources)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes,
                    metadata={"source_count": len(state.sources)},
                )
            )
            state.add_trace_event(
                "agent_completed",
                {
                    "agent": self.name,
                    "source_count": len(state.sources),
                    "note_preview": state.research_notes[:160],
                },
            )
        return state
