"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span, traceable
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    @traceable(name="writer_agent.run")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Final answer synthesis stays separate from research and analysis so prompts remain scoped.
        """
        with trace_span("writer.run", {"has_analysis_notes": bool(state.analysis_notes)}) as span:
            source_labels = ", ".join(f"[Source {index}]" for index, _ in enumerate(state.sources, start=1))
            user_prompt = "\n\n".join(
                [
                    f"Query: {state.request.query}",
                    f"Audience: {state.request.audience}",
                    state.research_notes or "Research notes unavailable.",
                    state.analysis_notes or "Analysis notes unavailable.",
                    f"Available references: {source_labels or 'No references'}",
                ]
            )
            response = self.llm_client.complete(
                system_prompt=(
                    "You are the writer agent in a multi-agent research lab. Produce a concise, "
                    "clear synthesis for technical learners and retain explicit references."
                ),
                user_prompt=user_prompt,
            )
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"]["final_answer_ready"] = True
            state.add_trace_event(
                "agent_completed",
                {
                    "agent": self.name,
                    "final_answer_preview": state.final_answer[:160],
                    "cost_usd": response.cost_usd,
                },
            )
        return state
