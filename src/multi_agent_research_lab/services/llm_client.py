"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.observability.tracing import traceable


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    @traceable(run_type="llm", name="llm_client.complete")
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        This lab default is a deterministic mock. Students can later swap the implementation
        with OpenAI, Azure OpenAI, or another provider without changing agent code.
        """

        system_lower = system_prompt.lower()
        user_lower = user_prompt.lower()
        joined_prompt = f"{system_prompt}\n{user_prompt}"
        input_tokens = max(24, len(joined_prompt.split()))

        if "researcher" in system_lower:
            content = (
                "Research notes:\n"
                "- GraphRAG improves multi-hop retrieval by combining graph structure with RAG.\n"
                "- Multi-agent workflows help separate retrieval, analysis, and writing concerns.\n"
                "- Benchmarking should track latency, quality, citation coverage, and failure modes."
            )
        elif "analyst" in system_lower:
            content = (
                "Key findings:\n"
                "- Graph-aware retrieval helps when a query needs relationships across sources.\n"
                "- Multi-agent design improves interpretability more reliably than raw answer quality.\n"
                "Comparisons / trade-offs:\n"
                "- Single-agent is faster and simpler.\n"
                "- Multi-agent is slower but easier to debug and audit.\n"
                "Weak evidence / gaps:\n"
                "- Mock sources mean claims should be treated as a lab demonstration, not production evidence."
            )
        elif "writer" in system_lower:
            content = (
                "GraphRAG is useful when a task depends on relationships across documents rather than "
                "simple keyword matching. In this lab workflow, the researcher gathers sources, the "
                "analyst converts them into trade-offs, and the writer produces the final answer with "
                "clear references. The main trade-off is extra orchestration overhead in exchange for "
                "better traceability and easier debugging.\n\n"
                "References: [Source 1], [Source 2], [Source 3]"
            )
        elif "baseline" in system_lower or "single-agent" in system_lower:
            content = (
                "Single-agent baseline summary: GraphRAG combines retrieval with graph structure to "
                "support multi-hop reasoning and clearer grounding, but it still depends heavily on "
                "retrieval quality and prompt discipline."
            )
        else:
            topic = "the requested topic"
            if "graphrag" in user_lower:
                topic = "GraphRAG"
            elif "multi-agent" in user_lower:
                topic = "multi-agent systems"
            content = f"Deterministic mock response about {topic} for lab orchestration."

        output_tokens = max(20, len(content.split()))
        estimated_cost = round((input_tokens + output_tokens) * 0.000002, 6)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimated_cost,
        )
