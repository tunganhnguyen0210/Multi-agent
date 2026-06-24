"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.observability.tracing import traceable

_DEFAULT_DOCUMENTS: tuple[SourceDocument, ...] = (
    SourceDocument(
        title="GraphRAG overview and workflow patterns",
        url="https://example.local/graphrag-overview",
        snippet=(
            "GraphRAG combines knowledge-graph structure with retrieval to improve grounding, "
            "multi-hop reasoning, and explainability for long-form research tasks."
        ),
        metadata={"topic": "graphrag", "source_type": "overview"},
    ),
    SourceDocument(
        title="Multi-agent orchestration trade-offs",
        url="https://example.local/multi-agent-tradeoffs",
        snippet=(
            "Supervisor-worker systems improve specialization and traceability, but they add "
            "routing overhead and need strong state design plus stop conditions."
        ),
        metadata={"topic": "multi-agent", "source_type": "analysis"},
    ),
    SourceDocument(
        title="Benchmarking agent systems",
        url="https://example.local/agent-benchmarking",
        snippet=(
            "Useful evaluation dimensions include latency, citation coverage, failure rate, "
            "quality review, and the cost of extra orchestration."
        ),
        metadata={"topic": "benchmark", "source_type": "guide"},
    ),
    SourceDocument(
        title="RAG failure modes and mitigation",
        url="https://example.local/rag-failure-modes",
        snippet=(
            "Common failures include weak retrieval, unsupported claims, duplicated evidence, "
            "and answers that skip uncertainty or source quality discussion."
        ),
        metadata={"topic": "rag", "source_type": "risk"},
    ),
    SourceDocument(
        title="Agent writing patterns for technical learners",
        url="https://example.local/technical-writing",
        snippet=(
            "Writer agents should synthesize evidence into a concise structure with practical "
            "takeaways and explicit references back to supporting sources."
        ),
        metadata={"topic": "writing", "source_type": "style"},
    ),
)


class SearchClient:
    """Provider-agnostic search client skeleton."""

    @traceable(run_type="tool", name="search_client.search")
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        The lab default is deterministic mock retrieval so students can focus on orchestration
        before connecting a real search provider.
        """

        normalized = query.lower()
        scored: list[tuple[int, SourceDocument]] = []
        for document in _DEFAULT_DOCUMENTS:
            haystack = " ".join(
                [
                    document.title.lower(),
                    document.snippet.lower(),
                    str(document.metadata.get("topic", "")).lower(),
                ]
            )
            score = sum(1 for token in normalized.split() if token in haystack)
            if "graphrag" in normalized and document.metadata.get("topic") == "graphrag":
                score += 3
            if "multi-agent" in normalized and document.metadata.get("topic") == "multi-agent":
                score += 3
            scored.append((score, document))

        scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
        selected = [document for score, document in scored if score > 0]
        if not selected:
            selected = list(_DEFAULT_DOCUMENTS[: max(1, min(max_results, 3))])
        return selected[:max_results]
