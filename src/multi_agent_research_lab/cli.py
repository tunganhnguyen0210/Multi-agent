"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    load_dotenv(override=False)
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline(query: str) -> ResearchState:
    search_client = SearchClient()
    llm_client = LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    state.sources = search_client.search(query=query, max_results=request.max_sources)
    source_block = "\n".join(
        f"- {document.title}: {document.snippet}" for document in state.sources
    )
    response = llm_client.complete(
        system_prompt=(
            "You are a single-agent baseline for a research lab. Summarize the query directly "
            "for technical learners using the provided context."
        ),
        user_prompt=f"Query: {query}\nSources:\n{source_block}",
    )
    state.final_answer = response.content
    state.add_trace_event(
        "baseline_completed",
        {"source_count": len(state.sources), "cost_usd": response.cost_usd},
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
