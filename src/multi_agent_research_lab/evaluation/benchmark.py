"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and summarize metrics for the current lab implementation."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    total_cost = 0.0
    cost_found = False
    for result in state.agent_results:
        maybe_cost = result.metadata.get("cost_usd")
        if isinstance(maybe_cost, (float, int)):
            total_cost += float(maybe_cost)
            cost_found = True

    notes: list[str] = []
    if state.errors:
        notes.append(f"errors={len(state.errors)}")
    if state.sources:
        notes.append(f"sources={len(state.sources)}")
    if state.route_history:
        notes.append("routes=" + " -> ".join(state.route_history))

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=round(total_cost, 6) if cost_found else None,
        notes="; ".join(notes),
    )
    return state, metrics
