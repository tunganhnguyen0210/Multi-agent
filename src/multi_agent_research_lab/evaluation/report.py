"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    The default report is concise but includes the comparison table expected by the README.
    """

    lines = ["# Benchmark Report", "", "| Run | Latency (s) | Cost (USD) | Quality | Notes |", "|---|---:|---:|---:|---|"]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare latency first to understand orchestration overhead.",
            "- Review notes for route history, source count, and any execution errors.",
            "- Add screenshots or external trace links here when you connect a real tracing backend.",
        ]
    )
    return "\n".join(lines) + "\n"
