"""LangGraph workflow skeleton."""

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, SupervisorAgent, WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import traceable


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()

    def build(self) -> object:
        """Create a LangGraph graph.

        The starter implementation returns a graph description that mirrors the control flow.
        Students can later translate this directly into a LangGraph builder.
        """
        return {
            "nodes": ["supervisor", "researcher", "analyst", "writer"],
            "entrypoint": "supervisor",
            "conditional_routes": {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": "__end__",
            },
        }

    @traceable(name="multi_agent_workflow.run")
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Execute the workflow with a plain Python loop first so the orchestration stays explicit.
        """
        _ = self.build()
        settings = get_settings()
        while state.iteration < settings.max_iterations + 1:
            state = self.supervisor.run(state)
            current_route = state.route_history[-1]
            if current_route == "done":
                break
            if current_route == "researcher":
                state = self.researcher.run(state)
            elif current_route == "analyst":
                state = self.analyst.run(state)
            elif current_route == "writer":
                state = self.writer.run(state)
                state = self.supervisor.run(state)
                break
            else:
                state.errors.append(f"Unknown route requested by supervisor: {current_route}")
                break
        return state
