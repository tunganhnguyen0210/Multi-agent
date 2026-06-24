"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span, traceable


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    @traceable(name="supervisor_agent.run")
    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Default lab policy is rule-based so the routing logic stays transparent.
        """

        settings = get_settings()
        with trace_span(
            "supervisor.run",
            {"iteration": state.iteration, "route_history": list(state.route_history)},
        ) as span:
            if state.final_answer:
                next_route = "done"
            elif state.iteration >= settings.max_iterations:
                next_route = "writer" if state.analysis_notes or state.research_notes else "done"
            elif state.errors and len(state.errors) >= 2 and not state.final_answer:
                next_route = "writer" if state.research_notes else "done"
            elif not state.research_notes:
                next_route = AgentName.RESEARCHER.value
            elif not state.analysis_notes:
                next_route = AgentName.ANALYST.value
            elif not state.final_answer:
                next_route = AgentName.WRITER.value
            else:
                next_route = "done"

            state.record_route(next_route)
            span["attributes"]["next_route"] = next_route
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=f"Supervisor selected route: {next_route}",
                    metadata={"iteration": state.iteration},
                )
            )
            state.add_trace_event(
                "supervisor_route",
                {
                    "agent": self.name,
                    "next_route": next_route,
                    "iteration": state.iteration,
                    "history_size": len(state.route_history),
                },
            )
        return state
