# Re-export root_agent so 'adk web' and 'adk run' can discover it
from agents.orchestrator import root_agent

__all__ = ["root_agent"]