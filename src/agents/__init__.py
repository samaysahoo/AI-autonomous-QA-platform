"""Multi-agent architecture for autonomous test automation."""

from .base_agent import BaseAgent, AgentState, AgentMessage
from .test_planner_agent import TestPlannerAgent
from .execution_agent import ExecutionAgent
from .diagnosis_agent import DiagnosisAgent
from .learning_agent import LearningAgent
from .agent_coordinator import AgentCoordinator
from .agent_registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentState", 
    "AgentMessage",
    "TestPlannerAgent",
    "ExecutionAgent", 
    "DiagnosisAgent",
    "LearningAgent",
    "AgentCoordinator",
    "AgentRegistry"
]
