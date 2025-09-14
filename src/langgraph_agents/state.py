"""State management for LangGraph-based multi-agent system."""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import operator

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EscalationLevel(Enum):
    """Escalation levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a task in the system."""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    escalation_count: int = 0


@dataclass
class TestScenario:
    """Test scenario definition."""
    scenario_id: str
    title: str
    description: str
    test_type: str
    framework: str
    priority: str
    tags: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    expected_result: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Test execution result."""
    result_id: str
    scenario_id: str
    status: str
    execution_time: float
    error_message: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureCluster:
    """Failure cluster analysis."""
    cluster_id: str
    size: int
    common_patterns: List[str]
    representative_failure: Dict[str, Any]
    confidence_score: float
    suggested_fixes: List[str] = field(default_factory=list)


@dataclass
class LearningFeedback:
    """Learning feedback data."""
    feedback_id: str
    source_type: str  # human, system, logs, prs
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False


class AgentState(TypedDict):
    """State for individual agents."""
    agent_id: str
    name: str
    status: AgentStatus
    capabilities: Dict[str, Any]
    metrics: AgentMetrics
    current_task: Optional[Task]
    message_history: Annotated[List[BaseMessage], add_messages]


class WorkflowState(TypedDict):
    """State for workflow execution."""
    workflow_id: str
    execution_id: str
    status: TaskStatus
    current_step: int
    total_steps: int
    results: Dict[str, Any]
    errors: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    escalation_level: EscalationLevel


class TestAutomationState(TypedDict):
    """Main state for the test automation system."""
    # Messages for LangGraph
    messages: Annotated[List[BaseMessage], add_messages]
    
    # System state
    system_status: str
    active_tasks: List[Task]
    completed_tasks: List[Task]
    failed_tasks: List[Task]
    
    # Agent states
    agents: Dict[str, AgentState]
    
    # Workflow state
    workflow: Optional[WorkflowState]
    
    # Test data
    test_scenarios: List[TestScenario]
    test_results: List[TestResult]
    failure_clusters: List[FailureCluster]
    
    # Learning data
    feedback_data: List[LearningFeedback]
    learning_insights: Dict[str, Any]
    
    # Context data
    code_changes: Dict[str, Any]
    commit_metadata: Dict[str, Any]
    requirements: List[str]
    
    # Configuration
    config: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    escalation_needed: bool
    escalation_level: EscalationLevel
    
    # Performance metrics
    system_metrics: Dict[str, Any]
    
    # Timestamps
    created_at: datetime
    updated_at: datetime


def create_initial_state(
    workflow_id: str = "default",
    execution_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> TestAutomationState:
    """Create initial state for the test automation system."""
    if execution_id is None:
        execution_id = f"{workflow_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    if config is None:
        config = {
            "max_parallel_tasks": 5,
            "default_timeout": 300,
            "escalation_threshold": EscalationLevel.MEDIUM,
            "learning_mode": "incremental"
        }
    
    current_time = datetime.now()
    
    return TestAutomationState(
        messages=[],
        system_status="initializing",
        active_tasks=[],
        completed_tasks=[],
        failed_tasks=[],
        agents={},
        workflow=WorkflowState(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=TaskStatus.PENDING,
            current_step=0,
            total_steps=0,
            results={},
            errors=[],
            start_time=current_time,
            end_time=None,
            escalation_level=EscalationLevel.NONE
        ),
        test_scenarios=[],
        test_results=[],
        failure_clusters=[],
        feedback_data=[],
        learning_insights={},
        code_changes={},
        commit_metadata={},
        requirements=[],
        config=config,
        errors=[],
        escalation_needed=False,
        escalation_level=EscalationLevel.NONE,
        system_metrics={},
        created_at=current_time,
        updated_at=current_time
    )


def update_state_timestamp(state: TestAutomationState) -> TestAutomationState:
    """Update the timestamp in the state."""
    state["updated_at"] = datetime.now()
    return state


def add_task_to_state(state: TestAutomationState, task: Task) -> TestAutomationState:
    """Add a task to the state."""
    state["active_tasks"].append(task)
    return update_state_timestamp(state)


def move_task_to_completed(state: TestAutomationState, task_id: str, result: Dict[str, Any]) -> TestAutomationState:
    """Move a task from active to completed."""
    # Find and remove task from active
    for i, task in enumerate(state["active_tasks"]):
        if task.task_id == task_id:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            state["completed_tasks"].append(task)
            state["active_tasks"].pop(i)
            break
    
    return update_state_timestamp(state)


def move_task_to_failed(state: TestAutomationState, task_id: str, error: str) -> TestAutomationState:
    """Move a task from active to failed."""
    # Find and remove task from active
    for i, task in enumerate(state["active_tasks"]):
        if task.task_id == task_id:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now()
            state["failed_tasks"].append(task)
            state["active_tasks"].pop(i)
            break
    
    return update_state_timestamp(state)


def add_error_to_state(state: TestAutomationState, error: str, escalate: bool = False) -> TestAutomationState:
    """Add an error to the state and optionally trigger escalation."""
    state["errors"].append(f"{datetime.now().isoformat()}: {error}")
    
    if escalate:
        state["escalation_needed"] = True
        state["escalation_level"] = EscalationLevel.HIGH
    
    return update_state_timestamp(state)


def update_agent_status(state: TestAutomationState, agent_id: str, status: AgentStatus) -> TestAutomationState:
    """Update agent status in the state."""
    if agent_id in state["agents"]:
        state["agents"][agent_id]["status"] = status
        state["agents"][agent_id]["metrics"].last_activity = datetime.now()
    
    return update_state_timestamp(state)


def add_test_scenario(state: TestAutomationState, scenario: TestScenario) -> TestAutomationState:
    """Add a test scenario to the state."""
    state["test_scenarios"].append(scenario)
    return update_state_timestamp(state)


def add_test_result(state: TestAutomationState, result: TestResult) -> TestAutomationState:
    """Add a test result to the state."""
    state["test_results"].append(result)
    return update_state_timestamp(state)


def add_failure_cluster(state: TestAutomationState, cluster: FailureCluster) -> TestAutomationState:
    """Add a failure cluster to the state."""
    state["failure_clusters"].append(cluster)
    return update_state_timestamp(state)


def add_feedback_data(state: TestAutomationState, feedback: LearningFeedback) -> TestAutomationState:
    """Add feedback data to the state."""
    state["feedback_data"].append(feedback)
    return update_state_timestamp(state)


def update_learning_insights(state: TestAutomationState, insights: Dict[str, Any]) -> TestAutomationState:
    """Update learning insights in the state."""
    state["learning_insights"].update(insights)
    return update_state_timestamp(state)


def set_workflow_step(state: TestAutomationState, step: int, total_steps: int) -> TestAutomationState:
    """Set the current workflow step."""
    if state["workflow"]:
        state["workflow"]["current_step"] = step
        state["workflow"]["total_steps"] = total_steps
    
    return update_state_timestamp(state)


def complete_workflow(state: TestAutomationState, success: bool = True) -> TestAutomationState:
    """Complete the workflow."""
    if state["workflow"]:
        state["workflow"]["status"] = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        state["workflow"]["end_time"] = datetime.now()
    
    return update_state_timestamp(state)


def should_escalate(state: TestAutomationState) -> bool:
    """Check if escalation is needed based on state."""
    return (
        state["escalation_needed"] or
        len(state["failed_tasks"]) > 3 or
        len(state["errors"]) > 5 or
        any(agent["status"] == AgentStatus.ERROR for agent in state["agents"].values())
    )


def get_escalation_level(state: TestAutomationState) -> EscalationLevel:
    """Get the appropriate escalation level based on state."""
    if len(state["failed_tasks"]) > 5 or len(state["errors"]) > 10:
        return EscalationLevel.CRITICAL
    elif len(state["failed_tasks"]) > 3 or len(state["errors"]) > 5:
        return EscalationLevel.HIGH
    elif len(state["failed_tasks"]) > 1 or len(state["errors"]) > 2:
        return EscalationLevel.MEDIUM
    elif len(state["failed_tasks"]) > 0 or len(state["errors"]) > 0:
        return EscalationLevel.LOW
    else:
        return EscalationLevel.NONE


def get_system_health_score(state: TestAutomationState) -> float:
    """Calculate system health score (0.0 to 1.0)."""
    total_tasks = len(state["completed_tasks"]) + len(state["failed_tasks"])
    if total_tasks == 0:
        return 1.0
    
    success_rate = len(state["completed_tasks"]) / total_tasks
    
    # Penalize for errors and escalations
    error_penalty = min(len(state["errors"]) * 0.1, 0.5)
    escalation_penalty = 0.3 if state["escalation_needed"] else 0.0
    
    health_score = success_rate - error_penalty - escalation_penalty
    return max(0.0, min(1.0, health_score))


def get_next_agent_for_task(state: TestAutomationState, task_type: str) -> Optional[str]:
    """Get the next available agent for a task type."""
    # Simple agent selection logic - in practice, this would be more sophisticated
    agent_capabilities = {
        "plan_tests": ["test-planner"],
        "execute_tests": ["execution"],
        "diagnose_failures": ["diagnosis"],
        "learn_from_feedback": ["learning"]
    }
    
    suitable_agents = agent_capabilities.get(task_type, [])
    
    for agent_id in suitable_agents:
        if agent_id in state["agents"]:
            agent_state = state["agents"][agent_id]
            if agent_state["status"] in [AgentStatus.IDLE, AgentStatus.COMPLETED]:
                return agent_id
    
    return None


def create_task_from_data(task_type: str, data: Dict[str, Any]) -> Task:
    """Create a task from task type and data."""
    task_id = f"{task_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(str(data)) % 10000}"
    
    return Task(
        task_id=task_id,
        task_type=task_type,
        data=data,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )
