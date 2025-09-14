"""Multi-Agent API - REST API for the multi-agent system."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio

from .agent_coordinator import AgentCoordinator, WorkflowDefinition, CoordinationStrategy, EscalationLevel
from .agent_registry import AgentRegistry
from .test_planner_agent import TestPlannerAgent
from .execution_agent import ExecutionAgent
from .diagnosis_agent import DiagnosisAgent
from .learning_agent import LearningAgent

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Test Automation API",
    description="REST API for the AI-powered multi-agent test automation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator and registry instances
coordinator: Optional[AgentCoordinator] = None
registry: Optional[AgentRegistry] = None


# Pydantic models for API requests/responses
class TaskRequest(BaseModel):
    """Request model for task execution."""
    task_type: str = Field(..., description="Type of task to execute")
    task_data: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    agent_id: Optional[str] = Field(None, description="Specific agent ID (optional)")
    timeout: int = Field(300, description="Task timeout in seconds")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str = Field(..., description="Unique task ID")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for workflow")
    coordination_strategy: str = Field("adaptive", description="Coordination strategy")
    escalation_threshold: str = Field("medium", description="Escalation threshold")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    execution_id: str = Field(..., description="Workflow execution ID")
    status: str = Field(..., description="Execution status")
    workflow_id: str = Field(..., description="Workflow ID")
    results: Dict[str, Any] = Field(default_factory=dict, description="Workflow results")
    errors: List[str] = Field(default_factory=list, description="Execution errors")


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status")
    capabilities: Dict[str, Any] = Field(..., description="Agent capabilities")
    metrics: Dict[str, Any] = Field(..., description="Agent metrics")
    active_tasks: int = Field(..., description="Number of active tasks")


class CoordinatorStatusResponse(BaseModel):
    """Response model for coordinator status."""
    coordinator_id: str = Field(..., description="Coordinator ID")
    active_agents: int = Field(..., description="Number of active agents")
    active_workflows: int = Field(..., description="Number of active workflows")
    agent_status: Dict[str, Dict[str, Any]] = Field(..., description="Agent status details")
    workflow_status: Dict[str, Dict[str, Any]] = Field(..., description="Workflow status details")


class TestPlanningRequest(BaseModel):
    """Request model for test planning."""
    change_type: str = Field(..., description="Type of change (feature, bug_fix, etc.)")
    diff_content: str = Field("", description="Code diff content")
    changed_files: List[str] = Field(default_factory=list, description="Changed files")
    commit_metadata: Dict[str, Any] = Field(default_factory=dict, description="Commit metadata")
    requirements: List[str] = Field(default_factory=list, description="Requirements")


class TestExecutionRequest(BaseModel):
    """Request model for test execution."""
    scenarios: List[Dict[str, Any]] = Field(..., description="Test scenarios to execute")
    execution_strategy: str = Field("adaptive", description="Execution strategy")
    max_parallel: int = Field(3, description="Maximum parallel executions")
    timeout: int = Field(1800, description="Execution timeout in seconds")


class DiagnosisRequest(BaseModel):
    """Request model for diagnosis."""
    test_results: List[Dict[str, Any]] = Field(..., description="Test results to analyze")
    clustering_method: str = Field("auto", description="Clustering method")
    min_cluster_size: int = Field(2, description="Minimum cluster size")


class LearningRequest(BaseModel):
    """Request model for learning."""
    feedback_data: List[Dict[str, Any]] = Field(..., description="Feedback data")
    learning_mode: str = Field("incremental", description="Learning mode")
    model_types: List[str] = Field(["all"], description="Model types to update")


# Dependency functions
async def get_coordinator() -> AgentCoordinator:
    """Get the coordinator instance."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    return coordinator


async def get_registry() -> AgentRegistry:
    """Get the registry instance."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return registry


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the multi-agent system on startup."""
    global coordinator, registry
    
    logger.info("Starting Multi-Agent Test Automation API")
    
    try:
        # Initialize registry
        registry = AgentRegistry()
        
        # Initialize coordinator
        coordinator = AgentCoordinator()
        
        # Register default agents
        await register_default_agents()
        
        # Start coordinator
        await coordinator.start_coordination()
        
        logger.info("Multi-Agent system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize multi-agent system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global coordinator, registry
    
    logger.info("Shutting down Multi-Agent Test Automation API")
    
    try:
        if coordinator:
            await coordinator.stop_coordinator()
        
        if registry:
            registry.cleanup()
        
        logger.info("Multi-Agent system shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def register_default_agents():
    """Register default agents with the system."""
    if not registry or not coordinator:
        return
    
    # Create and register agents
    agents = [
        TestPlannerAgent("test-planner-001"),
        ExecutionAgent("execution-001"),
        DiagnosisAgent("diagnosis-001"),
        LearningAgent("learning-001")
    ]
    
    for agent in agents:
        registry.register_agent(agent)
        coordinator.agents[agent.agent_id] = agent
    
    logger.info(f"Registered {len(agents)} default agents")


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Agent Test Automation API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if not coordinator or not registry:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_agents": len(coordinator.agents),
        "active_workflows": len(coordinator.active_workflows)
    }


# Agent Management Endpoints

@app.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents(registry: AgentRegistry = Depends(get_registry)):
    """List all registered agents."""
    return registry.get_agent_list()


@app.get("/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(agent_id: str, coordinator: AgentCoordinator = Depends(get_coordinator)):
    """Get status of a specific agent."""
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = coordinator.agents[agent_id]
    status = agent.get_status()
    
    return AgentStatusResponse(**status)


@app.post("/agents/{agent_id}/tasks", response_model=TaskResponse)
async def execute_agent_task(
    agent_id: str,
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute a task on a specific agent."""
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = coordinator.agents[agent_id]
    
    # Check if agent can handle the task
    if not agent.can_handle_task(task_request.task_type, task_request.task_data):
        raise HTTPException(
            status_code=400, 
            detail=f"Agent {agent_id} cannot handle task type {task_request.task_type}"
        )
    
    # Create task message
    from .base_agent import AgentMessage, MessageType
    
    task_message = AgentMessage(
        sender_id="api",
        receiver_id=agent_id,
        message_type=MessageType.TASK_REQUEST,
        payload={
            "task_data": task_request.task_data,
            "task_type": task_request.task_type
        }
    )
    
    try:
        start_time = datetime.now()
        
        # Execute task
        response = await asyncio.wait_for(
            agent.receive_message(task_message),
            timeout=task_request.timeout
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if response and response.message_type == MessageType.TASK_RESPONSE:
            payload = response.payload
            
            return TaskResponse(
                task_id=task_message.message_id,
                status=payload.get("status", "completed"),
                result=payload.get("result"),
                error=payload.get("error"),
                execution_time=execution_time
            )
        else:
            raise HTTPException(status_code=500, detail="Invalid response from agent")
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Task execution timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


# Workflow Management Endpoints

@app.get("/workflows", response_model=List[Dict[str, Any]])
async def list_workflows(coordinator: AgentCoordinator = Depends(get_coordinator)):
    """List available workflows."""
    workflows = []
    
    for workflow_id, workflow_def in coordinator.workflow_definitions.items():
        workflows.append({
            "workflow_id": workflow_id,
            "name": workflow_def.name,
            "description": workflow_def.description,
            "steps": len(workflow_def.steps),
            "coordination_strategy": workflow_def.coordination_strategy.value,
            "escalation_threshold": workflow_def.escalation_threshold.value
        })
    
    return workflows


@app.post("/workflows/execute", response_model=WorkflowResponse)
async def execute_workflow(
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute a workflow."""
    if workflow_request.workflow_id not in coordinator.workflow_definitions:
        raise HTTPException(
            status_code=404, 
            detail=f"Workflow {workflow_request.workflow_id} not found"
        )
    
    try:
        execution_id = await coordinator.start_workflow(
            workflow_request.workflow_id,
            workflow_request.input_data
        )
        
        return WorkflowResponse(
            execution_id=execution_id,
            status="started",
            workflow_id=workflow_request.workflow_id,
            results={},
            errors=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@app.get("/workflows/{execution_id}", response_model=WorkflowResponse)
async def get_workflow_status(
    execution_id: str,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Get workflow execution status."""
    if execution_id not in coordinator.active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow execution {execution_id} not found")
    
    execution = coordinator.active_workflows[execution_id]
    
    return WorkflowResponse(
        execution_id=execution_id,
        status=execution.status,
        workflow_id=execution.workflow_definition.workflow_id,
        results=execution.results,
        errors=execution.errors
    )


# Specialized Agent Endpoints

@app.post("/test-planner/plan", response_model=Dict[str, Any])
async def plan_tests(
    planning_request: TestPlanningRequest,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Plan tests for a change."""
    agent_id = "test-planner-001"
    
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail="Test Planner Agent not found")
    
    task_request = TaskRequest(
        task_type="plan_tests_for_change",
        task_data={
            "change_type": planning_request.change_type,
            "diff_content": planning_request.diff_content,
            "changed_files": planning_request.changed_files,
            "commit_metadata": planning_request.commit_metadata,
            "requirements": planning_request.requirements
        }
    )
    
    # Execute task
    task_response = await execute_agent_task(agent_id, task_request, BackgroundTasks(), coordinator)
    
    if task_response.status != "completed":
        raise HTTPException(status_code=500, detail=f"Test planning failed: {task_response.error}")
    
    return task_response.result


@app.post("/execution/run", response_model=Dict[str, Any])
async def execute_tests(
    execution_request: TestExecutionRequest,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute test scenarios."""
    agent_id = "execution-001"
    
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail="Execution Agent not found")
    
    task_request = TaskRequest(
        task_type="execute_test_scenarios",
        task_data={
            "scenarios": execution_request.scenarios,
            "strategy": execution_request.execution_strategy,
            "max_parallel": execution_request.max_parallel,
            "timeout": execution_request.timeout
        }
    )
    
    # Execute task
    task_response = await execute_agent_task(agent_id, task_request, BackgroundTasks(), coordinator)
    
    if task_response.status != "completed":
        raise HTTPException(status_code=500, detail=f"Test execution failed: {task_response.error}")
    
    return task_response.result


@app.post("/diagnosis/analyze", response_model=Dict[str, Any])
async def analyze_failures(
    diagnosis_request: DiagnosisRequest,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Analyze test failures."""
    agent_id = "diagnosis-001"
    
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail="Diagnosis Agent not found")
    
    task_request = TaskRequest(
        task_type="cluster_test_failures",
        task_data={
            "test_results": diagnosis_request.test_results,
            "clustering_method": diagnosis_request.clustering_method,
            "min_cluster_size": diagnosis_request.min_cluster_size
        }
    )
    
    # Execute task
    task_response = await execute_agent_task(agent_id, task_request, BackgroundTasks(), coordinator)
    
    if task_response.status != "completed":
        raise HTTPException(status_code=500, detail=f"Failure analysis failed: {task_response.error}")
    
    return task_response.result


@app.post("/learning/update", response_model=Dict[str, Any])
async def update_models(
    learning_request: LearningRequest,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Update models based on feedback."""
    agent_id = "learning-001"
    
    if agent_id not in coordinator.agents:
        raise HTTPException(status_code=404, detail="Learning Agent not found")
    
    task_request = TaskRequest(
        task_type="learn_from_feedback",
        task_data={
            "feedback": learning_request.feedback_data,
            "learning_mode": learning_request.learning_mode,
            "model_types": learning_request.model_types
        }
    )
    
    # Execute task
    task_response = await execute_agent_task(agent_id, task_request, BackgroundTasks(), coordinator)
    
    if task_response.status != "completed":
        raise HTTPException(status_code=500, detail=f"Model update failed: {task_response.error}")
    
    return task_response.result


# System Status Endpoints

@app.get("/coordinator/status", response_model=CoordinatorStatusResponse)
async def get_coordinator_status(coordinator: AgentCoordinator = Depends(get_coordinator)):
    """Get coordinator status."""
    status = coordinator.get_coordinator_status()
    
    return CoordinatorStatusResponse(**status)


@app.get("/registry/status", response_model=Dict[str, Any])
async def get_registry_status(registry: AgentRegistry = Depends(get_registry)):
    """Get registry status."""
    return registry.get_registry_status()


# End-to-End Workflow Endpoints

@app.post("/workflows/e2e-test", response_model=WorkflowResponse)
async def execute_e2e_test_workflow(
    planning_request: TestPlanningRequest,
    background_tasks: BackgroundTasks,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute complete end-to-end test workflow."""
    workflow_request = WorkflowRequest(
        workflow_id="e2e-test-workflow",
        input_data={
            "change_type": planning_request.change_type,
            "diff_content": planning_request.diff_content,
            "changed_files": planning_request.changed_files,
            "commit_metadata": planning_request.commit_metadata,
            "requirements": planning_request.requirements
        }
    )
    
    return await execute_workflow(workflow_request, background_tasks, coordinator)


@app.post("/workflows/bug-triage", response_model=WorkflowResponse)
async def execute_bug_triage_workflow(
    diagnosis_request: DiagnosisRequest,
    background_tasks: BackgroundTasks,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute bug triage workflow."""
    workflow_request = WorkflowRequest(
        workflow_id="bug-triage-workflow",
        input_data={
            "test_results": diagnosis_request.test_results,
            "clustering_method": diagnosis_request.clustering_method,
            "min_cluster_size": diagnosis_request.min_cluster_size
        }
    )
    
    return await execute_workflow(workflow_request, background_tasks, coordinator)


@app.post("/workflows/performance-optimization", response_model=WorkflowResponse)
async def execute_performance_optimization_workflow(
    background_tasks: BackgroundTasks,
    coordinator: AgentCoordinator = Depends(get_coordinator)
):
    """Execute performance optimization workflow."""
    workflow_request = WorkflowRequest(
        workflow_id="performance-optimization-workflow",
        input_data={}
    )
    
    return await execute_workflow(workflow_request, background_tasks, coordinator)


# Utility Endpoints

@app.get("/capabilities", response_model=Dict[str, List[str]])
async def get_system_capabilities(registry: AgentRegistry = Depends(get_registry)):
    """Get system capabilities by category."""
    capabilities = {
        "planning": [],
        "execution": [],
        "diagnosis": [],
        "learning": [],
        "coordination": []
    }
    
    for agent_info in registry.get_agent_list():
        agent_caps = agent_info["capabilities"]
        
        if agent_caps["can_plan_tests"]:
            capabilities["planning"].append(agent_info["agent_id"])
        
        if agent_caps["can_execute_tests"]:
            capabilities["execution"].append(agent_info["agent_id"])
        
        if agent_caps["can_diagnose_failures"]:
            capabilities["diagnosis"].append(agent_info["agent_id"])
        
        if agent_caps["can_learn_from_data"]:
            capabilities["learning"].append(agent_info["agent_id"])
        
        if agent_caps["can_coordinate_agents"]:
            capabilities["coordination"].append(agent_info["agent_id"])
    
    return capabilities


@app.get("/platforms", response_model=Dict[str, List[str]])
async def get_supported_platforms(registry: AgentRegistry = Depends(get_registry)):
    """Get supported platforms."""
    platforms = {}
    
    for agent_info in registry.get_agent_list():
        agent_id = agent_info["agent_id"]
        supported_platforms = agent_info["capabilities"]["supported_platforms"]
        
        for platform in supported_platforms:
            if platform not in platforms:
                platforms[platform] = []
            platforms[platform].append(agent_id)
    
    return platforms


@app.get("/frameworks", response_model=Dict[str, List[str]])
async def get_supported_frameworks(registry: AgentRegistry = Depends(get_registry)):
    """Get supported frameworks."""
    frameworks = {}
    
    for agent_info in registry.get_agent_list():
        agent_id = agent_info["agent_id"]
        supported_frameworks = agent_info["capabilities"]["supported_frameworks"]
        
        for framework in supported_frameworks:
            if framework not in frameworks:
                frameworks[framework] = []
            frameworks[framework].append(agent_id)
    
    return frameworks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
