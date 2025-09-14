"""FastAPI application for LangGraph-based multi-agent system."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio

from .workflow_graph import TestAutomationWorkflow
from .state import TestAutomationState

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for workflow")
    config: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_id: str = Field(..., description="Workflow ID")
    execution_id: str = Field(..., description="Execution ID")
    status: str = Field(..., description="Execution status")
    summary: Optional[Dict[str, Any]] = Field(None, description="Workflow summary")
    messages: List[str] = Field(default_factory=list, description="Workflow messages")
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    escalation_needed: bool = Field(False, description="Whether escalation is needed")
    results: Dict[str, Any] = Field(default_factory=dict, description="Execution results")


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


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = Field(..., description="System status")
    active_workflows: int = Field(..., description="Number of active workflows")
    agents_status: Dict[str, Dict[str, Any]] = Field(..., description="Agent status")
    system_metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: str = Field(..., description="Status timestamp")


# Global workflow instance
workflow_instance: Optional[TestAutomationWorkflow] = None


def create_langgraph_app() -> FastAPI:
    """Create the FastAPI application for LangGraph-based multi-agent system."""
    
    app = FastAPI(
        title="LangGraph Multi-Agent Test Automation API",
        description="REST API for the LangGraph-based AI-powered multi-agent test automation system",
        version="2.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency to get workflow instance
    async def get_workflow() -> TestAutomationWorkflow:
        global workflow_instance
        if workflow_instance is None:
            workflow_instance = TestAutomationWorkflow()
        return workflow_instance
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize the LangGraph workflow on startup."""
        global workflow_instance
        logger.info("Starting LangGraph Multi-Agent Test Automation API")
        
        try:
            workflow_instance = TestAutomationWorkflow()
            logger.info("LangGraph workflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph workflow: {e}")
            raise
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down LangGraph Multi-Agent Test Automation API")
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "LangGraph Multi-Agent Test Automation API",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "framework": "LangGraph"
        }
    
    # Health check endpoint
    @app.get("/health", response_model=Dict[str, str])
    async def health_check(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "framework": "LangGraph",
            "workflow_available": "true"
        }
    
    # Workflow Management Endpoints
    
    @app.get("/workflows", response_model=List[Dict[str, Any]])
    async def list_workflows(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """List available workflows."""
        return workflow.get_available_workflows()
    
    @app.post("/workflows/execute", response_model=WorkflowResponse)
    async def execute_workflow(
        workflow_request: WorkflowRequest,
        background_tasks: BackgroundTasks,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
    ):
        """Execute a workflow."""
        try:
            result = await workflow.execute_workflow(
                workflow_request.workflow_id,
                workflow_request.input_data,
                workflow_request.config
            )
            
            return WorkflowResponse(**result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
    
    @app.get("/workflows/{execution_id}", response_model=Dict[str, Any])
    async def get_workflow_status(
        execution_id: str,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
    ):
        """Get workflow execution status."""
        try:
            status = await workflow.get_workflow_status(execution_id)
            return status
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Workflow {execution_id} not found: {str(e)}")
    
    # Specialized Workflow Endpoints
    
    @app.post("/workflows/e2e-test", response_model=WorkflowResponse)
    async def execute_e2e_test_workflow(
        planning_request: TestPlanningRequest,
        background_tasks: BackgroundTasks,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
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
        
        result = await workflow.execute_workflow(
            workflow_request.workflow_id,
            workflow_request.input_data
        )
        
        return WorkflowResponse(**result)
    
    @app.post("/workflows/bug-triage", response_model=WorkflowResponse)
    async def execute_bug_triage_workflow(
        diagnosis_request: DiagnosisRequest,
        background_tasks: BackgroundTasks,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
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
        
        result = await workflow.execute_workflow(
            workflow_request.workflow_id,
            workflow_request.input_data
        )
        
        return WorkflowResponse(**result)
    
    @app.post("/workflows/performance-optimization", response_model=WorkflowResponse)
    async def execute_performance_optimization_workflow(
        background_tasks: BackgroundTasks,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
    ):
        """Execute performance optimization workflow."""
        workflow_request = WorkflowRequest(
            workflow_id="performance-optimization-workflow",
            input_data={}
        )
        
        result = await workflow.execute_workflow(
            workflow_request.workflow_id,
            workflow_request.input_data
        )
        
        return WorkflowResponse(**result)
    
    # Agent Status Endpoints
    
    @app.get("/agents", response_model=Dict[str, Dict[str, Any]])
    async def list_agents(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """List all agents and their status."""
        agents_status = {}
        
        for agent_id, agent in workflow.agents.items():
            agents_status[agent_id] = await agent.get_agent_status()
        
        return agents_status
    
    @app.get("/agents/{agent_id}", response_model=Dict[str, Any])
    async def get_agent_status(
        agent_id: str,
        workflow: TestAutomationWorkflow = Depends(get_workflow)
    ):
        """Get status of a specific agent."""
        if agent_id not in workflow.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = workflow.agents[agent_id]
        return await agent.get_agent_status()
    
    # System Status Endpoints
    
    @app.get("/system/status", response_model=SystemStatusResponse)
    async def get_system_status(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """Get comprehensive system status."""
        agents_status = {}
        
        for agent_id, agent in workflow.agents.items():
            agents_status[agent_id] = await agent.get_agent_status()
        
        return SystemStatusResponse(
            status="healthy",
            active_workflows=0,  # Would track active workflows in practice
            agents_status=agents_status,
            system_metrics={
                "total_agents": len(workflow.agents),
                "available_workflows": len(workflow.get_available_workflows()),
                "framework": "LangGraph",
                "uptime": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
    
    @app.get("/system/metrics", response_model=Dict[str, Any])
    async def get_system_metrics(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """Get system performance metrics."""
        metrics = {
            "agents": {},
            "workflows": {},
            "system": {
                "uptime": datetime.now().isoformat(),
                "framework": "LangGraph",
                "version": "2.0.0"
            }
        }
        
        # Collect agent metrics
        for agent_id, agent in workflow.agents.items():
            agent_status = await agent.get_agent_status()
            metrics["agents"][agent_id] = agent_status.get("metrics", {})
        
        return metrics
    
    # Capabilities Endpoints
    
    @app.get("/capabilities", response_model=Dict[str, List[str]])
    async def get_system_capabilities(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """Get system capabilities by category."""
        capabilities = {
            "planning": [],
            "execution": [],
            "diagnosis": [],
            "learning": [],
            "workflows": []
        }
        
        # Get agent capabilities
        for agent_id, agent in workflow.agents.items():
            agent_status = await agent.get_agent_status()
            agent_caps = agent_status.get("capabilities", {})
            
            if agent_caps.get("plan_tests_for_change"):
                capabilities["planning"].append(agent_id)
            
            if agent_caps.get("execute_test_scenarios"):
                capabilities["execution"].append(agent_id)
            
            if agent_caps.get("cluster_test_failures"):
                capabilities["diagnosis"].append(agent_id)
            
            if agent_caps.get("learn_from_feedback"):
                capabilities["learning"].append(agent_id)
        
        # Get workflow capabilities
        for workflow_info in workflow.get_available_workflows():
            capabilities["workflows"].append(workflow_info["workflow_id"])
        
        return capabilities
    
    @app.get("/workflows/capabilities", response_model=Dict[str, Any])
    async def get_workflow_capabilities(workflow: TestAutomationWorkflow = Depends(get_workflow)):
        """Get detailed workflow capabilities."""
        capabilities = {}
        
        for workflow_info in workflow.get_available_workflows():
            capabilities[workflow_info["workflow_id"]] = {
                "name": workflow_info["name"],
                "description": workflow_info["description"],
                "steps": workflow_info["steps"],
                "estimated_duration": workflow_info["estimated_duration"],
                "required_inputs": self._get_required_inputs(workflow_info["workflow_id"]),
                "expected_outputs": self._get_expected_outputs(workflow_info["workflow_id"])
            }
        
        return capabilities
    
    # Utility Endpoints
    
    @app.get("/version", response_model=Dict[str, str])
    async def get_version():
        """Get API version information."""
        return {
            "api_version": "2.0.0",
            "framework": "LangGraph",
            "langchain_version": "0.1.0",
            "langgraph_version": "0.0.20",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/docs/architecture", response_model=Dict[str, Any])
    async def get_architecture_docs():
        """Get architecture documentation."""
        return {
            "framework": "LangGraph",
            "architecture": "Multi-Agent System",
            "components": {
                "agents": [
                    "TestPlannerAgent",
                    "ExecutionAgent", 
                    "DiagnosisAgent",
                    "LearningAgent"
                ],
                "workflow_engine": "LangGraph StateGraph",
                "state_management": "TypedDict with message history",
                "coordination": "Graph-based workflow orchestration"
            },
            "features": [
                "Stateful workflow execution",
                "Message-based agent communication",
                "Autonomous escalation handling",
                "Continuous learning and adaptation",
                "Checkpoint-based persistence"
            ],
            "advantages": [
                "Industry-standard LangGraph framework",
                "Better state management and persistence",
                "More robust error handling and recovery",
                "Enhanced debugging and observability",
                "Built-in checkpointing and resumability"
            ]
        }
    
    # Helper methods for workflow capabilities
    def _get_required_inputs(self, workflow_id: str) -> List[str]:
        """Get required inputs for a workflow."""
        input_mapping = {
            "e2e-test-workflow": ["change_type", "diff_content", "changed_files"],
            "bug-triage-workflow": ["test_results", "clustering_method"],
            "performance-optimization-workflow": []
        }
        return input_mapping.get(workflow_id, [])
    
    def _get_expected_outputs(self, workflow_id: str) -> List[str]:
        """Get expected outputs for a workflow."""
        output_mapping = {
            "e2e-test-workflow": ["test_scenarios", "test_results", "failure_clusters", "learning_insights"],
            "bug-triage-workflow": ["failure_clusters", "root_cause_analysis", "bug_triage_results"],
            "performance-optimization-workflow": ["performance_analysis", "optimization_recommendations"]
        }
        return output_mapping.get(workflow_id, [])
    
    return app


# Create the app instance
app = create_langgraph_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
