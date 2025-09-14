"""LangGraph workflow orchestration for the multi-agent system."""

import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import TestAutomationState, create_initial_state, update_state_timestamp
from .test_planner_agent import TestPlannerAgent
from .execution_agent import ExecutionAgent
from .diagnosis_agent import DiagnosisAgent
from .learning_agent import LearningAgent

logger = logging.getLogger(__name__)


class TestAutomationWorkflow:
    """LangGraph-based workflow for test automation with multi-agent coordination."""
    
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self.agents = {}
        self.workflow_definitions = {}
        
        # Initialize agents
        self._initialize_agents()
        
        # Build workflow graph
        self._build_workflow_graph()
        
        logger.info("TestAutomationWorkflow initialized")
    
    def _initialize_agents(self):
        """Initialize all agents."""
        self.agents = {
            "test_planner": TestPlannerAgent("test-planner"),
            "execution": ExecutionAgent("execution"),
            "diagnosis": DiagnosisAgent("diagnosis"),
            "learning": LearningAgent("learning")
        }
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def _build_workflow_graph(self):
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(TestAutomationState)
        
        # Add nodes for each agent
        workflow.add_node("test_planner", self._test_planner_node)
        workflow.add_node("execution", self._execution_node)
        workflow.add_node("diagnosis", self._diagnosis_node)
        workflow.add_node("learning", self._learning_node)
        
        # Add coordination nodes
        workflow.add_node("coordination", self._coordination_node)
        workflow.add_node("escalation", self._escalation_node)
        workflow.add_node("error_handling", self._error_handling_node)
        
        # Add routing logic
        workflow.add_conditional_edges(
            "test_planner",
            self._route_after_planning,
            {
                "continue": "test_planner",
                "next": "execution",
                "escalate": "escalation",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "execution",
            self._route_after_execution,
            {
                "continue": "execution",
                "next": "diagnosis",
                "escalate": "escalation",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "diagnosis",
            self._route_after_diagnosis,
            {
                "continue": "diagnosis",
                "next": "learning",
                "escalate": "escalation",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "learning",
            self._route_after_learning,
            {
                "continue": "learning",
                "next": "coordination",
                "escalate": "escalation",
                "error": "error_handling"
            }
        )
        
        # Add final edges
        workflow.add_conditional_edges(
            "coordination",
            self._route_after_coordination,
            {
                "complete": END,
                "escalate": "escalation",
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("escalation", END)
        workflow.add_edge("error_handling", END)
        
        # Set entry point
        workflow.set_entry_point("test_planner")
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
        
        logger.info("Workflow graph built successfully")
    
    # Node functions
    
    async def _test_planner_node(self, state: TestAutomationState) -> TestAutomationState:
        """Test planner node."""
        logger.info("Executing test planner node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Starting test planning phase"))
            
            # Execute test planning based on change type
            change_type = state.get("code_changes", {}).get("change_type", "feature_implementation")
            
            if change_type == "feature_implementation":
                state = await self.agents["test_planner"].plan_tests_for_change(state)
            elif change_type == "bug_fix":
                state = await self.agents["test_planner"].plan_tests_for_ticket(state)
            elif change_type == "regression":
                state = await self.agents["test_planner"].plan_regression_tests(state)
            else:
                state = await self.agents["test_planner"].plan_tests_for_change(state)
            
            # Perform risk assessment
            state = await self.agents["test_planner"].assess_risk(state)
            
            # Analyze test coverage
            state = await self.agents["test_planner"].analyze_test_coverage(state)
            
            state["messages"].append(AIMessage(content="Test planning phase completed"))
            
        except Exception as e:
            logger.error(f"Error in test planner node: {e}")
            state["errors"].append(f"Test planner error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Test planning failed: {str(e)}"))
        
        return state
    
    async def _execution_node(self, state: TestAutomationState) -> TestAutomationState:
        """Execution node."""
        logger.info("Executing execution node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Starting test execution phase"))
            
            # Execute test scenarios
            state = await self.agents["execution"].execute_test_scenarios(state)
            
            # Heal failing tests if any
            failed_results = [r for r in state["test_results"] if r.status == "failed"]
            if failed_results:
                state = await self.agents["execution"].heal_failing_tests(state)
            
            # Optimize execution performance
            state = await self.agents["execution"].optimize_execution_performance(state)
            
            state["messages"].append(AIMessage(content="Test execution phase completed"))
            
        except Exception as e:
            logger.error(f"Error in execution node: {e}")
            state["errors"].append(f"Execution error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Test execution failed: {str(e)}"))
        
        return state
    
    async def _diagnosis_node(self, state: TestAutomationState) -> TestAutomationState:
        """Diagnosis node."""
        logger.info("Executing diagnosis node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Starting failure diagnosis phase"))
            
            # Cluster test failures
            state = await self.agents["diagnosis"].cluster_test_failures(state)
            
            # Analyze root causes
            state = await self.agents["diagnosis"].analyze_root_causes(state)
            
            # Triage bugs
            state = await self.agents["diagnosis"].triage_bugs(state)
            
            # Suggest fixes
            state = await self.agents["diagnosis"].suggest_fixes(state)
            
            # Analyze failure trends
            state = await self.agents["diagnosis"].analyze_failure_trends(state)
            
            state["messages"].append(AIMessage(content="Failure diagnosis phase completed"))
            
        except Exception as e:
            logger.error(f"Error in diagnosis node: {e}")
            state["errors"].append(f"Diagnosis error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Failure diagnosis failed: {str(e)}"))
        
        return state
    
    async def _learning_node(self, state: TestAutomationState) -> TestAutomationState:
        """Learning node."""
        logger.info("Executing learning node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Starting learning phase"))
            
            # Learn from feedback
            if state["feedback_data"]:
                state = await self.agents["learning"].learn_from_feedback(state)
            
            # Learn from patterns
            state = await self.agents["learning"].learn_from_patterns(state)
            
            # Optimize models
            state = await self.agents["learning"].optimize_models(state)
            
            # Sync knowledge base
            state = await self.agents["learning"].sync_knowledge_base(state)
            
            state["messages"].append(AIMessage(content="Learning phase completed"))
            
        except Exception as e:
            logger.error(f"Error in learning node: {e}")
            state["errors"].append(f"Learning error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Learning failed: {str(e)}"))
        
        return state
    
    async def _coordination_node(self, state: TestAutomationState) -> TestAutomationState:
        """Coordination node for final workflow coordination."""
        logger.info("Executing coordination node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Starting workflow coordination"))
            
            # Generate final summary
            summary = self._generate_workflow_summary(state)
            state["learning_insights"]["workflow_summary"] = summary
            
            # Update workflow status
            if state["workflow"]:
                state["workflow"]["status"] = "completed"
                state["workflow"]["end_time"] = datetime.now()
            
            state["messages"].append(AIMessage(content="Workflow coordination completed successfully"))
            
        except Exception as e:
            logger.error(f"Error in coordination node: {e}")
            state["errors"].append(f"Coordination error: {str(e)}")
        
        return state
    
    async def _escalation_node(self, state: TestAutomationState) -> TestAutomationState:
        """Escalation node for handling escalations."""
        logger.info("Executing escalation node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Handling escalation"))
            
            # Determine escalation level
            escalation_level = self._determine_escalation_level(state)
            
            # Handle escalation based on level
            if escalation_level == "critical":
                await self._handle_critical_escalation(state)
            elif escalation_level == "high":
                await self._handle_high_escalation(state)
            elif escalation_level == "medium":
                await self._handle_medium_escalation(state)
            else:
                await self._handle_low_escalation(state)
            
            state["messages"].append(AIMessage(content=f"Escalation handled at {escalation_level} level"))
            
        except Exception as e:
            logger.error(f"Error in escalation node: {e}")
            state["errors"].append(f"Escalation error: {str(e)}")
        
        return state
    
    async def _error_handling_node(self, state: TestAutomationState) -> TestAutomationState:
        """Error handling node."""
        logger.info("Executing error handling node")
        
        try:
            # Update state timestamp
            state = update_state_timestamp(state)
            
            # Add system message
            state["messages"].append(SystemMessage(content="Handling errors"))
            
            # Log errors
            error_summary = {
                "total_errors": len(state["errors"]),
                "failed_tasks": len(state["failed_tasks"]),
                "error_types": self._categorize_errors(state["errors"])
            }
            
            state["learning_insights"]["error_summary"] = error_summary
            
            # Attempt error recovery
            recovery_attempted = await self._attempt_error_recovery(state)
            
            state["messages"].append(AIMessage(content=f"Error handling completed, recovery attempted: {recovery_attempted}"))
            
        except Exception as e:
            logger.error(f"Error in error handling node: {e}")
            state["errors"].append(f"Error handling error: {str(e)}")
        
        return state
    
    # Routing functions
    
    def _route_after_planning(self, state: TestAutomationState) -> Literal["continue", "next", "escalate", "error"]:
        """Route after test planning."""
        if state["errors"] and len(state["errors"]) > 2:
            return "error"
        elif state.get("escalation_needed", False):
            return "escalate"
        elif not state["test_scenarios"]:
            return "error"
        else:
            return "next"
    
    def _route_after_execution(self, state: TestAutomationState) -> Literal["continue", "next", "escalate", "error"]:
        """Route after test execution."""
        if state["errors"] and len(state["errors"]) > 3:
            return "error"
        elif state.get("escalation_needed", False):
            return "escalate"
        elif not state["test_results"]:
            return "error"
        else:
            return "next"
    
    def _route_after_diagnosis(self, state: TestAutomationState) -> Literal["continue", "next", "escalate", "error"]:
        """Route after diagnosis."""
        if state["errors"] and len(state["errors"]) > 4:
            return "error"
        elif state.get("escalation_needed", False):
            return "escalate"
        elif not state["failure_clusters"] and not state["learning_insights"].get("root_cause_analysis"):
            return "error"
        else:
            return "next"
    
    def _route_after_learning(self, state: TestAutomationState) -> Literal["continue", "next", "escalate", "error"]:
        """Route after learning."""
        if state["errors"] and len(state["errors"]) > 5:
            return "error"
        elif state.get("escalation_needed", False):
            return "escalate"
        else:
            return "next"
    
    def _route_after_coordination(self, state: TestAutomationState) -> Literal["complete", "escalate", "error"]:
        """Route after coordination."""
        if state["errors"] and len(state["errors"]) > 6:
            return "error"
        elif state.get("escalation_needed", False):
            return "escalate"
        else:
            return "complete"
    
    # Helper methods
    
    def _generate_workflow_summary(self, state: TestAutomationState) -> Dict[str, Any]:
        """Generate workflow summary."""
        return {
            "workflow_id": state.get("workflow", {}).get("workflow_id", "unknown"),
            "execution_id": state.get("workflow", {}).get("execution_id", "unknown"),
            "total_scenarios": len(state["test_scenarios"]),
            "total_results": len(state["test_results"]),
            "success_rate": len([r for r in state["test_results"] if r.status == "passed"]) / max(1, len(state["test_results"])),
            "failure_clusters": len(state["failure_clusters"]),
            "tasks_completed": len(state["completed_tasks"]),
            "tasks_failed": len(state["failed_tasks"]),
            "errors": len(state["errors"]),
            "escalation_needed": state.get("escalation_needed", False),
            "execution_time": (datetime.now() - state["created_at"]).total_seconds(),
            "insights_generated": len(state["learning_insights"])
        }
    
    def _determine_escalation_level(self, state: TestAutomationState) -> str:
        """Determine escalation level based on state."""
        if len(state["failed_tasks"]) > 5 or len(state["errors"]) > 10:
            return "critical"
        elif len(state["failed_tasks"]) > 3 or len(state["errors"]) > 5:
            return "high"
        elif len(state["failed_tasks"]) > 1 or len(state["errors"]) > 2:
            return "medium"
        else:
            return "low"
    
    def _categorize_errors(self, errors: List[str]) -> Dict[str, int]:
        """Categorize errors by type."""
        categories = {}
        for error in errors:
            if "timeout" in error.lower():
                categories["timeout"] = categories.get("timeout", 0) + 1
            elif "connection" in error.lower():
                categories["connection"] = categories.get("connection", 0) + 1
            elif "permission" in error.lower():
                categories["permission"] = categories.get("permission", 0) + 1
            else:
                categories["other"] = categories.get("other", 0) + 1
        return categories
    
    async def _handle_critical_escalation(self, state: TestAutomationState):
        """Handle critical escalation."""
        logger.critical("Handling critical escalation")
        # Implementation would notify all stakeholders immediately
    
    async def _handle_high_escalation(self, state: TestAutomationState):
        """Handle high escalation."""
        logger.error("Handling high escalation")
        # Implementation would notify senior team members
    
    async def _handle_medium_escalation(self, state: TestAutomationState):
        """Handle medium escalation."""
        logger.warning("Handling medium escalation")
        # Implementation would notify team leads
    
    async def _handle_low_escalation(self, state: TestAutomationState):
        """Handle low escalation."""
        logger.info("Handling low escalation")
        # Implementation would log and attempt automatic resolution
    
    async def _attempt_error_recovery(self, state: TestAutomationState) -> bool:
        """Attempt error recovery."""
        logger.info("Attempting error recovery")
        
        # Simple recovery logic - in practice, this would be more sophisticated
        if state["failed_tasks"]:
            # Retry failed tasks
            return True
        
        return False
    
    # Public methods
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any], 
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow with the given input data."""
        
        # Create initial state
        initial_state = create_initial_state(workflow_id, config=config)
        
        # Set input data
        initial_state.update(input_data)
        
        # Add initial message
        initial_state["messages"].append(HumanMessage(content=f"Starting workflow: {workflow_id}"))
        
        try:
            # Execute the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Generate result
            result = {
                "workflow_id": workflow_id,
                "execution_id": final_state.get("workflow", {}).get("execution_id", "unknown"),
                "status": "completed",
                "summary": final_state.get("learning_insights", {}).get("workflow_summary", {}),
                "messages": [msg.content for msg in final_state["messages"] if hasattr(msg, 'content')],
                "errors": final_state["errors"],
                "escalation_needed": final_state.get("escalation_needed", False),
                "results": {
                    "test_scenarios": len(final_state["test_scenarios"]),
                    "test_results": len(final_state["test_results"]),
                    "failure_clusters": len(final_state["failure_clusters"]),
                    "tasks_completed": len(final_state["completed_tasks"]),
                    "tasks_failed": len(final_state["failed_tasks"])
                }
            }
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_id": initial_state.get("workflow", {}).get("execution_id", "unknown")
            }
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow status by execution ID."""
        # This would retrieve state from memory/checkpoint
        return {
            "execution_id": execution_id,
            "status": "running",  # Would be retrieved from actual state
            "current_step": 2,
            "total_steps": 4,
            "progress": 0.5
        }
    
    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get list of available workflows."""
        return [
            {
                "workflow_id": "e2e-test-workflow",
                "name": "End-to-End Test Workflow",
                "description": "Complete test planning, execution, and analysis workflow",
                "steps": ["planning", "execution", "diagnosis", "learning"],
                "estimated_duration": "30-60 minutes"
            },
            {
                "workflow_id": "bug-triage-workflow",
                "name": "Bug Triage Workflow",
                "description": "Automated bug triage and fix suggestion workflow",
                "steps": ["diagnosis", "triage", "fix_suggestions"],
                "estimated_duration": "10-20 minutes"
            },
            {
                "workflow_id": "performance-optimization-workflow",
                "name": "Performance Optimization Workflow",
                "description": "Continuous performance monitoring and optimization",
                "steps": ["execution", "learning", "optimization"],
                "estimated_duration": "20-40 minutes"
            }
        ]
