"""Agent Coordinator - orchestrates agent communication and autonomous escalation."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, field
from enum import Enum
import json

from .base_agent import BaseAgent, AgentMessage, MessageType, AgentState
from .test_planner_agent import TestPlannerAgent
from .execution_agent import ExecutionAgent
from .diagnosis_agent import DiagnosisAgent
from .learning_agent import LearningAgent

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Coordination strategies for agent orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


class EscalationLevel(Enum):
    """Escalation levels for human intervention."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowDefinition:
    """Definition of a multi-agent workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
    escalation_threshold: EscalationLevel = EscalationLevel.MEDIUM
    timeout_seconds: int = 3600
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_retries": 3, "backoff_factor": 2})


@dataclass
class WorkflowExecution:
    """Execution state of a workflow."""
    execution_id: str
    workflow_definition: WorkflowDefinition
    current_step: int = 0
    status: str = "running"
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    escalation_level: EscalationLevel = EscalationLevel.NONE


class AgentCoordinator:
    """Coordinates communication and workflow between agents."""
    
    def __init__(self, coordinator_id: str = "coordinator-001"):
        self.coordinator_id = coordinator_id
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        
        # Communication handlers
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_RESPONSE: self._handle_task_response,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.ERROR_REPORT: self._handle_error_report,
            MessageType.ESCALATION_REQUEST: self._handle_escalation_request,
            MessageType.LEARNING_UPDATE: self._handle_learning_update,
            MessageType.COORDINATION_REQUEST: self._handle_coordination_request
        }
        
        # Escalation handlers
        self.escalation_handlers: Dict[EscalationLevel, Callable] = {
            EscalationLevel.LOW: self._handle_low_escalation,
            EscalationLevel.MEDIUM: self._handle_medium_escalation,
            EscalationLevel.HIGH: self._handle_high_escalation,
            EscalationLevel.CRITICAL: self._handle_critical_escalation
        }
        
        # Initialize default agents
        self._initialize_default_agents()
        
        # Initialize default workflows
        self._initialize_default_workflows()
        
        logger.info(f"Agent Coordinator initialized with {len(self.agents)} agents")
    
    def _initialize_default_agents(self):
        """Initialize default agents."""
        self.agents["test-planner"] = TestPlannerAgent("test-planner-001")
        self.agents["execution"] = ExecutionAgent("execution-001")
        self.agents["diagnosis"] = DiagnosisAgent("diagnosis-001")
        self.agents["learning"] = LearningAgent("learning-001")
        
        # Register agents with coordinator
        for agent in self.agents.values():
            agent._send_response = self._route_message
    
    def _initialize_default_workflows(self):
        """Initialize default workflows."""
        # End-to-end test workflow
        e2e_workflow = WorkflowDefinition(
            workflow_id="e2e-test-workflow",
            name="End-to-End Test Workflow",
            description="Complete test planning, execution, and analysis workflow",
            steps=[
                {
                    "step_id": "plan_tests",
                    "agent": "test-planner",
                    "task_type": "plan_tests_for_change",
                    "dependencies": [],
                    "timeout": 300
                },
                {
                    "step_id": "execute_tests",
                    "agent": "execution",
                    "task_type": "execute_test_scenarios",
                    "dependencies": ["plan_tests"],
                    "timeout": 1800
                },
                {
                    "step_id": "analyze_results",
                    "agent": "diagnosis",
                    "task_type": "cluster_test_failures",
                    "dependencies": ["execute_tests"],
                    "timeout": 600
                },
                {
                    "step_id": "learn_from_results",
                    "agent": "learning",
                    "task_type": "learn_from_feedback",
                    "dependencies": ["analyze_results"],
                    "timeout": 300
                }
            ],
            coordination_strategy=CoordinationStrategy.ADAPTIVE,
            escalation_threshold=EscalationLevel.MEDIUM
        )
        
        # Bug triage workflow
        bug_triage_workflow = WorkflowDefinition(
            workflow_id="bug-triage-workflow",
            name="Bug Triage Workflow",
            description="Automated bug triage and fix suggestion workflow",
            steps=[
                {
                    "step_id": "analyze_bugs",
                    "agent": "diagnosis",
                    "task_type": "triage_bugs",
                    "dependencies": [],
                    "timeout": 600
                },
                {
                    "step_id": "suggest_fixes",
                    "agent": "diagnosis",
                    "task_type": "suggest_fixes",
                    "dependencies": ["analyze_bugs"],
                    "timeout": 900
                },
                {
                    "step_id": "plan_validation_tests",
                    "agent": "test-planner",
                    "task_type": "plan_tests_for_ticket",
                    "dependencies": ["suggest_fixes"],
                    "timeout": 300
                }
            ],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            escalation_threshold=EscalationLevel.LOW
        )
        
        # Performance optimization workflow
        performance_workflow = WorkflowDefinition(
            workflow_id="performance-optimization-workflow",
            name="Performance Optimization Workflow",
            description="Continuous performance monitoring and optimization",
            steps=[
                {
                    "step_id": "analyze_performance",
                    "agent": "execution",
                    "task_type": "performance_optimization",
                    "dependencies": [],
                    "timeout": 1200
                },
                {
                    "step_id": "optimize_models",
                    "agent": "learning",
                    "task_type": "optimize_performance",
                    "dependencies": ["analyze_performance"],
                    "timeout": 1800
                },
                {
                    "step_id": "update_strategies",
                    "agent": "test-planner",
                    "task_type": "prioritize_test_scenarios",
                    "dependencies": ["optimize_models"],
                    "timeout": 600
                }
            ],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            escalation_threshold=EscalationLevel.HIGH
        )
        
        self.workflow_definitions["e2e-test"] = e2e_workflow
        self.workflow_definitions["bug-triage"] = bug_triage_workflow
        self.workflow_definitions["performance-optimization"] = performance_workflow
    
    async def start_coordination(self):
        """Start the coordination system."""
        logger.info("Starting Agent Coordinator")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        # Start workflow monitoring
        asyncio.create_task(self._workflow_monitoring_loop())
        
        # Start agent health monitoring
        asyncio.create_task(self._agent_health_monitoring_loop())
        
        logger.info("Agent Coordinator started successfully")
    
    async def _message_processing_loop(self):
        """Main message processing loop."""
        while True:
            try:
                # Wait for messages
                message = await self.message_queue.get()
                
                # Process message
                await self._process_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _workflow_monitoring_loop(self):
        """Monitor active workflows."""
        while True:
            try:
                current_time = datetime.now()
                
                for execution_id, execution in list(self.active_workflows.items()):
                    # Check for timeout
                    if (current_time - execution.start_time).total_seconds() > execution.workflow_definition.timeout_seconds:
                        logger.warning(f"Workflow {execution_id} timed out")
                        await self._handle_workflow_timeout(execution)
                    
                    # Check for stuck workflows
                    if execution.status == "running" and execution.current_step > 0:
                        last_activity = max(
                            (step.get("last_activity", execution.start_time) 
                             for step in execution.results.values() 
                             if isinstance(step, dict)),
                            default=execution.start_time
                        )
                        
                        if (current_time - last_activity).total_seconds() > 300:  # 5 minutes
                            logger.warning(f"Workflow {execution_id} appears stuck")
                            await self._handle_stuck_workflow(execution)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in workflow monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _agent_health_monitoring_loop(self):
        """Monitor agent health and availability."""
        while True:
            try:
                for agent_id, agent in self.agents.items():
                    # Check agent state
                    if agent.state == AgentState.ERROR:
                        logger.warning(f"Agent {agent_id} is in error state")
                        await self._handle_agent_error(agent)
                    
                    # Check if agent is responsive
                    if agent.state == AgentState.PROCESSING:
                        # Check if processing has been going on too long
                        if hasattr(agent, 'last_processing_start'):
                            processing_time = (datetime.now() - agent.last_processing_start).total_seconds()
                            if processing_time > 1800:  # 30 minutes
                                logger.warning(f"Agent {agent_id} has been processing for {processing_time} seconds")
                                await self._handle_agent_stuck(agent)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in agent health monitoring loop: {e}")
                await asyncio.sleep(120)
    
    async def _process_message(self, message: AgentMessage):
        """Process a message from an agent."""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task request messages."""
        task_data = message.payload.get("task_data", {})
        task_type = message.payload.get("task_type", "")
        requesting_agent = message.sender_id
        
        # Find suitable agent for the task
        suitable_agents = self._find_suitable_agents(task_type, task_data)
        
        if not suitable_agents:
            logger.warning(f"No suitable agents found for task type: {task_type}")
            await self._send_error_response(message, "No suitable agents available")
            return
        
        # Select best agent based on availability and capability
        selected_agent = self._select_best_agent(suitable_agents, task_type, task_data)
        
        # Forward task to selected agent
        task_message = AgentMessage(
            sender_id=self.coordinator_id,
            receiver_id=selected_agent.agent_id,
            message_type=MessageType.TASK_REQUEST,
            payload={
                "task_data": task_data,
                "task_type": task_type,
                "original_sender": requesting_agent
            },
            correlation_id=message.message_id
        )
        
        await self._route_message(task_message)
    
    async def _handle_task_response(self, message: AgentMessage):
        """Handle task response messages."""
        original_sender = message.payload.get("original_sender")
        if original_sender:
            # Forward response to original sender
            response_message = AgentMessage(
                sender_id=self.coordinator_id,
                receiver_id=original_sender,
                message_type=MessageType.TASK_RESPONSE,
                payload=message.payload,
                correlation_id=message.correlation_id
            )
            await self._route_message(response_message)
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update messages."""
        agent_id = message.sender_id
        status_data = message.payload
        
        logger.debug(f"Status update from agent {agent_id}: {status_data}")
        
        # Update agent status in coordinator
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            # Update agent status based on the update
        
        # Broadcast status update to interested parties
        await self._broadcast_status_update(agent_id, status_data)
    
    async def _handle_error_report(self, message: AgentMessage):
        """Handle error report messages."""
        agent_id = message.sender_id
        error_data = message.payload
        
        logger.error(f"Error report from agent {agent_id}: {error_data}")
        
        # Determine escalation level
        escalation_level = self._determine_escalation_level(error_data)
        
        # Handle escalation
        if escalation_level != EscalationLevel.NONE:
            await self._handle_escalation(escalation_level, agent_id, error_data)
    
    async def _handle_escalation_request(self, message: AgentMessage):
        """Handle escalation request messages."""
        agent_id = message.sender_id
        escalation_data = message.payload
        
        escalation_level = EscalationLevel(escalation_data.get("escalation_level", "medium"))
        
        logger.warning(f"Escalation request from agent {agent_id}: {escalation_level}")
        
        await self._handle_escalation(escalation_level, agent_id, escalation_data)
    
    async def _handle_learning_update(self, message: AgentMessage):
        """Handle learning update messages."""
        learning_data = message.payload
        
        # Broadcast learning updates to relevant agents
        await self._broadcast_learning_update(learning_data)
    
    async def _handle_coordination_request(self, message: AgentMessage):
        """Handle coordination request messages."""
        request_type = message.payload.get("request_type", "")
        
        if request_type == "start_workflow":
            await self._handle_workflow_start_request(message)
        elif request_type == "stop_workflow":
            await self._handle_workflow_stop_request(message)
        elif request_type == "get_status":
            await self._handle_status_request(message)
        else:
            logger.warning(f"Unknown coordination request type: {request_type}")
    
    async def _handle_escalation(self, level: EscalationLevel, agent_id: str, escalation_data: Dict[str, Any]):
        """Handle escalation based on level."""
        handler = self.escalation_handlers.get(level)
        if handler:
            await handler(agent_id, escalation_data)
        else:
            logger.error(f"No handler for escalation level: {level}")
    
    async def _handle_low_escalation(self, agent_id: str, escalation_data: Dict[str, Any]):
        """Handle low-level escalation."""
        logger.info(f"Low escalation from {agent_id}: {escalation_data}")
        
        # Log escalation
        await self._log_escalation(agent_id, EscalationLevel.LOW, escalation_data)
        
        # Attempt automatic resolution
        await self._attempt_automatic_resolution(agent_id, escalation_data)
    
    async def _handle_medium_escalation(self, agent_id: str, escalation_data: Dict[str, Any]):
        """Handle medium-level escalation."""
        logger.warning(f"Medium escalation from {agent_id}: {escalation_data}")
        
        # Log escalation
        await self._log_escalation(agent_id, EscalationLevel.MEDIUM, escalation_data)
        
        # Notify team leads
        await self._notify_team_leads(escalation_data, level="medium")
        
        # Attempt automatic resolution with more resources
        await self._attempt_automatic_resolution(agent_id, escalation_data, use_more_resources=True)
    
    async def _handle_high_escalation(self, agent_id: str, escalation_data: Dict[str, Any]):
        """Handle high-level escalation."""
        logger.error(f"High escalation from {agent_id}: {escalation_data}")
        
        # Log escalation
        await self._log_escalation(agent_id, EscalationLevel.HIGH, escalation_data)
        
        # Notify senior team members
        await self._notify_senior_team(escalation_data, level="high")
        
        # Create incident ticket
        await self._create_incident_ticket(escalation_data)
    
    async def _handle_critical_escalation(self, agent_id: str, escalation_data: Dict[str, Any]):
        """Handle critical-level escalation."""
        logger.critical(f"CRITICAL escalation from {agent_id}: {escalation_data}")
        
        # Log escalation
        await self._log_escalation(agent_id, EscalationLevel.CRITICAL, escalation_data)
        
        # Immediate notification to all stakeholders
        await self._notify_all_stakeholders(escalation_data)
        
        # Create high-priority incident
        await self._create_critical_incident(escalation_data)
        
        # Activate emergency procedures
        await self._activate_emergency_procedures(escalation_data)
    
    async def start_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> str:
        """Start a workflow execution."""
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_def = self.workflow_definitions[workflow_id]
        execution_id = f"{workflow_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_definition=workflow_def
        )
        
        self.active_workflows[execution_id] = execution
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow(execution, input_data))
        
        logger.info(f"Started workflow {workflow_id} with execution ID: {execution_id}")
        return execution_id
    
    async def _execute_workflow(self, execution: WorkflowExecution, input_data: Dict[str, Any]):
        """Execute a workflow."""
        try:
            workflow_def = execution.workflow_definition
            current_data = input_data.copy()
            
            for step_index, step in enumerate(workflow_def.steps):
                execution.current_step = step_index
                
                # Check dependencies
                dependencies = step.get("dependencies", [])
                if not self._check_dependencies(execution, dependencies):
                    logger.warning(f"Dependencies not met for step {step['step_id']}")
                    continue
                
                # Execute step
                step_result = await self._execute_workflow_step(execution, step, current_data)
                
                # Store result
                execution.results[step["step_id"]] = step_result
                
                # Update data for next step
                if step_result and step_result.get("success", False):
                    current_data.update(step_result.get("data", {}))
                else:
                    # Handle step failure
                    execution.errors.append(f"Step {step['step_id']} failed")
                    
                    # Check if we should continue or abort
                    if workflow_def.escalation_threshold == EscalationLevel.LOW:
                        break  # Abort on any failure
            
            # Mark workflow as completed
            execution.status = "completed" if not execution.errors else "failed"
            execution.end_time = datetime.now()
            
            logger.info(f"Workflow {execution.execution_id} completed with status: {execution.status}")
            
        except Exception as e:
            logger.error(f"Error executing workflow {execution.execution_id}: {e}")
            execution.status = "error"
            execution.errors.append(str(e))
            execution.end_time = datetime.now()
    
    async def _execute_workflow_step(self, execution: WorkflowExecution, step: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        agent_id = step["agent"]
        task_type = step["task_type"]
        timeout = step.get("timeout", 300)
        
        if agent_id not in self.agents:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        
        # Create task message
        task_message = AgentMessage(
            sender_id=self.coordinator_id,
            receiver_id=agent_id,
            message_type=MessageType.TASK_REQUEST,
            payload={
                "task_data": input_data,
                "task_type": task_type
            }
        )
        
        try:
            # Execute task with timeout
            response = await asyncio.wait_for(
                agent.receive_message(task_message),
                timeout=timeout
            )
            
            if response and response.message_type == MessageType.TASK_RESPONSE:
                payload = response.payload
                return {
                    "success": payload.get("status") == "completed",
                    "data": payload.get("result", {}),
                    "step_id": step["step_id"],
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid response from agent",
                    "step_id": step["step_id"],
                    "agent_id": agent_id
                }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Step timed out after {timeout} seconds",
                "step_id": step["step_id"],
                "agent_id": agent_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step_id": step["step_id"],
                "agent_id": agent_id
            }
    
    def _check_dependencies(self, execution: WorkflowExecution, dependencies: List[str]) -> bool:
        """Check if workflow step dependencies are met."""
        for dep in dependencies:
            if dep not in execution.results:
                return False
            
            dep_result = execution.results[dep]
            if not dep_result.get("success", False):
                return False
        
        return True
    
    def _find_suitable_agents(self, task_type: str, task_data: Dict[str, Any]) -> List[BaseAgent]:
        """Find agents suitable for a task."""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.can_handle_task(task_type, task_data):
                suitable_agents.append(agent)
        
        return suitable_agents
    
    def _select_best_agent(self, agents: List[BaseAgent], task_type: str, task_data: Dict[str, Any]) -> BaseAgent:
        """Select the best agent for a task."""
        # Simple selection based on availability and metrics
        # In practice, this could be more sophisticated
        
        available_agents = [a for a in agents if a.state == AgentState.IDLE]
        
        if available_agents:
            # Select agent with best success rate
            return max(available_agents, key=lambda a: a.metrics.success_rate)
        else:
            # Select agent with fewest active tasks
            return min(agents, key=lambda a: len(a.active_tasks))
    
    async def _route_message(self, message: AgentMessage):
        """Route a message to the appropriate agent."""
        if message.receiver_id == self.coordinator_id:
            await self.message_queue.put(message)
        elif message.receiver_id in self.agents:
            agent = self.agents[message.receiver_id]
            await agent.receive_message(message)
        else:
            logger.warning(f"Unknown message receiver: {message.receiver_id}")
    
    async def _broadcast_status_update(self, agent_id: str, status_data: Dict[str, Any]):
        """Broadcast status update to interested parties."""
        # This would implement broadcasting logic
        logger.debug(f"Broadcasting status update from {agent_id}")
    
    async def _broadcast_learning_update(self, learning_data: Dict[str, Any]):
        """Broadcast learning updates to relevant agents."""
        # This would implement learning update broadcasting
        logger.debug("Broadcasting learning update")
    
    async def _log_escalation(self, agent_id: str, level: EscalationLevel, escalation_data: Dict[str, Any]):
        """Log escalation for audit purposes."""
        escalation_log = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "escalation_level": level.value,
            "escalation_data": escalation_data
        }
        
        # This would log to a persistent store
        logger.info(f"Escalation logged: {escalation_log}")
    
    async def _attempt_automatic_resolution(self, agent_id: str, escalation_data: Dict[str, Any], use_more_resources: bool = False):
        """Attempt automatic resolution of escalation."""
        logger.info(f"Attempting automatic resolution for {agent_id}")
        
        # This would implement automatic resolution logic
        # For example, restarting agents, reallocating resources, etc.
    
    async def _notify_team_leads(self, escalation_data: Dict[str, Any], level: str):
        """Notify team leads of escalation."""
        logger.info(f"Notifying team leads of {level} escalation")
        
        # This would implement notification logic
        # Email, Slack, etc.
    
    async def _notify_senior_team(self, escalation_data: Dict[str, Any], level: str):
        """Notify senior team members of escalation."""
        logger.info(f"Notifying senior team of {level} escalation")
        
        # This would implement notification logic
    
    async def _notify_all_stakeholders(self, escalation_data: Dict[str, Any]):
        """Notify all stakeholders of critical escalation."""
        logger.critical("Notifying all stakeholders of critical escalation")
        
        # This would implement immediate notification logic
    
    async def _create_incident_ticket(self, escalation_data: Dict[str, Any]):
        """Create incident ticket for escalation."""
        logger.info("Creating incident ticket")
        
        # This would integrate with ticketing systems
    
    async def _create_critical_incident(self, escalation_data: Dict[str, Any]):
        """Create critical incident for high-priority escalation."""
        logger.critical("Creating critical incident ticket")
        
        # This would create high-priority tickets
    
    async def _activate_emergency_procedures(self, escalation_data: Dict[str, Any]):
        """Activate emergency procedures for critical escalation."""
        logger.critical("Activating emergency procedures")
        
        # This would implement emergency response procedures
    
    async def _handle_workflow_timeout(self, execution: WorkflowExecution):
        """Handle workflow timeout."""
        execution.status = "timeout"
        execution.end_time = datetime.now()
        
        logger.warning(f"Workflow {execution.execution_id} timed out")
        
        # Escalate if needed
        await self._handle_escalation(EscalationLevel.MEDIUM, "coordinator", {
            "type": "workflow_timeout",
            "workflow_id": execution.execution_id,
            "timeout_duration": execution.workflow_definition.timeout_seconds
        })
    
    async def _handle_stuck_workflow(self, execution: WorkflowExecution):
        """Handle stuck workflow."""
        logger.warning(f"Workflow {execution.execution_id} appears stuck")
        
        # Attempt recovery
        await self._attempt_workflow_recovery(execution)
    
    async def _attempt_workflow_recovery(self, execution: WorkflowExecution):
        """Attempt to recover a stuck workflow."""
        logger.info(f"Attempting recovery for workflow {execution.execution_id}")
        
        # This would implement recovery logic
        # For example, skipping problematic steps, retrying with different parameters, etc.
    
    async def _handle_agent_error(self, agent: BaseAgent):
        """Handle agent in error state."""
        logger.error(f"Agent {agent.agent_id} is in error state")
        
        # Attempt to restart agent
        await self._attempt_agent_restart(agent)
    
    async def _handle_agent_stuck(self, agent: BaseAgent):
        """Handle stuck agent."""
        logger.warning(f"Agent {agent.agent_id} appears stuck")
        
        # Attempt to unstick agent
        await self._attempt_agent_unstick(agent)
    
    async def _attempt_agent_restart(self, agent: BaseAgent):
        """Attempt to restart an agent."""
        logger.info(f"Attempting to restart agent {agent.agent_id}")
        
        # This would implement agent restart logic
    
    async def _attempt_agent_unstick(self, agent: BaseAgent):
        """Attempt to unstick an agent."""
        logger.info(f"Attempting to unstick agent {agent.agent_id}")
        
        # This would implement agent unsticking logic
    
    async def _send_error_response(self, original_message: AgentMessage, error_message: str):
        """Send error response to original sender."""
        error_response = AgentMessage(
            sender_id=self.coordinator_id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.ERROR_REPORT,
            payload={"error": error_message},
            correlation_id=original_message.message_id
        )
        
        await self._route_message(error_response)
    
    def _determine_escalation_level(self, error_data: Dict[str, Any]) -> EscalationLevel:
        """Determine escalation level based on error data."""
        error_type = error_data.get("error_type", "")
        severity = error_data.get("severity", "medium")
        
        if severity == "critical" or "critical" in error_type.lower():
            return EscalationLevel.CRITICAL
        elif severity == "high":
            return EscalationLevel.HIGH
        elif severity == "medium":
            return EscalationLevel.MEDIUM
        elif severity == "low":
            return EscalationLevel.LOW
        else:
            return EscalationLevel.NONE
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "coordinator_id": self.coordinator_id,
            "active_agents": len(self.agents),
            "agent_status": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            },
            "active_workflows": len(self.active_workflows),
            "workflow_status": {
                exec_id: {
                    "workflow_id": exec.workflow_definition.workflow_id,
                    "status": exec.status,
                    "current_step": exec.current_step,
                    "start_time": exec.start_time.isoformat(),
                    "errors": len(exec.errors)
                }
                for exec_id, exec in self.active_workflows.items()
            },
            "message_queue_size": self.message_queue.qsize(),
            "coordinator_uptime": datetime.now().isoformat()
        }
    
    async def stop_coordinator(self):
        """Stop the coordinator and cleanup resources."""
        logger.info("Stopping Agent Coordinator")
        
        # Stop all active workflows
        for execution in self.active_workflows.values():
            execution.status = "stopped"
            execution.end_time = datetime.now()
        
        # Cleanup agents
        for agent in self.agents.values():
            agent.cleanup()
        
        logger.info("Agent Coordinator stopped")
