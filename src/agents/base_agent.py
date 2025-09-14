"""Base agent class for the multi-agent architecture."""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class MessageType(Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    ESCALATION_REQUEST = "escalation_request"
    LEARNING_UPDATE = "learning_update"
    COORDINATION_REQUEST = "coordination_request"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, 10 being highest
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    timeout_seconds: int = 300


@dataclass
class AgentCapabilities:
    """Agent capability definitions."""
    can_plan_tests: bool = False
    can_execute_tests: bool = False
    can_diagnose_failures: bool = False
    can_learn_from_data: bool = False
    can_coordinate_agents: bool = False
    supported_platforms: List[str] = field(default_factory=list)
    supported_frameworks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    escalation_count: int = 0


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, agent_id: str, name: str, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=capabilities.max_concurrent_tasks)
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        logger.info(f"Initialized agent {self.name} (ID: {self.agent_id})")
    
    def _setup_message_handlers(self):
        """Setup default message handlers."""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_task_request
        self.message_handlers[MessageType.STATUS_UPDATE] = self._handle_status_update
        self.message_handlers[MessageType.ERROR_REPORT] = self._handle_error_report
        self.message_handlers[MessageType.LEARNING_UPDATE] = self._handle_learning_update
    
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this agent can handle a specific task type."""
        pass
    
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message from another agent."""
        try:
            logger.debug(f"Agent {self.name} received message: {message.message_type}")
            
            handler = self.message_handlers.get(message.message_type)
            if handler:
                response = await handler(message)
                self.metrics.last_activity = datetime.now()
                return response
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message in {self.name}: {e}")
            await self._escalate_error(f"Message processing error: {e}", message)
            return None
    
    async def _handle_task_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle task request messages."""
        task_data = message.payload.get("task_data", {})
        task_type = message.payload.get("task_type", "")
        
        if not self.can_handle_task(task_type, task_data):
            logger.warning(f"Agent {self.name} cannot handle task type: {task_type}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={"status": "rejected", "reason": "Cannot handle task type"},
                correlation_id=message.message_id
            )
        
        # Check if we can accept the task
        if len(self.active_tasks) >= self.capabilities.max_concurrent_tasks:
            logger.warning(f"Agent {self.name} is at capacity")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={"status": "rejected", "reason": "Agent at capacity"},
                correlation_id=message.message_id
            )
        
        # Accept the task
        task_id = str(uuid.uuid4())
        self.active_tasks[task_id] = {
            "message": message,
            "start_time": datetime.now(),
            "status": "accepted"
        }
        
        # Process task asynchronously
        asyncio.create_task(self._execute_task(task_id, task_data))
        
        return AgentMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            payload={"status": "accepted", "task_id": task_id},
            correlation_id=message.message_id
        )
    
    async def _execute_task(self, task_id: str, task_data: Dict[str, Any]):
        """Execute a task and send response."""
        try:
            self.state = AgentState.PROCESSING
            logger.info(f"Agent {self.name} starting task {task_id}")
            
            # Process the task
            result = await self.process_task(task_data)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["result"] = result
            self.active_tasks[task_id]["end_time"] = datetime.now()
            
            # Update metrics
            self.metrics.tasks_completed += 1
            execution_time = (self.active_tasks[task_id]["end_time"] - 
                            self.active_tasks[task_id]["start_time"]).total_seconds()
            self._update_average_execution_time(execution_time)
            
            # Send response
            original_message = self.active_tasks[task_id]["message"]
            response = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={
                    "status": "completed",
                    "task_id": task_id,
                    "result": result
                },
                correlation_id=original_message.message_id
            )
            
            # Send response (this would be handled by the coordinator)
            await self._send_response(response)
            
            logger.info(f"Agent {self.name} completed task {task_id}")
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed task {task_id}: {e}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            self.active_tasks[task_id]["end_time"] = datetime.now()
            
            # Update metrics
            self.metrics.tasks_failed += 1
            
            # Send error response
            original_message = self.active_tasks[task_id]["message"]
            error_response = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.ERROR_REPORT,
                payload={
                    "status": "failed",
                    "task_id": task_id,
                    "error": str(e)
                },
                correlation_id=original_message.message_id
            )
            
            await self._send_response(error_response)
            
            # Escalate if needed
            await self._escalate_error(f"Task execution failed: {e}", original_message)
        
        finally:
            self.state = AgentState.IDLE
    
    async def _handle_status_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle status update messages."""
        logger.debug(f"Agent {self.name} received status update from {message.sender_id}")
        # Agents can process status updates as needed
        return None
    
    async def _handle_error_report(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle error report messages."""
        logger.warning(f"Agent {self.name} received error report from {message.sender_id}: {message.payload}")
        # Agents can process error reports as needed
        return None
    
    async def _handle_learning_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle learning update messages."""
        logger.info(f"Agent {self.name} received learning update from {message.sender_id}")
        # Agents can process learning updates as needed
        return None
    
    async def _escalate_error(self, error_message: str, original_message: AgentMessage):
        """Escalate errors to human operators or coordinator."""
        try:
            escalation_message = AgentMessage(
                sender_id=self.agent_id,
                receiver_id="coordinator",
                message_type=MessageType.ESCALATION_REQUEST,
                payload={
                    "error": error_message,
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "original_message": original_message.payload,
                    "timestamp": datetime.now().isoformat()
                },
                priority=9  # High priority for escalations
            )
            
            await self._send_response(escalation_message)
            self.metrics.escalation_count += 1
            self.state = AgentState.ESCALATED
            
            logger.warning(f"Agent {self.name} escalated error: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to escalate error from {self.name}: {e}")
    
    async def _send_response(self, message: AgentMessage):
        """Send response message (to be implemented by coordinator integration)."""
        # This would be handled by the agent coordinator
        logger.debug(f"Agent {self.name} sending message to {message.receiver_id}")
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric."""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            current_avg = self.metrics.average_execution_time
            self.metrics.average_execution_time = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "active_tasks": len(self.active_tasks),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": (
                    self.metrics.tasks_completed / 
                    max(1, self.metrics.tasks_completed + self.metrics.tasks_failed)
                ),
                "average_execution_time": self.metrics.average_execution_time,
                "escalation_count": self.metrics.escalation_count,
                "last_activity": self.metrics.last_activity.isoformat()
            },
            "capabilities": {
                "can_plan_tests": self.capabilities.can_plan_tests,
                "can_execute_tests": self.capabilities.can_execute_tests,
                "can_diagnose_failures": self.capabilities.can_diagnose_failures,
                "can_learn_from_data": self.capabilities.can_learn_from_data,
                "supported_platforms": self.capabilities.supported_platforms,
                "supported_frameworks": self.capabilities.supported_frameworks
            }
        }
    
    def cleanup(self):
        """Cleanup agent resources."""
        self.executor.shutdown(wait=True)
        logger.info(f"Agent {self.name} cleaned up")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, state={self.state.value})>"
