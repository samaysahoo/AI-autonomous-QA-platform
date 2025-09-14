"""Agent Registry - manages agent discovery and registration."""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseAgent, AgentCapabilities

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent registration status."""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNREGISTERED = "unregistered"


@dataclass
class AgentRegistration:
    """Agent registration information."""
    agent_id: str
    agent_class: str
    capabilities: AgentCapabilities
    status: AgentStatus = AgentStatus.REGISTERED
    registration_time: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[BaseAgent] = None


class AgentRegistry:
    """Registry for managing agent discovery and registration."""
    
    def __init__(self):
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.agent_capabilities_index: Dict[str, Set[str]] = {}
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 90  # seconds
        
        # Start heartbeat monitoring
        asyncio.create_task(self._heartbeat_monitoring_loop())
        
        logger.info("Agent Registry initialized")
    
    def register_agent(self, agent: BaseAgent, metadata: Dict[str, Any] = None) -> bool:
        """Register an agent in the registry."""
        try:
            registration = AgentRegistration(
                agent_id=agent.agent_id,
                agent_class=agent.__class__.__name__,
                capabilities=agent.capabilities,
                metadata=metadata or {},
                instance=agent
            )
            
            self.registered_agents[agent.agent_id] = registration
            
            # Update capabilities index
            self._update_capabilities_index(agent)
            
            logger.info(f"Registered agent {agent.agent_id} ({agent.__class__.__name__})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry."""
        if agent_id in self.registered_agents:
            registration = self.registered_agents[agent_id]
            
            # Update capabilities index
            self._remove_from_capabilities_index(registration)
            
            # Mark as unregistered
            registration.status = AgentStatus.UNREGISTERED
            
            del self.registered_agents[agent_id]
            
            logger.info(f"Unregistered agent {agent_id}")
            return True
        
        return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent instance by ID."""
        registration = self.registered_agents.get(agent_id)
        if registration and registration.status == AgentStatus.ACTIVE:
            return registration.instance
        return None
    
    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents that have a specific capability."""
        agent_ids = self.agent_capabilities_index.get(capability, set())
        
        agents = []
        for agent_id in agent_ids:
            agent = self.get_agent(agent_id)
            if agent:
                agents.append(agent)
        
        return agents
    
    def get_agents_by_platform(self, platform: str) -> List[BaseAgent]:
        """Get all agents that support a specific platform."""
        agents = []
        
        for registration in self.registered_agents.values():
            if (registration.status == AgentStatus.ACTIVE and 
                registration.capabilities.supported_platforms and 
                platform in registration.capabilities.supported_platforms):
                if registration.instance:
                    agents.append(registration.instance)
        
        return agents
    
    def get_agents_by_framework(self, framework: str) -> List[BaseAgent]:
        """Get all agents that support a specific framework."""
        agents = []
        
        for registration in self.registered_agents.values():
            if (registration.status == AgentStatus.ACTIVE and 
                registration.capabilities.supported_frameworks and 
                framework in registration.capabilities.supported_frameworks):
                if registration.instance:
                    agents.append(registration.instance)
        
        return agents
    
    def find_best_agent_for_task(self, task_type: str, task_data: Dict[str, Any]) -> Optional[BaseAgent]:
        """Find the best agent for a specific task."""
        suitable_agents = []
        
        for registration in self.registered_agents.values():
            if (registration.status == AgentStatus.ACTIVE and 
                registration.instance and 
                registration.instance.can_handle_task(task_type, task_data)):
                suitable_agents.append(registration.instance)
        
        if not suitable_agents:
            return None
        
        # Select best agent based on availability and performance
        return self._select_best_agent(suitable_agents)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status."""
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].status = status
            logger.debug(f"Updated agent {agent_id} status to {status.value}")
    
    def record_heartbeat(self, agent_id: str) -> bool:
        """Record agent heartbeat."""
        if agent_id in self.registered_agents:
            registration = self.registered_agents[agent_id]
            registration.last_heartbeat = datetime.now()
            
            # Mark as active if it was inactive
            if registration.status == AgentStatus.INACTIVE:
                registration.status = AgentStatus.ACTIVE
            
            return True
        
        return False
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status information."""
        total_agents = len(self.registered_agents)
        active_agents = len([r for r in self.registered_agents.values() if r.status == AgentStatus.ACTIVE])
        inactive_agents = len([r for r in self.registered_agents.values() if r.status == AgentStatus.INACTIVE])
        error_agents = len([r for r in self.registered_agents.values() if r.status == AgentStatus.ERROR])
        
        # Get capability distribution
        capability_distribution = {}
        for capability, agent_ids in self.agent_capabilities_index.items():
            capability_distribution[capability] = len(agent_ids)
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": inactive_agents,
            "error_agents": error_agents,
            "capability_distribution": capability_distribution,
            "registry_uptime": datetime.now().isoformat()
        }
    
    def get_agent_list(self) -> List[Dict[str, Any]]:
        """Get list of all registered agents."""
        agent_list = []
        
        for agent_id, registration in self.registered_agents.items():
            agent_info = {
                "agent_id": agent_id,
                "agent_class": registration.agent_class,
                "status": registration.status.value,
                "capabilities": {
                    "can_plan_tests": registration.capabilities.can_plan_tests,
                    "can_execute_tests": registration.capabilities.can_execute_tests,
                    "can_diagnose_failures": registration.capabilities.can_diagnose_failures,
                    "can_learn_from_data": registration.capabilities.can_learn_from_data,
                    "can_coordinate_agents": registration.capabilities.can_coordinate_agents,
                    "supported_platforms": registration.capabilities.supported_platforms,
                    "supported_frameworks": registration.capabilities.supported_frameworks,
                    "max_concurrent_tasks": registration.capabilities.max_concurrent_tasks
                },
                "registration_time": registration.registration_time.isoformat(),
                "last_heartbeat": registration.last_heartbeat.isoformat(),
                "metadata": registration.metadata
            }
            
            agent_list.append(agent_info)
        
        return agent_list
    
    def _update_capabilities_index(self, agent: BaseAgent):
        """Update the capabilities index for an agent."""
        capabilities = agent.capabilities
        
        # Index by capability
        if capabilities.can_plan_tests:
            self._add_to_capability_index("plan_tests", agent.agent_id)
        
        if capabilities.can_execute_tests:
            self._add_to_capability_index("execute_tests", agent.agent_id)
        
        if capabilities.can_diagnose_failures:
            self._add_to_capability_index("diagnose_failures", agent.agent_id)
        
        if capabilities.can_learn_from_data:
            self._add_to_capability_index("learn_from_data", agent.agent_id)
        
        if capabilities.can_coordinate_agents:
            self._add_to_capability_index("coordinate_agents", agent.agent_id)
        
        # Index by platform
        for platform in capabilities.supported_platforms:
            self._add_to_capability_index(f"platform_{platform}", agent.agent_id)
        
        # Index by framework
        for framework in capabilities.supported_frameworks:
            self._add_to_capability_index(f"framework_{framework}", agent.agent_id)
    
    def _remove_from_capabilities_index(self, registration: AgentRegistration):
        """Remove agent from capabilities index."""
        agent_id = registration.agent_id
        capabilities = registration.capabilities
        
        # Remove from capability indices
        if capabilities.can_plan_tests:
            self._remove_from_capability_index("plan_tests", agent_id)
        
        if capabilities.can_execute_tests:
            self._remove_from_capability_index("execute_tests", agent_id)
        
        if capabilities.can_diagnose_failures:
            self._remove_from_capability_index("diagnose_failures", agent_id)
        
        if capabilities.can_learn_from_data:
            self._remove_from_capability_index("learn_from_data", agent_id)
        
        if capabilities.can_coordinate_agents:
            self._remove_from_capability_index("coordinate_agents", agent_id)
        
        # Remove from platform indices
        for platform in capabilities.supported_platforms:
            self._remove_from_capability_index(f"platform_{platform}", agent_id)
        
        # Remove from framework indices
        for framework in capabilities.supported_frameworks:
            self._remove_from_capability_index(f"framework_{framework}", agent_id)
    
    def _add_to_capability_index(self, capability: str, agent_id: str):
        """Add agent to capability index."""
        if capability not in self.agent_capabilities_index:
            self.agent_capabilities_index[capability] = set()
        
        self.agent_capabilities_index[capability].add(agent_id)
    
    def _remove_from_capability_index(self, capability: str, agent_id: str):
        """Remove agent from capability index."""
        if capability in self.agent_capabilities_index:
            self.agent_capabilities_index[capability].discard(agent_id)
            
            # Clean up empty capability entries
            if not self.agent_capabilities_index[capability]:
                del self.agent_capabilities_index[capability]
    
    def _select_best_agent(self, agents: List[BaseAgent]) -> BaseAgent:
        """Select the best agent from a list of suitable agents."""
        # Simple selection based on availability and metrics
        # In practice, this could be more sophisticated
        
        available_agents = [a for a in agents if a.state.value == "idle"]
        
        if available_agents:
            # Select agent with best success rate
            return max(available_agents, key=lambda a: a.metrics.success_rate)
        else:
            # Select agent with fewest active tasks
            return min(agents, key=lambda a: len(a.active_tasks))
    
    async def _heartbeat_monitoring_loop(self):
        """Monitor agent heartbeats and mark inactive agents."""
        while True:
            try:
                current_time = datetime.now()
                
                for agent_id, registration in self.registered_agents.items():
                    if registration.status == AgentStatus.ACTIVE:
                        time_since_heartbeat = (current_time - registration.last_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > self.heartbeat_timeout:
                            logger.warning(f"Agent {agent_id} missed heartbeat, marking as inactive")
                            registration.status = AgentStatus.INACTIVE
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    def cleanup(self):
        """Cleanup registry resources."""
        logger.info("Cleaning up Agent Registry")
        
        # Mark all agents as unregistered
        for registration in self.registered_agents.values():
            registration.status = AgentStatus.UNREGISTERED
        
        # Clear registrations
        self.registered_agents.clear()
        self.agent_capabilities_index.clear()
        
        logger.info("Agent Registry cleaned up")
