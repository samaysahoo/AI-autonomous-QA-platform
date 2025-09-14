"""Demo script for the multi-agent system."""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the multi-agent system components
from src.agents.agent_coordinator import AgentCoordinator, WorkflowDefinition, CoordinationStrategy, EscalationLevel
from src.agents.agent_registry import AgentRegistry
from src.agents.test_planner_agent import TestPlannerAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.learning_agent import LearningAgent
from src.agents.base_agent import AgentMessage, MessageType
from src.test_generation.test_scenario import TestScenario, TestType, TestFramework, TestPriority


class MultiAgentSystemDemo:
    """Demo class for the multi-agent system."""
    
    def __init__(self):
        self.coordinator = None
        self.registry = None
        self.agents = {}
        
    async def setup_system(self):
        """Setup the multi-agent system."""
        logger.info("Setting up multi-agent system...")
        
        # Initialize registry
        self.registry = AgentRegistry()
        
        # Initialize coordinator
        self.coordinator = AgentCoordinator()
        
        # Create and register agents
        self.agents = {
            "test-planner": TestPlannerAgent("test-planner-001"),
            "execution": ExecutionAgent("execution-001"),
            "diagnosis": DiagnosisAgent("diagnosis-001"),
            "learning": LearningAgent("learning-001")
        }
        
        # Register agents
        for agent_id, agent in self.agents.items():
            self.registry.register_agent(agent)
            self.coordinator.agents[agent_id] = agent
        
        # Start coordinator
        await self.coordinator.start_coordination()
        
        logger.info("Multi-agent system setup completed")
    
    async def demo_test_planning(self):
        """Demo test planning workflow."""
        logger.info("=== Demo: Test Planning ===")
        
        # Create a test planning task
        task_data = {
            "task_type": "plan_tests_for_change",
            "task_data": {
                "change_type": "feature_implementation",
                "diff_content": """
                +def new_feature():
                +    # New authentication feature
                +    return authenticate_user()
                """,
                "changed_files": ["auth.py", "user_service.py"],
                "commit_metadata": {
                    "message": "Add new authentication feature",
                    "author": "developer@example.com",
                    "timestamp": datetime.now().isoformat()
                },
                "requirements": [
                    "User authentication must be secure",
                    "Feature must be testable",
                    "Performance impact should be minimal"
                ]
            }
        }
        
        # Execute task
        planner = self.agents["test-planner"]
        message = AgentMessage(
            sender_id="demo",
            receiver_id="test-planner-001",
            message_type=MessageType.TASK_REQUEST,
            payload=task_data
        )
        
        response = await planner.receive_message(message)
        
        if response and response.message_type == MessageType.TASK_RESPONSE:
            result = response.payload.get("result", {})
            logger.info(f"Test planning completed: {len(result.get('scenarios', []))} scenarios generated")
            
            # Display some results
            scenarios = result.get("scenarios", [])
            if scenarios:
                logger.info(f"Sample scenario: {scenarios[0].get('title', 'No title')}")
                logger.info(f"Risk level: {result.get('risk_analysis', {}).get('risk_level', 'unknown')}")
            
            return result
        else:
            logger.error("Test planning failed")
            return None
    
    async def demo_test_execution(self, scenarios_data: Dict[str, Any]):
        """Demo test execution workflow."""
        logger.info("=== Demo: Test Execution ===")
        
        if not scenarios_data:
            logger.warning("No scenarios to execute")
            return None
        
        # Create test execution task
        task_data = {
            "task_type": "execute_test_scenarios",
            "task_data": {
                "scenarios": scenarios_data.get("scenarios", []),
                "strategy": "adaptive",
                "max_parallel": 2,
                "timeout": 600
            }
        }
        
        # Execute task
        executor = self.agents["execution"]
        message = AgentMessage(
            sender_id="demo",
            receiver_id="execution-001",
            message_type=MessageType.TASK_REQUEST,
            payload=task_data
        )
        
        response = await executor.receive_message(message)
        
        if response and response.message_type == MessageType.TASK_RESPONSE:
            result = response.payload.get("result", {})
            logger.info(f"Test execution completed: {result.get('scenarios_executed', 0)} scenarios executed")
            logger.info(f"Success rate: {result.get('summary', {}).get('success_rate', 0):.2%}")
            
            return result
        else:
            logger.error("Test execution failed")
            return None
    
    async def demo_failure_diagnosis(self, execution_results: Dict[str, Any]):
        """Demo failure diagnosis workflow."""
        logger.info("=== Demo: Failure Diagnosis ===")
        
        if not execution_results:
            logger.warning("No execution results to analyze")
            return None
        
        # Create diagnosis task
        task_data = {
            "task_type": "cluster_test_failures",
            "task_data": {
                "test_results": execution_results.get("results", []),
                "clustering_method": "auto",
                "min_cluster_size": 2
            }
        }
        
        # Execute task
        diagnoser = self.agents["diagnosis"]
        message = AgentMessage(
            sender_id="demo",
            receiver_id="diagnosis-001",
            message_type=MessageType.TASK_REQUEST,
            payload=task_data
        )
        
        response = await diagnoser.receive_message(message)
        
        if response and response.message_type == MessageType.TASK_RESPONSE:
            result = response.payload.get("result", {})
            clusters = result.get("clusters", [])
            logger.info(f"Failure analysis completed: {len(clusters)} clusters identified")
            
            if clusters:
                logger.info(f"Largest cluster size: {result.get('cluster_analysis', {}).get('largest_cluster_size', 0)}")
            
            return result
        else:
            logger.error("Failure diagnosis failed")
            return None
    
    async def demo_learning_update(self, diagnosis_results: Dict[str, Any]):
        """Demo learning update workflow."""
        logger.info("=== Demo: Learning Update ===")
        
        if not diagnosis_results:
            logger.warning("No diagnosis results to learn from")
            return None
        
        # Create learning task
        task_data = {
            "task_type": "learn_from_feedback",
            "task_data": {
                "feedback": [
                    {
                        "feedback_id": "demo-feedback-001",
                        "test_result_id": "demo-result-001",
                        "feedback_type": "test_correction",
                        "feedback_text": "Test locator needs to be more robust",
                        "rating": 4,
                        "corrections": ["Update element locator", "Add wait conditions"],
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "learning_mode": "incremental"
            }
        }
        
        # Execute task
        learner = self.agents["learning"]
        message = AgentMessage(
            sender_id="demo",
            receiver_id="learning-001",
            message_type=MessageType.TASK_REQUEST,
            payload=task_data
        )
        
        response = await learner.receive_message(message)
        
        if response and response.message_type == MessageType.TASK_RESPONSE:
            result = response.payload.get("result", {})
            logger.info(f"Learning update completed: {result.get('total_feedback_processed', 0)} feedback items processed")
            
            model_updates = result.get("model_updates", [])
            if model_updates:
                logger.info(f"Model updates: {len(model_updates)} models updated")
            
            return result
        else:
            logger.error("Learning update failed")
            return None
    
    async def demo_workflow_execution(self):
        """Demo complete workflow execution."""
        logger.info("=== Demo: Complete Workflow Execution ===")
        
        # Start end-to-end test workflow
        workflow_id = "e2e-test-workflow"
        input_data = {
            "change_type": "bug_fix",
            "diff_content": """
            -def buggy_function():
            -    return None  # Bug: should return value
            +def buggy_function():
            +    return "fixed"  # Fixed: returns proper value
            """,
            "changed_files": ["buggy_module.py"],
            "commit_metadata": {
                "message": "Fix null return bug",
                "author": "developer@example.com"
            },
            "requirements": ["Function must return proper value"]
        }
        
        try:
            execution_id = await self.coordinator.start_workflow(workflow_id, input_data)
            logger.info(f"Started workflow {workflow_id} with execution ID: {execution_id}")
            
            # Wait for workflow completion
            max_wait_time = 300  # 5 minutes
            wait_interval = 10   # 10 seconds
            
            for _ in range(max_wait_time // wait_interval):
                if execution_id in self.coordinator.active_workflows:
                    execution = self.coordinator.active_workflows[execution_id]
                    logger.info(f"Workflow status: {execution.status}, step: {execution.current_step}")
                    
                    if execution.status in ["completed", "failed", "error"]:
                        logger.info(f"Workflow {execution_id} finished with status: {execution.status}")
                        if execution.errors:
                            logger.warning(f"Workflow errors: {execution.errors}")
                        break
                    
                    await asyncio.sleep(wait_interval)
                else:
                    logger.error(f"Workflow {execution_id} not found")
                    break
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return None
    
    async def demo_agent_coordination(self):
        """Demo agent coordination and communication."""
        logger.info("=== Demo: Agent Coordination ===")
        
        # Create a coordination task that requires multiple agents
        coordination_task = {
            "task_type": "coordination_request",
            "task_data": {
                "request_type": "get_status",
                "include_metrics": True
            }
        }
        
        # Send coordination request
        message = AgentMessage(
            sender_id="demo",
            receiver_id="coordinator-001",
            message_type=MessageType.COORDINATION_REQUEST,
            payload=coordination_task
        )
        
        response = await self.coordinator._handle_coordination_request(message)
        
        if response:
            logger.info("Coordination request processed successfully")
        else:
            logger.info("Coordination request completed")
        
        # Display system status
        status = self.coordinator.get_coordinator_status()
        logger.info(f"System status: {status['active_agents']} agents, {status['active_workflows']} workflows")
        
        # Display agent statuses
        for agent_id, agent_status in status["agent_status"].items():
            logger.info(f"Agent {agent_id}: {agent_status['state']} ({agent_status['active_tasks']} active tasks)")
    
    async def demo_escalation_scenario(self):
        """Demo escalation scenario."""
        logger.info("=== Demo: Escalation Scenario ===")
        
        # Simulate a critical error that should trigger escalation
        error_data = {
            "error_type": "critical_system_failure",
            "severity": "critical",
            "description": "Test execution system completely down",
            "affected_components": ["execution_engine", "test_orchestrator"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Send error report
        message = AgentMessage(
            sender_id="execution-001",
            receiver_id="coordinator-001",
            message_type=MessageType.ESCALATION_REQUEST,
            payload={
                "escalation_level": "critical",
                "error": "System failure detected",
                "agent_id": "execution-001",
                "agent_name": "Execution Agent",
                "original_message": error_data
            },
            priority=10  # Highest priority
        )
        
        # Process escalation
        await self.coordinator._handle_escalation_request(message)
        
        logger.info("Escalation scenario completed")
    
    async def run_complete_demo(self):
        """Run the complete multi-agent system demo."""
        logger.info("Starting Multi-Agent System Demo")
        
        try:
            # Setup system
            await self.setup_system()
            
            # Demo 1: Test Planning
            planning_results = await self.demo_test_planning()
            
            # Demo 2: Test Execution
            execution_results = await self.demo_test_execution(planning_results)
            
            # Demo 3: Failure Diagnosis
            diagnosis_results = await self.demo_failure_diagnosis(execution_results)
            
            # Demo 4: Learning Update
            learning_results = await self.demo_learning_update(diagnosis_results)
            
            # Demo 5: Workflow Execution
            workflow_id = await self.demo_workflow_execution()
            
            # Demo 6: Agent Coordination
            await self.demo_agent_coordination()
            
            # Demo 7: Escalation Scenario
            await self.demo_escalation_scenario()
            
            logger.info("Multi-Agent System Demo completed successfully!")
            
            # Display final system status
            final_status = self.coordinator.get_coordinator_status()
            logger.info(f"Final system status: {json.dumps(final_status, indent=2, default=str)}")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.coordinator:
                await self.coordinator.stop_coordinator()
            if self.registry:
                self.registry.cleanup()


async def main():
    """Main function to run the demo."""
    demo = MultiAgentSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
