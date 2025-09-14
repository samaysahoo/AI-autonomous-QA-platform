"""Demo script for the LangGraph-based multi-agent system."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the LangGraph multi-agent system components
from src.langgraph_agents.workflow_graph import TestAutomationWorkflow
from src.langgraph_agents.state import create_initial_state

logger.info("Starting LangGraph Multi-Agent System Demo")


class LangGraphSystemDemo:
    """Demo class for the LangGraph-based multi-agent system."""
    
    def __init__(self):
        self.workflow = None
        
    async def setup_system(self):
        """Setup the LangGraph workflow system."""
        logger.info("Setting up LangGraph multi-agent system...")
        
        try:
            # Initialize workflow
            self.workflow = TestAutomationWorkflow()
            
            logger.info("LangGraph workflow system setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup LangGraph system: {e}")
            raise
    
    async def demo_e2e_test_workflow(self):
        """Demo end-to-end test workflow."""
        logger.info("=== Demo: End-to-End Test Workflow ===")
        
        # Prepare input data for feature implementation
        input_data = {
            "code_changes": {
                "change_type": "feature_implementation",
                "diff_content": """
                +def new_authentication_feature():
                +    # New OAuth2 authentication implementation
                +    return authenticate_with_oauth2()
                +    
                +def validate_user_permissions(user_id):
                +    # Validate user permissions for new feature
                +    return check_user_access(user_id)
                """,
                "changed_files": ["auth.py", "user_service.py", "permissions.py"],
                "commit_metadata": {
                    "message": "Add OAuth2 authentication and user permission validation",
                    "author": "developer@example.com",
                    "timestamp": datetime.now().isoformat()
                }
            },
            "requirements": [
                "OAuth2 authentication must be secure",
                "User permissions must be validated",
                "Feature must be testable across platforms",
                "Performance impact should be minimal"
            ],
            "config": {
                "max_parallel_tasks": 3,
                "default_timeout": 300,
                "escalation_threshold": "medium",
                "learning_mode": "incremental"
            }
        }
        
        try:
            # Execute the workflow
            result = await self.workflow.execute_workflow(
                workflow_id="e2e-test-workflow",
                input_data=input_data
            )
            
            logger.info(f"E2E Test Workflow completed with status: {result['status']}")
            logger.info(f"Execution ID: {result['execution_id']}")
            
            # Display results
            summary = result.get("summary", {})
            logger.info(f"Test Scenarios Generated: {summary.get('total_scenarios', 0)}")
            logger.info(f"Test Results: {summary.get('total_results', 0)}")
            logger.info(f"Success Rate: {summary.get('success_rate', 0):.2%}")
            logger.info(f"Failure Clusters: {summary.get('failure_clusters', 0)}")
            logger.info(f"Tasks Completed: {summary.get('tasks_completed', 0)}")
            logger.info(f"Tasks Failed: {summary.get('tasks_failed', 0)}")
            logger.info(f"Execution Time: {summary.get('execution_time', 0):.2f} seconds")
            
            # Display messages
            messages = result.get("messages", [])
            logger.info(f"Workflow Messages ({len(messages)}):")
            for i, message in enumerate(messages[-5:], 1):  # Show last 5 messages
                logger.info(f"  {i}. {message}")
            
            # Display errors if any
            errors = result.get("errors", [])
            if errors:
                logger.warning(f"Errors encountered ({len(errors)}):")
                for i, error in enumerate(errors, 1):
                    logger.warning(f"  {i}. {error}")
            
            # Display escalation status
            if result.get("escalation_needed", False):
                logger.warning("⚠️  Escalation was needed during execution")
            else:
                logger.info("✅ No escalation needed")
            
            return result
            
        except Exception as e:
            logger.error(f"E2E Test Workflow failed: {e}")
            return None
    
    async def demo_bug_triage_workflow(self):
        """Demo bug triage workflow."""
        logger.info("=== Demo: Bug Triage Workflow ===")
        
        # Prepare input data for bug triage
        input_data = {
            "test_results": [
                {
                    "scenario_id": "test-001",
                    "status": "failed",
                    "error_message": "Element not found: login_button",
                    "execution_time": 45.2,
                    "screenshots": ["screenshot1.png"]
                },
                {
                    "scenario_id": "test-002", 
                    "status": "failed",
                    "error_message": "Element not found: login_button",
                    "execution_time": 43.8,
                    "screenshots": ["screenshot2.png"]
                },
                {
                    "scenario_id": "test-003",
                    "status": "failed", 
                    "error_message": "Timeout waiting for element: submit_form",
                    "execution_time": 120.0,
                    "screenshots": ["screenshot3.png"]
                },
                {
                    "scenario_id": "test-004",
                    "status": "passed",
                    "execution_time": 12.5
                }
            ],
            "clustering_method": "auto",
            "min_cluster_size": 2
        }
        
        try:
            # Execute the workflow
            result = await self.workflow.execute_workflow(
                workflow_id="bug-triage-workflow",
                input_data=input_data
            )
            
            logger.info(f"Bug Triage Workflow completed with status: {result['status']}")
            
            # Display results
            summary = result.get("summary", {})
            logger.info(f"Failure Clusters Created: {summary.get('failure_clusters', 0)}")
            logger.info(f"Bugs Triaged: {summary.get('bugs_triaged', 0)}")
            logger.info(f"Root Causes Identified: {summary.get('root_causes_identified', 0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Bug Triage Workflow failed: {e}")
            return None
    
    async def demo_performance_optimization_workflow(self):
        """Demo performance optimization workflow."""
        logger.info("=== Demo: Performance Optimization Workflow ===")
        
        # Prepare input data for performance optimization
        input_data = {
            "performance_data": {
                "current_metrics": {
                    "success_rate": 0.85,
                    "average_execution_time": 65.2,
                    "resource_utilization": 0.78
                },
                "optimization_targets": ["execution_time", "success_rate"],
                "historical_data": {
                    "previous_success_rate": 0.82,
                    "previous_execution_time": 72.1
                }
            }
        }
        
        try:
            # Execute the workflow
            result = await self.workflow.execute_workflow(
                workflow_id="performance-optimization-workflow",
                input_data=input_data
            )
            
            logger.info(f"Performance Optimization Workflow completed with status: {result['status']}")
            
            # Display results
            summary = result.get("summary", {})
            logger.info(f"Optimizations Applied: {summary.get('optimizations_applied', 0)}")
            logger.info(f"Expected Improvement: {summary.get('expected_improvement', 0):.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance Optimization Workflow failed: {e}")
            return None
    
    async def demo_agent_capabilities(self):
        """Demo individual agent capabilities."""
        logger.info("=== Demo: Agent Capabilities ===")
        
        # Display agent status and capabilities
        for agent_id, agent in self.workflow.agents.items():
            try:
                status = await agent.get_agent_status()
                
                logger.info(f"Agent: {status['name']} ({agent_id})")
                logger.info(f"  Status: {status['status']}")
                logger.info(f"  Capabilities: {list(status.get('capabilities', {}).keys())}")
                
                if 'metrics' in status:
                    metrics = status['metrics']
                    logger.info(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
                    logger.info(f"  Last Activity: {metrics.get('last_activity', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error getting status for agent {agent_id}: {e}")
    
    async def demo_workflow_list(self):
        """Demo available workflows."""
        logger.info("=== Demo: Available Workflows ===")
        
        try:
            workflows = self.workflow.get_available_workflows()
            
            logger.info(f"Available Workflows ({len(workflows)}):")
            for workflow in workflows:
                logger.info(f"  - {workflow['name']} ({workflow['workflow_id']})")
                logger.info(f"    Description: {workflow['description']}")
                logger.info(f"    Steps: {', '.join(workflow['steps'])}")
                logger.info(f"    Duration: {workflow['estimated_duration']}")
                logger.info("")
                
        except Exception as e:
            logger.error(f"Error getting workflows: {e}")
    
    async def demo_state_management(self):
        """Demo state management capabilities."""
        logger.info("=== Demo: State Management ===")
        
        try:
            # Create initial state
            initial_state = create_initial_state(
                workflow_id="demo-workflow",
                config={"demo_mode": True}
            )
            
            logger.info("Initial State Created:")
            logger.info(f"  Workflow ID: {initial_state['workflow']['workflow_id']}")
            logger.info(f"  Execution ID: {initial_state['workflow']['execution_id']}")
            logger.info(f"  System Status: {initial_state['system_status']}")
            logger.info(f"  Created At: {initial_state['created_at']}")
            logger.info(f"  Config: {initial_state['config']}")
            
            # Demonstrate state updates
            initial_state["code_changes"] = {
                "change_type": "demo_change",
                "diff_content": "+def demo(): pass"
            }
            initial_state["requirements"] = ["Demo requirement"]
            
            logger.info("State Updated:")
            logger.info(f"  Code Changes: {initial_state['code_changes']['change_type']}")
            logger.info(f"  Requirements: {len(initial_state['requirements'])}")
            
        except Exception as e:
            logger.error(f"Error in state management demo: {e}")
    
    async def run_complete_demo(self):
        """Run the complete LangGraph system demo."""
        logger.info("Starting LangGraph Multi-Agent System Demo")
        
        try:
            # Setup system
            await self.setup_system()
            
            # Demo 1: Available Workflows
            await self.demo_workflow_list()
            
            # Demo 2: Agent Capabilities
            await self.demo_agent_capabilities()
            
            # Demo 3: State Management
            await self.demo_state_management()
            
            # Demo 4: E2E Test Workflow
            e2e_result = await self.demo_e2e_test_workflow()
            
            # Demo 5: Bug Triage Workflow
            bug_triage_result = await self.demo_bug_triage_workflow()
            
            # Demo 6: Performance Optimization Workflow
            perf_result = await self.demo_performance_optimization_workflow()
            
            logger.info("LangGraph Multi-Agent System Demo completed successfully!")
            
            # Display final summary
            logger.info("=== Final Demo Summary ===")
            if e2e_result:
                logger.info(f"✅ E2E Test Workflow: {e2e_result['status']}")
            if bug_triage_result:
                logger.info(f"✅ Bug Triage Workflow: {bug_triage_result['status']}")
            if perf_result:
                logger.info(f"✅ Performance Optimization Workflow: {perf_result['status']}")
            
            logger.info("🎉 All demos completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        
        finally:
            logger.info("Demo cleanup completed")


async def main():
    """Main function to run the demo."""
    demo = LangGraphSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
