"""Test orchestration using Kubernetes and Temporal for distributed test execution."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from temporalio import activity, workflow
from temporalio.client import Client
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from config.settings import get_settings
from src.test_generation.test_scenario import TestScenario

logger = logging.getLogger(__name__)


@dataclass
class TestExecution:
    """Represents a test execution instance."""
    execution_id: str
    scenario: TestScenario
    status: str  # pending, running, completed, failed, healed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    healing_attempts: int = 0
    max_healing_attempts: int = 3
    logs: List[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []


class TestOrchestrator:
    """Orchestrates test execution using Kubernetes and Temporal."""
    
    def __init__(self):
        self.settings = get_settings()
        self.temporal_client = None
        self.k8s_client = None
        self._initialize_temporal()
        self._initialize_kubernetes()
    
    async def _initialize_temporal(self):
        """Initialize Temporal client."""
        try:
            self.temporal_client = await Client.connect(self.settings.temporal_host)
            logger.info("Temporal client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Temporal client: {e}")
    
    def _initialize_kubernetes(self):
        """Initialize Kubernetes client."""
        try:
            config.load_kube_config(config_file=self.settings.kubeconfig_path)
            self.k8s_client = client.BatchV1Api()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
    
    async def execute_tests(self, scenarios: List[TestScenario]) -> List[TestExecution]:
        """Execute multiple test scenarios in parallel."""
        
        executions = []
        for scenario in scenarios:
            execution = TestExecution(
                execution_id=f"exec_{scenario.scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                scenario=scenario,
                status="pending"
            )
            executions.append(execution)
        
        # Execute tests using Temporal workflows
        if self.temporal_client:
            await self._execute_with_temporal(executions)
        else:
            await self._execute_with_kubernetes(executions)
        
        return executions
    
    async def _execute_with_temporal(self, executions: List[TestExecution]):
        """Execute tests using Temporal workflows."""
        
        for execution in executions:
            try:
                # Start workflow
                handle = await self.temporal_client.start_workflow(
                    TestExecutionWorkflow.run,
                    execution,
                    id=f"test-exec-{execution.execution_id}",
                    task_queue="test-execution"
                )
                
                # Update execution status
                execution.status = "running"
                execution.start_time = datetime.now()
                
                logger.info(f"Started Temporal workflow for execution: {execution.execution_id}")
                
            except Exception as e:
                logger.error(f"Error starting Temporal workflow: {e}")
                execution.status = "failed"
                execution.error_message = str(e)
    
    async def _execute_with_kubernetes(self, executions: List[TestExecution]):
        """Execute tests using Kubernetes Jobs."""
        
        for execution in executions:
            try:
                # Create Kubernetes Job for test execution
                job = self._create_test_job(execution)
                
                # Submit job to Kubernetes
                api_response = self.k8s_client.create_namespaced_job(
                    body=job,
                    namespace="default"
                )
                
                execution.status = "running"
                execution.start_time = datetime.now()
                
                logger.info(f"Created Kubernetes job for execution: {execution.execution_id}")
                
            except ApiException as e:
                logger.error(f"Error creating Kubernetes job: {e}")
                execution.status = "failed"
                execution.error_message = str(e)
    
    def _create_test_job(self, execution: TestExecution) -> Dict[str, Any]:
        """Create Kubernetes Job manifest for test execution."""
        
        job_name = f"test-exec-{execution.execution_id}"
        
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {
                    "test-execution-id": execution.execution_id,
                    "scenario-id": execution.scenario.scenario_id
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "test-runner",
                            "image": "test-automation-runner:latest",
                            "env": [
                                {"name": "TEST_SCENARIO", "value": json.dumps(execution.scenario.to_dict())},
                                {"name": "EXECUTION_ID", "value": execution.execution_id}
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"}
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": 3
            }
        }
        
        return job_manifest
    
    async def monitor_executions(self, executions: List[TestExecution]) -> List[TestExecution]:
        """Monitor test executions and handle failures."""
        
        while any(exec.status in ["running", "pending"] for exec in executions):
            for execution in executions:
                if execution.status == "running":
                    await self._check_execution_status(execution)
            
            # Wait before next check
            await asyncio.sleep(10)
        
        return executions
    
    async def _check_execution_status(self, execution: TestExecution):
        """Check the status of a test execution."""
        
        try:
            if self.temporal_client:
                await self._check_temporal_status(execution)
            else:
                await self._check_kubernetes_status(execution)
                
        except Exception as e:
            logger.error(f"Error checking execution status: {e}")
    
    async def _check_temporal_status(self, execution: TestExecution):
        """Check Temporal workflow status."""
        # Implementation would check Temporal workflow status
        # For now, simulate completion
        execution.status = "completed"
        execution.end_time = datetime.now()
    
    async def _check_kubernetes_status(self, execution: TestExecution):
        """Check Kubernetes job status."""
        
        job_name = f"test-exec-{execution.execution_id}"
        
        try:
            job = self.k8s_client.read_namespaced_job(
                name=job_name,
                namespace="default"
            )
            
            if job.status.succeeded:
                execution.status = "completed"
                execution.end_time = datetime.now()
            elif job.status.failed:
                execution.status = "failed"
                execution.end_time = datetime.now()
                execution.error_message = "Kubernetes job failed"
                
        except ApiException as e:
            if e.status == 404:
                # Job not found, might have completed and been cleaned up
                execution.status = "completed"
                execution.end_time = datetime.now()
            else:
                logger.error(f"Error checking Kubernetes job status: {e}")
    
    async def handle_test_failure(self, execution: TestExecution) -> bool:
        """Handle test failure with self-healing."""
        
        if execution.healing_attempts >= execution.max_healing_attempts:
            logger.warning(f"Max healing attempts reached for execution: {execution.execution_id}")
            return False
        
        execution.healing_attempts += 1
        
        try:
            # Attempt to heal the test using vision-based resolver
            healed = await self._attempt_healing(execution)
            
            if healed:
                execution.status = "healed"
                logger.info(f"Successfully healed execution: {execution.execution_id}")
                return True
            else:
                logger.warning(f"Healing failed for execution: {execution.execution_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error during healing attempt: {e}")
            return False
    
    async def _attempt_healing(self, execution: TestExecution) -> bool:
        """Attempt to heal a failed test."""
        # This would integrate with the VisionHealer
        # For now, return False to indicate no healing occurred
        return False


# Temporal Workflow Definitions

@workflow.defn
class TestExecutionWorkflow:
    """Temporal workflow for test execution."""
    
    @workflow.run
    async def run(self, execution: TestExecution) -> TestExecution:
        """Execute a test scenario workflow."""
        
        try:
            # Execute test activity
            result = await workflow.execute_activity(
                ExecuteTestActivity.run,
                execution,
                start_to_close_timeout=timedelta(minutes=30)
            )
            
            execution.status = "completed"
            execution.end_time = datetime.now()
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            # Attempt healing
            try:
                healed = await workflow.execute_activity(
                    HealingActivity.attempt_healing,
                    execution,
                    start_to_close_timeout=timedelta(minutes=10)
                )
                
                if healed:
                    execution.status = "healed"
                    
            except Exception as healing_error:
                logger.error(f"Healing failed: {healing_error}")
        
        return execution


@activity.defn
class ExecuteTestActivity:
    """Temporal activity for executing individual tests."""
    
    @staticmethod
    async def run(execution: TestExecution) -> bool:
        """Execute a test scenario."""
        
        try:
            # This would integrate with the actual test runner
            logger.info(f"Executing test: {execution.scenario.title}")
            
            # Simulate test execution
            await asyncio.sleep(5)
            
            # Simulate occasional failure
            import random
            if random.random() < 0.3:  # 30% failure rate for demo
                raise Exception("Simulated test failure")
            
            return True
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise


@activity.defn
class HealingActivity:
    """Temporal activity for test healing."""
    
    @staticmethod
    async def run(execution: TestExecution) -> bool:
        """Attempt to heal a failed test."""
        
        try:
            # This would integrate with the VisionHealer
            logger.info(f"Attempting to heal test: {execution.scenario.title}")
            
            # Simulate healing attempt
            await asyncio.sleep(2)
            
            # Simulate healing success
            import random
            return random.random() < 0.7  # 70% healing success rate
            
        except Exception as e:
            logger.error(f"Healing attempt failed: {e}")
            return False
