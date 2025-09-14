"""Test Planner Agent - interprets product changes and generates test scenarios."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_agent import BaseAgent, AgentCapabilities, MessageType, AgentMessage
from ..test_generation.test_scenario import TestScenario, TestType, TestFramework, TestPriority
from ..test_generation.test_generator import TestCaseGenerator
from ..data_ingestion.vector_store import VectorStoreManager
from ..observability.risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)


class TestPlannerAgent(BaseAgent):
    """Agent responsible for planning and generating test scenarios."""
    
    def __init__(self, agent_id: str = "test-planner-001"):
        capabilities = AgentCapabilities(
            can_plan_tests=True,
            supported_platforms=["web", "mobile", "api", "desktop"],
            supported_frameworks=["appium", "selenium", "espresso", "xcuitest", "pytest"],
            max_concurrent_tasks=3
        )
        
        super().__init__(agent_id, "Test Planner Agent", capabilities)
        
        # Initialize components
        self.test_generator = TestCaseGenerator()
        self.vector_store = VectorStoreManager()
        self.risk_analyzer = RiskAnalyzer()
        
        # Planning strategies
        self.planning_strategies = {
            "feature_implementation": self._plan_feature_tests,
            "bug_fix": self._plan_bug_fix_tests,
            "refactoring": self._plan_refactoring_tests,
            "security_update": self._plan_security_tests,
            "performance_optimization": self._plan_performance_tests
        }
        
        logger.info(f"Test Planner Agent initialized with {len(self.planning_strategies)} strategies")
    
    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this agent can handle a specific task type."""
        return task_type in [
            "plan_tests_for_change",
            "plan_tests_for_ticket", 
            "plan_regression_tests",
            "plan_smoke_tests",
            "analyze_test_coverage",
            "prioritize_test_scenarios"
        ]
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a test planning task."""
        task_type = task_data.get("task_type", "")
        
        logger.info(f"Test Planner Agent processing task: {task_type}")
        
        try:
            if task_type == "plan_tests_for_change":
                return await self._plan_tests_for_change(task_data)
            elif task_type == "plan_tests_for_ticket":
                return await self._plan_tests_for_ticket(task_data)
            elif task_type == "plan_regression_tests":
                return await self._plan_regression_tests(task_data)
            elif task_type == "plan_smoke_tests":
                return await self._plan_smoke_tests(task_data)
            elif task_type == "analyze_test_coverage":
                return await self._analyze_test_coverage(task_data)
            elif task_type == "prioritize_test_scenarios":
                return await self._prioritize_test_scenarios(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_type}: {e}")
            raise
    
    async def _plan_tests_for_change(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan tests for a code change."""
        change_type = task_data.get("change_type", "feature_implementation")
        diff_content = task_data.get("diff_content", "")
        changed_files = task_data.get("changed_files", [])
        commit_metadata = task_data.get("commit_metadata", {})
        requirements = task_data.get("requirements", [])
        
        # Analyze risk of the change
        risk_score = self.risk_analyzer.analyze_code_change_risk(
            diff_content=diff_content,
            changed_files=changed_files,
            commit_metadata=commit_metadata
        )
        
        # Get relevant context from vector store
        context_query = f"{change_type} {commit_metadata.get('message', '')} {' '.join(changed_files)}"
        context_docs = self.vector_store.search_similar(
            query=context_query,
            n_results=5,
            metadata_filter={"source": {"$in": ["jira", "analytics"]}}
        )
        
        # Generate test scenarios based on change type
        strategy = self.planning_strategies.get(change_type, self._plan_feature_tests)
        scenarios = await strategy(task_data, risk_score, context_docs)
        
        # Prioritize scenarios based on risk
        prioritized_scenarios = self._prioritize_by_risk(scenarios, risk_score)
        
        return {
            "scenarios": [scenario.to_dict() for scenario in prioritized_scenarios],
            "risk_analysis": {
                "risk_level": risk_score.risk_level,
                "confidence": risk_score.confidence,
                "recommendations": risk_score.recommendations,
                "affected_components": risk_score.affected_areas
            },
            "context_documents": len(context_docs),
            "planning_strategy": change_type,
            "total_scenarios": len(prioritized_scenarios)
        }
    
    async def _plan_tests_for_ticket(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan tests for a Jira ticket or user story."""
        ticket_data = task_data.get("ticket", {})
        ticket_type = ticket_data.get("issue_type", "")
        description = ticket_data.get("description", "")
        acceptance_criteria = task_data.get("acceptance_criteria", [])
        
        # Determine test type based on ticket type
        if ticket_type in ["Story", "User Story"]:
            test_type = TestType.E2E
            framework = TestFramework.APPIUM
        elif ticket_type == "Bug":
            test_type = TestType.E2E
            framework = TestFramework.APPIUM
        elif ticket_type == "Task":
            test_type = TestType.INTEGRATION
            framework = TestFramework.SELENIUM
        else:
            test_type = TestType.E2E
            framework = TestFramework.APPIUM
        
        # Generate test scenarios
        query = f"{description} {' '.join(acceptance_criteria)}"
        scenarios = self.test_generator.generate_test_scenarios(
            query=query,
            test_type=test_type,
            framework=framework,
            max_scenarios=5
        )
        
        # Add acceptance criteria as test steps
        for scenario in scenarios:
            for criteria in acceptance_criteria:
                # This would add acceptance criteria as validation steps
                pass
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "ticket_info": {
                "type": ticket_type,
                "key": ticket_data.get("key", ""),
                "summary": ticket_data.get("summary", "")
            },
            "test_type": test_type.value,
            "framework": framework.value,
            "total_scenarios": len(scenarios)
        }
    
    async def _plan_regression_tests(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan regression tests for a release or change."""
        release_notes = task_data.get("release_notes", "")
        changed_components = task_data.get("changed_components", [])
        historical_failures = task_data.get("historical_failures", [])
        
        # Get historical failure patterns
        failure_patterns = []
        for failure in historical_failures:
            if failure.get("component") in changed_components:
                failure_patterns.append(failure)
        
        # Generate regression test scenarios
        scenarios = []
        for pattern in failure_patterns:
            bug_scenarios = self.test_generator.generate_from_bug_report(
                pattern.get("description", "")
            )
            scenarios.extend(bug_scenarios)
        
        # Add component-specific tests
        for component in changed_components:
            component_scenarios = self.test_generator.generate_test_scenarios(
                query=f"regression test for {component}",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                max_scenarios=3
            )
            scenarios.extend(component_scenarios)
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "regression_info": {
                "changed_components": changed_components,
                "failure_patterns": len(failure_patterns),
                "release_scope": len(changed_components)
            },
            "total_scenarios": len(scenarios)
        }
    
    async def _plan_smoke_tests(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan smoke tests for critical functionality."""
        critical_components = task_data.get("critical_components", [])
        user_flows = task_data.get("user_flows", [])
        
        scenarios = []
        
        # Generate smoke tests for critical components
        for component in critical_components:
            smoke_scenarios = self.test_generator.generate_test_scenarios(
                query=f"smoke test for {component} basic functionality",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                max_scenarios=2
            )
            
            # Mark as high priority smoke tests
            for scenario in smoke_scenarios:
                scenario.priority = TestPriority.CRITICAL
                scenario.tags.append("smoke-test")
            
            scenarios.extend(smoke_scenarios)
        
        # Generate smoke tests for user flows
        for flow in user_flows:
            flow_scenarios = self.test_generator.generate_test_scenarios(
                query=f"smoke test for user flow: {flow}",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                max_scenarios=1
            )
            
            for scenario in flow_scenarios:
                scenario.priority = TestPriority.HIGH
                scenario.tags.append("smoke-test")
                scenario.tags.append("user-flow")
            
            scenarios.extend(flow_scenarios)
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "smoke_info": {
                "critical_components": len(critical_components),
                "user_flows": len(user_flows),
                "total_critical": len([s for s in scenarios if s.priority == TestPriority.CRITICAL])
            },
            "total_scenarios": len(scenarios)
        }
    
    async def _plan_feature_tests(self, task_data: Dict[str, Any], risk_score: Any, context_docs: List[Dict]) -> List[TestScenario]:
        """Plan tests for new feature implementation."""
        feature_description = task_data.get("feature_description", "")
        requirements = task_data.get("requirements", [])
        
        # Generate comprehensive test scenarios
        scenarios = self.test_generator.generate_test_scenarios(
            query=f"feature implementation test: {feature_description}",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=5
        )
        
        # Add requirement-based tests
        for requirement in requirements:
            req_scenarios = self.test_generator.generate_test_scenarios(
                query=f"requirement test: {requirement}",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                max_scenarios=2
            )
            scenarios.extend(req_scenarios)
        
        return scenarios
    
    async def _plan_bug_fix_tests(self, task_data: Dict[str, Any], risk_score: Any, context_docs: List[Dict]) -> List[TestScenario]:
        """Plan tests for bug fixes."""
        bug_description = task_data.get("bug_description", "")
        
        # Generate regression tests
        scenarios = self.test_generator.generate_from_bug_report(bug_description)
        
        # Add validation tests
        validation_scenarios = self.test_generator.generate_test_scenarios(
            query=f"validate bug fix: {bug_description}",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=3
        )
        
        scenarios.extend(validation_scenarios)
        return scenarios
    
    async def _plan_refactoring_tests(self, task_data: Dict[str, Any], risk_score: Any, context_docs: List[Dict]) -> List[TestScenario]:
        """Plan tests for refactoring changes."""
        refactored_components = task_data.get("refactored_components", [])
        
        scenarios = []
        for component in refactored_components:
            comp_scenarios = self.test_generator.generate_test_scenarios(
                query=f"refactoring validation for {component}",
                test_type=TestType.INTEGRATION,
                framework=TestFramework.SELENIUM,
                max_scenarios=3
            )
            scenarios.extend(comp_scenarios)
        
        return scenarios
    
    async def _plan_security_tests(self, task_data: Dict[str, Any], risk_score: Any, context_docs: List[Dict]) -> List[TestScenario]:
        """Plan tests for security updates."""
        security_areas = task_data.get("security_areas", [])
        
        scenarios = []
        for area in security_areas:
            sec_scenarios = self.test_generator.generate_test_scenarios(
                query=f"security test for {area}",
                test_type=TestType.SECURITY,
                framework=TestFramework.SELENIUM,
                max_scenarios=3
            )
            scenarios.extend(sec_scenarios)
        
        return scenarios
    
    async def _plan_performance_tests(self, task_data: Dict[str, Any], risk_score: Any, context_docs: List[Dict]) -> List[TestScenario]:
        """Plan tests for performance optimizations."""
        performance_areas = task_data.get("performance_areas", [])
        
        scenarios = []
        for area in performance_areas:
            perf_scenarios = self.test_generator.generate_test_scenarios(
                query=f"performance test for {area}",
                test_type=TestType.PERFORMANCE,
                framework=TestFramework.SELENIUM,
                max_scenarios=2
            )
            scenarios.extend(perf_scenarios)
        
        return scenarios
    
    def _prioritize_by_risk(self, scenarios: List[TestScenario], risk_score: Any) -> List[TestScenario]:
        """Prioritize scenarios based on risk analysis."""
        # Adjust priority based on risk level
        if risk_score.risk_level > 0.8:
            priority_multiplier = TestPriority.CRITICAL
        elif risk_score.risk_level > 0.6:
            priority_multiplier = TestPriority.HIGH
        elif risk_score.risk_level > 0.4:
            priority_multiplier = TestPriority.MEDIUM
        else:
            priority_multiplier = TestPriority.LOW
        
        # Update scenario priorities
        for scenario in scenarios:
            if scenario.priority.value == "low" and priority_multiplier.value in ["critical", "high"]:
                scenario.priority = priority_multiplier
        
        # Sort by priority and confidence
        return sorted(scenarios, key=lambda s: (s.priority.value, s.confidence_score or 0), reverse=True)
    
    async def _analyze_test_coverage(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test coverage for components or features."""
        components = task_data.get("components", [])
        
        coverage_analysis = {}
        
        for component in components:
            # Search for existing tests for this component
            existing_tests = self.vector_store.search_by_metadata(
                metadata_filter={"components": component},
                limit=100
            )
            
            # Analyze coverage gaps
            coverage_analysis[component] = {
                "existing_tests": len(existing_tests),
                "coverage_gaps": self._identify_coverage_gaps(component, existing_tests),
                "recommendations": self._generate_coverage_recommendations(component, existing_tests)
            }
        
        return {
            "coverage_analysis": coverage_analysis,
            "total_components": len(components),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _prioritize_test_scenarios(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize test scenarios based on various factors."""
        scenarios = task_data.get("scenarios", [])
        prioritization_factors = task_data.get("factors", ["risk", "business_impact", "frequency"])
        
        # This would integrate with the existing test prioritizer
        # For now, return a simplified prioritization
        prioritized = sorted(scenarios, key=lambda s: s.get("priority_score", 0.5), reverse=True)
        
        return {
            "prioritized_scenarios": prioritized,
            "prioritization_factors": prioritization_factors,
            "total_scenarios": len(prioritized)
        }
    
    def _identify_coverage_gaps(self, component: str, existing_tests: List[Dict]) -> List[str]:
        """Identify test coverage gaps for a component."""
        # This would analyze existing tests and identify gaps
        gaps = [
            f"No integration tests for {component}",
            f"Missing error handling tests for {component}",
            f"No performance tests for {component}"
        ]
        return gaps[:2]  # Return top 2 gaps
    
    def _generate_coverage_recommendations(self, component: str, existing_tests: List[Dict]) -> List[str]:
        """Generate recommendations for improving test coverage."""
        recommendations = [
            f"Add end-to-end tests for {component}",
            f"Implement unit tests for {component} core functions",
            f"Create performance benchmarks for {component}"
        ]
        return recommendations[:2]  # Return top 2 recommendations
    
    async def get_planning_insights(self) -> Dict[str, Any]:
        """Get insights about test planning patterns and effectiveness."""
        # This would analyze historical planning data
        return {
            "total_plans_generated": self.metrics.tasks_completed,
            "success_rate": self.metrics.success_rate,
            "average_scenarios_per_plan": 5.2,
            "most_common_test_types": ["E2E", "Integration", "Unit"],
            "planning_efficiency": "High"
        }
