"""LangGraph-based Test Planner Agent."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import (
    TestAutomationState, TestScenario, Task, TaskStatus, 
    AgentStatus, create_task_from_data, add_task_to_state
)
from ..test_generation.test_scenario import TestType, TestFramework, TestPriority
from ..test_generation.test_generator import TestCaseGenerator
from ..data_ingestion.vector_store import VectorStoreManager
from ..observability.risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)


class TestPlannerAgent:
    """LangGraph-based Test Planner Agent for generating test scenarios."""
    
    def __init__(self, agent_id: str = "test-planner"):
        self.agent_id = agent_id
        self.name = "Test Planner Agent"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize components
        self.test_generator = TestCaseGenerator()
        self.vector_store = VectorStoreManager()
        self.risk_analyzer = RiskAnalyzer()
        
        # Setup prompts
        self._setup_prompts()
        
        logger.info(f"Test Planner Agent {agent_id} initialized")
    
    def _setup_prompts(self):
        """Setup LangChain prompts for the agent."""
        
        # Test planning prompt
        self.planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Test Planner Agent responsible for analyzing code changes and generating comprehensive test scenarios.
            
            Your capabilities:
            - Analyze code changes and commit metadata
            - Generate test scenarios based on change type
            - Assess risk and prioritize tests
            - Integrate requirements and acceptance criteria
            - Suggest test coverage improvements
            
            Always provide structured, actionable test scenarios that can be executed by test automation tools.
            """),
            HumanMessage(content="""
            Analyze the following change and generate test scenarios:
            
            Change Type: {change_type}
            Code Diff: {diff_content}
            Changed Files: {changed_files}
            Commit Message: {commit_message}
            Requirements: {requirements}
            
            Please generate test scenarios in JSON format with the following structure:
            {{
                "scenarios": [
                    {{
                        "scenario_id": "unique_id",
                        "title": "Test scenario title",
                        "description": "Detailed description",
                        "test_type": "unit|integration|e2e|performance|security",
                        "framework": "appium|selenium|espresso|xcuitest|pytest",
                        "priority": "critical|high|medium|low",
                        "tags": ["tag1", "tag2"],
                        "steps": ["step1", "step2"],
                        "expected_result": "Expected outcome",
                        "metadata": {{"additional": "info"}}
                    }}
                ],
                "risk_assessment": {{
                    "risk_level": "low|medium|high|critical",
                    "confidence": 0.0-1.0,
                    "affected_areas": ["area1", "area2"],
                    "recommendations": ["rec1", "rec2"]
                }},
                "coverage_analysis": {{
                    "existing_coverage": 0.0-1.0,
                    "gaps_identified": ["gap1", "gap2"],
                    "improvement_suggestions": ["suggestion1", "suggestion2"]
                }}
            }}
            """)
        ])
        
        # Risk analysis prompt
        self.risk_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Risk Analysis specialist for test planning. Analyze code changes and assess testing risks.
            """),
            HumanMessage(content="""
            Analyze the risk of the following change:
            
            Change Type: {change_type}
            Code Diff: {diff_content}
            Changed Files: {changed_files}
            Historical Failures: {historical_failures}
            
            Provide risk assessment in JSON format:
            {{
                "risk_level": "low|medium|high|critical",
                "confidence": 0.0-1.0,
                "risk_factors": ["factor1", "factor2"],
                "affected_components": ["component1", "component2"],
                "testing_recommendations": ["rec1", "rec2"],
                "escalation_needed": true|false
            }}
            """)
        ])
    
    async def plan_tests_for_change(self, state: TestAutomationState) -> TestAutomationState:
        """Plan tests for a code change."""
        logger.info("Test Planner Agent: Planning tests for change")
        
        try:
            # Extract change information from state
            change_type = state.get("code_changes", {}).get("change_type", "feature_implementation")
            diff_content = state.get("code_changes", {}).get("diff_content", "")
            changed_files = state.get("code_changes", {}).get("changed_files", [])
            commit_message = state.get("commit_metadata", {}).get("message", "")
            requirements = state.get("requirements", [])
            
            # Get historical context
            context_query = f"{change_type} {commit_message} {' '.join(changed_files)}"
            context_docs = self.vector_store.search_similar(
                query=context_query,
                n_results=5,
                metadata_filter={"source": {"$in": ["jira", "analytics"]}}
            )
            
            # Generate test scenarios using LLM
            planning_chain = self.planning_prompt | self.llm | JsonOutputParser()
            
            result = await planning_chain.ainvoke({
                "change_type": change_type,
                "diff_content": diff_content,
                "changed_files": changed_files,
                "commit_message": commit_message,
                "requirements": requirements
            })
            
            # Convert to TestScenario objects
            scenarios = []
            for scenario_data in result.get("scenarios", []):
                scenario = TestScenario(
                    scenario_id=scenario_data["scenario_id"],
                    title=scenario_data["title"],
                    description=scenario_data["description"],
                    test_type=scenario_data["test_type"],
                    framework=scenario_data["framework"],
                    priority=scenario_data["priority"],
                    tags=scenario_data.get("tags", []),
                    steps=scenario_data.get("steps", []),
                    expected_result=scenario_data.get("expected_result", ""),
                    metadata=scenario_data.get("metadata", {})
                )
                scenarios.append(scenario)
            
            # Add scenarios to state
            for scenario in scenarios:
                state["test_scenarios"].append(scenario)
            
            # Update state with results
            state["messages"].append(AIMessage(content=f"Generated {len(scenarios)} test scenarios"))
            
            # Add planning task to completed
            planning_task = create_task_from_data("plan_tests", {
                "change_type": change_type,
                "scenarios_generated": len(scenarios),
                "risk_assessment": result.get("risk_assessment", {}),
                "coverage_analysis": result.get("coverage_analysis", {})
            })
            planning_task.status = TaskStatus.COMPLETED
            planning_task.completed_at = datetime.now()
            state["completed_tasks"].append(planning_task)
            
            logger.info(f"Test planning completed: {len(scenarios)} scenarios generated")
            
        except Exception as e:
            logger.error(f"Error in test planning: {e}")
            state["errors"].append(f"Test planning error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Test planning failed: {str(e)}"))
        
        return state
    
    async def plan_tests_for_ticket(self, state: TestAutomationState) -> TestAutomationState:
        """Plan tests for a Jira ticket or user story."""
        logger.info("Test Planner Agent: Planning tests for ticket")
        
        try:
            # Extract ticket information from state
            ticket_data = state.get("code_changes", {}).get("ticket", {})
            ticket_type = ticket_data.get("issue_type", "")
            description = ticket_data.get("description", "")
            acceptance_criteria = state.get("requirements", [])
            
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
            
            # Convert to TestScenario objects and add to state
            for scenario in scenarios:
                test_scenario = TestScenario(
                    scenario_id=scenario.scenario_id,
                    title=scenario.title,
                    description=scenario.description,
                    test_type=scenario.test_type.value,
                    framework=scenario.framework.value,
                    priority=scenario.priority.value,
                    tags=scenario.tags,
                    steps=scenario.test_steps,
                    expected_result=scenario.expected_result,
                    metadata=scenario.metadata
                )
                state["test_scenarios"].append(test_scenario)
            
            state["messages"].append(AIMessage(content=f"Generated {len(scenarios)} test scenarios for ticket"))
            
        except Exception as e:
            logger.error(f"Error in ticket test planning: {e}")
            state["errors"].append(f"Ticket test planning error: {str(e)}")
        
        return state
    
    async def plan_regression_tests(self, state: TestAutomationState) -> TestAutomationState:
        """Plan regression tests for a release or change."""
        logger.info("Test Planner Agent: Planning regression tests")
        
        try:
            release_notes = state.get("code_changes", {}).get("release_notes", "")
            changed_components = state.get("code_changes", {}).get("changed_components", [])
            historical_failures = state.get("code_changes", {}).get("historical_failures", [])
            
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
                for scenario in bug_scenarios:
                    test_scenario = TestScenario(
                        scenario_id=scenario.scenario_id,
                        title=f"Regression: {scenario.title}",
                        description=scenario.description,
                        test_type=scenario.test_type.value,
                        framework=scenario.framework.value,
                        priority=TestPriority.HIGH.value,
                        tags=scenario.tags + ["regression"],
                        steps=scenario.test_steps,
                        expected_result=scenario.expected_result,
                        metadata=scenario.metadata
                    )
                    scenarios.append(test_scenario)
            
            # Add component-specific tests
            for component in changed_components:
                component_scenarios = self.test_generator.generate_test_scenarios(
                    query=f"regression test for {component}",
                    test_type=TestType.E2E,
                    framework=TestFramework.APPIUM,
                    max_scenarios=3
                )
                
                for scenario in component_scenarios:
                    test_scenario = TestScenario(
                        scenario_id=scenario.scenario_id,
                        title=f"Component Regression: {scenario.title}",
                        description=scenario.description,
                        test_type=scenario.test_type.value,
                        framework=scenario.framework.value,
                        priority=TestPriority.MEDIUM.value,
                        tags=scenario.tags + ["regression", "component"],
                        steps=scenario.test_steps,
                        expected_result=scenario.expected_result,
                        metadata=scenario.metadata
                    )
                    scenarios.append(test_scenario)
            
            # Add scenarios to state
            for scenario in scenarios:
                state["test_scenarios"].append(scenario)
            
            state["messages"].append(AIMessage(content=f"Generated {len(scenarios)} regression test scenarios"))
            
        except Exception as e:
            logger.error(f"Error in regression test planning: {e}")
            state["errors"].append(f"Regression test planning error: {str(e)}")
        
        return state
    
    async def analyze_test_coverage(self, state: TestAutomationState) -> TestAutomationState:
        """Analyze test coverage for components or features."""
        logger.info("Test Planner Agent: Analyzing test coverage")
        
        try:
            components = state.get("code_changes", {}).get("components", [])
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
            
            # Update learning insights with coverage analysis
            state["learning_insights"]["coverage_analysis"] = coverage_analysis
            
            state["messages"].append(AIMessage(content=f"Coverage analysis completed for {len(components)} components"))
            
        except Exception as e:
            logger.error(f"Error in coverage analysis: {e}")
            state["errors"].append(f"Coverage analysis error: {str(e)}")
        
        return state
    
    async def assess_risk(self, state: TestAutomationState) -> TestAutomationState:
        """Assess risk of code changes."""
        logger.info("Test Planner Agent: Assessing risk")
        
        try:
            # Extract change information
            change_type = state.get("code_changes", {}).get("change_type", "")
            diff_content = state.get("code_changes", {}).get("diff_content", "")
            changed_files = state.get("code_changes", {}).get("changed_files", [])
            historical_failures = state.get("code_changes", {}).get("historical_failures", [])
            
            # Use LLM for risk analysis
            risk_chain = self.risk_analysis_prompt | self.llm | JsonOutputParser()
            
            result = await risk_chain.ainvoke({
                "change_type": change_type,
                "diff_content": diff_content,
                "changed_files": changed_files,
                "historical_failures": historical_failures
            })
            
            # Update state with risk assessment
            state["learning_insights"]["risk_assessment"] = result
            
            # Check if escalation is needed
            if result.get("escalation_needed", False):
                state["escalation_needed"] = True
                state["escalation_level"] = "high" if result.get("risk_level") == "critical" else "medium"
            
            state["messages"].append(AIMessage(content=f"Risk assessment completed: {result.get('risk_level', 'unknown')} risk"))
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            state["errors"].append(f"Risk assessment error: {str(e)}")
        
        return state
    
    def _identify_coverage_gaps(self, component: str, existing_tests: List[Dict[str, Any]]) -> List[str]:
        """Identify test coverage gaps for a component."""
        gaps = [
            f"No integration tests for {component}",
            f"Missing error handling tests for {component}",
            f"No performance tests for {component}"
        ]
        return gaps[:2]  # Return top 2 gaps
    
    def _generate_coverage_recommendations(self, component: str, existing_tests: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving test coverage."""
        recommendations = [
            f"Add end-to-end tests for {component}",
            f"Implement unit tests for {component} core functions",
            f"Create performance benchmarks for {component}"
        ]
        return recommendations[:2]  # Return top 2 recommendations
    
    async def should_continue(self, state: TestAutomationState) -> str:
        """Determine if the agent should continue or move to next step."""
        # Check if there are pending tasks
        pending_tasks = [task for task in state["active_tasks"] if task.task_type.startswith("plan_")]
        
        if pending_tasks:
            return "continue"
        elif state["test_scenarios"]:
            return "next"
        else:
            return "error"
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": "active",
            "capabilities": {
                "plan_tests_for_change": True,
                "plan_tests_for_ticket": True,
                "plan_regression_tests": True,
                "analyze_test_coverage": True,
                "assess_risk": True
            },
            "metrics": {
                "tasks_completed": 0,  # Would be tracked in practice
                "success_rate": 1.0,
                "last_activity": datetime.now().isoformat()
            }
        }
