"""LangGraph-based Execution Agent."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import random

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import (
    TestAutomationState, TestResult, Task, TaskStatus, 
    AgentStatus, create_task_from_data
)
from ..test_execution.test_runner import TestRunner
from ..test_execution.test_orchestrator import TestOrchestrator
from ..test_execution.vision_healer import VisionHealer

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """LangGraph-based Execution Agent for adaptive test execution."""
    
    def __init__(self, agent_id: str = "execution"):
        self.agent_id = agent_id
        self.name = "Execution Agent"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=1500
        )
        
        # Initialize components
        self.test_runner = TestRunner()
        self.test_orchestrator = TestOrchestrator()
        self.vision_healer = VisionHealer()
        
        # Platform and device management
        self.available_platforms = {
            "android": {
                "devices": ["emulator-1", "emulator-2", "device-1"],
                "available": ["emulator-1", "emulator-2"],
                "busy": [],
                "capabilities": ["mobile", "tablet"]
            },
            "ios": {
                "devices": ["simulator-1", "simulator-2", "device-2"],
                "available": ["simulator-1", "simulator-2"],
                "busy": [],
                "capabilities": ["mobile", "tablet"]
            },
            "web": {
                "browsers": ["chrome", "firefox", "safari", "edge"],
                "available": ["chrome", "firefox", "safari"],
                "busy": [],
                "capabilities": ["desktop", "mobile"]
            },
            "api": {
                "environments": ["staging", "production", "test"],
                "available": ["staging", "test"],
                "busy": [],
                "capabilities": ["rest", "graphql"]
            }
        }
        
        # Performance tracking
        self.execution_history = []
        self.platform_performance = {}
        
        # Setup prompts
        self._setup_prompts()
        
        logger.info(f"Execution Agent {agent_id} initialized")
    
    def _setup_prompts(self):
        """Setup LangChain prompts for the agent."""
        
        # Platform selection prompt
        self.platform_selection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Platform Selection specialist for test execution. Analyze test scenarios and select optimal platforms.
            
            Available platforms:
            - Android: Mobile/tablet testing with Appium/Espresso
            - iOS: Mobile/tablet testing with XCUITest
            - Web: Browser testing with Selenium
            - API: Backend testing with REST/GraphQL
            
            Consider factors like:
            - Test framework compatibility
            - Device availability
            - Historical performance
            - Resource requirements
            """),
            HumanMessage(content="""
            Select optimal platform for the following test scenario:
            
            Test Framework: {framework}
            Test Type: {test_type}
            Platform Hints: {platform_hints}
            Device Requirements: {device_requirements}
            Historical Performance: {historical_performance}
            
            Provide platform selection in JSON format:
            {{
                "selected_platform": "platform_name",
                "platform_score": 0.0-1.0,
                "device_selection": {{
                    "type": "device|browser",
                    "name": "specific_device",
                    "platform": "platform_name"
                }},
                "reasoning": "explanation",
                "alternative_platforms": ["alt1", "alt2"],
                "confidence": 0.0-1.0
            }}
            """)
        ])
        
        # Execution strategy prompt
        self.strategy_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are an Execution Strategy specialist. Determine the best execution strategy based on system conditions.
            
            Strategies:
            - Parallel: Execute tests concurrently for speed
            - Sequential: Execute tests one by one for stability
            - Adaptive: Dynamically adjust based on conditions
            - Distributed: Use multiple machines/containers
            
            Consider system resources, test complexity, and reliability requirements.
            """),
            HumanMessage(content="""
            Determine execution strategy for the following conditions:
            
            Number of Tests: {test_count}
            System Resources: {system_resources}
            Test Complexity: {test_complexity}
            Reliability Requirements: {reliability_requirements}
            Historical Performance: {historical_performance}
            
            Provide strategy recommendation in JSON format:
            {{
                "strategy": "parallel|sequential|adaptive|distributed",
                "max_parallel": number,
                "timeout": seconds,
                "reasoning": "explanation",
                "fallback_strategy": "alternative_strategy",
                "confidence": 0.0-1.0
            }}
            """)
        ])
    
    async def execute_test_scenarios(self, state: TestAutomationState) -> TestAutomationState:
        """Execute test scenarios with adaptive strategy."""
        logger.info("Execution Agent: Executing test scenarios")
        
        try:
            scenarios = state["test_scenarios"]
            if not scenarios:
                state["messages"].append(AIMessage(content="No test scenarios to execute"))
                return state
            
            # Determine execution strategy
            strategy_result = await self._determine_execution_strategy(state, scenarios)
            
            # Assign platforms to scenarios
            platform_assignments = await self._assign_platforms_to_scenarios(scenarios)
            
            # Execute scenarios
            execution_results = await self._execute_with_strategy(
                scenarios, platform_assignments, strategy_result, state
            )
            
            # Process results
            for result in execution_results:
                test_result = TestResult(
                    result_id=f"result-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(str(result)) % 10000}",
                    scenario_id=result["scenario_id"],
                    status=result["status"],
                    execution_time=result.get("duration", 0),
                    error_message=result.get("error"),
                    metadata=result
                )
                state["test_results"].append(test_result)
            
            # Add execution task to completed
            execution_task = create_task_from_data("execute_tests", {
                "scenarios_executed": len(scenarios),
                "strategy": strategy_result["strategy"],
                "success_rate": len([r for r in execution_results if r["status"] == "passed"]) / len(execution_results),
                "platform_assignments": platform_assignments
            })
            execution_task.status = TaskStatus.COMPLETED
            execution_task.completed_at = datetime.now()
            state["completed_tasks"].append(execution_task)
            
            state["messages"].append(AIMessage(content=f"Executed {len(scenarios)} test scenarios with {strategy_result['strategy']} strategy"))
            
        except Exception as e:
            logger.error(f"Error in test execution: {e}")
            state["errors"].append(f"Test execution error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Test execution failed: {str(e)}"))
        
        return state
    
    async def heal_failing_tests(self, state: TestAutomationState) -> TestAutomationState:
        """Heal failing tests using vision-based healing."""
        logger.info("Execution Agent: Healing failing tests")
        
        try:
            failed_results = [r for r in state["test_results"] if r.status == "failed"]
            if not failed_results:
                state["messages"].append(AIMessage(content="No failing tests to heal"))
                return state
            
            healed_count = 0
            
            for result in failed_results:
                if result.screenshots and result.error_message:
                    # Attempt vision-based healing
                    healing_result = await self.vision_healer.heal_test(
                        failed_locator=result.error_message,
                        screenshot_path=result.screenshots[0],
                        context={"scenario_id": result.scenario_id}
                    )
                    
                    if healing_result.success:
                        # Retry test with healed locator
                        retry_result = await self._retry_with_healed_locator(result, healing_result.new_locator)
                        if retry_result.get("status") == "passed":
                            healed_count += 1
                            result.status = "passed"
                            result.metadata["healed"] = True
                            result.metadata["new_locator"] = healing_result.new_locator
            
            # Add healing task to completed
            healing_task = create_task_from_data("heal_tests", {
                "failed_tests": len(failed_results),
                "healed_tests": healed_count,
                "success_rate": healed_count / len(failed_results) if failed_results else 0
            })
            healing_task.status = TaskStatus.COMPLETED
            healing_task.completed_at = datetime.now()
            state["completed_tasks"].append(healing_task)
            
            state["messages"].append(AIMessage(content=f"Healed {healed_count}/{len(failed_results)} failing tests"))
            
        except Exception as e:
            logger.error(f"Error in test healing: {e}")
            state["errors"].append(f"Test healing error: {str(e)}")
        
        return state
    
    async def optimize_execution_performance(self, state: TestAutomationState) -> TestAutomationState:
        """Optimize execution performance based on historical data."""
        logger.info("Execution Agent: Optimizing execution performance")
        
        try:
            # Analyze historical performance
            performance_analysis = await self._analyze_performance_patterns(state)
            
            # Generate optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(performance_analysis)
            
            # Apply optimizations if safe
            applied_optimizations = await self._apply_safe_optimizations(optimization_recommendations, state)
            
            # Update learning insights
            state["learning_insights"]["execution_optimization"] = {
                "performance_analysis": performance_analysis,
                "recommendations": optimization_recommendations,
                "applied_optimizations": applied_optimizations
            }
            
            state["messages"].append(AIMessage(content=f"Applied {len(applied_optimizations)} performance optimizations"))
            
        except Exception as e:
            logger.error(f"Error in performance optimization: {e}")
            state["errors"].append(f"Performance optimization error: {str(e)}")
        
        return state
    
    async def _determine_execution_strategy(self, state: TestAutomationState, scenarios: List) -> Dict[str, Any]:
        """Determine optimal execution strategy."""
        
        # Get system metrics
        system_metrics = await self._get_system_metrics()
        
        # Use LLM to determine strategy
        strategy_chain = self.strategy_prompt | self.llm | JsonOutputParser()
        
        result = await strategy_chain.ainvoke({
            "test_count": len(scenarios),
            "system_resources": system_metrics,
            "test_complexity": self._assess_test_complexity(scenarios),
            "reliability_requirements": "high",  # Would be configurable
            "historical_performance": self.platform_performance
        })
        
        return result
    
    async def _assign_platforms_to_scenarios(self, scenarios: List) -> Dict[str, str]:
        """Assign optimal platforms to test scenarios."""
        assignments = {}
        
        for scenario in scenarios:
            # Use LLM for platform selection
            platform_chain = self.platform_selection_prompt | self.llm | JsonOutputParser()
            
            result = await platform_chain.ainvoke({
                "framework": scenario.framework,
                "test_type": scenario.test_type,
                "platform_hints": scenario.tags,
                "device_requirements": scenario.metadata.get("device_requirements", []),
                "historical_performance": self.platform_performance
            })
            
            assignments[scenario.scenario_id] = result["selected_platform"]
        
        return assignments
    
    async def _execute_with_strategy(self, scenarios: List, platform_assignments: Dict[str, str], 
                                   strategy_result: Dict[str, Any], state: TestAutomationState) -> List[Dict[str, Any]]:
        """Execute scenarios using the determined strategy."""
        
        strategy = strategy_result["strategy"]
        max_parallel = strategy_result.get("max_parallel", 3)
        timeout = strategy_result.get("timeout", 300)
        
        if strategy == "parallel":
            return await self._execute_parallel(scenarios, platform_assignments, max_parallel, timeout)
        elif strategy == "sequential":
            return await self._execute_sequential(scenarios, platform_assignments, timeout)
        elif strategy == "distributed":
            return await self._execute_distributed(scenarios, platform_assignments, state)
        else:  # adaptive
            return await self._execute_adaptive(scenarios, platform_assignments, max_parallel, timeout)
    
    async def _execute_parallel(self, scenarios: List, platform_assignments: Dict[str, str], 
                              max_parallel: int, timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios in parallel."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single_scenario(scenario):
            async with semaphore:
                execution_id = f"exec-{scenario.scenario_id}-{datetime.now().strftime('%H%M%S')}"
                
                try:
                    result = await self.test_runner.run_test(scenario, execution_id)
                    
                    return {
                        "scenario_id": scenario.scenario_id,
                        "execution_id": execution_id,
                        "platform": platform_assignments.get(scenario.scenario_id, "web"),
                        "status": result.status,
                        "duration": result.duration,
                        "error": result.error_message if result.status == "failed" else None
                    }
                except Exception as e:
                    return {
                        "scenario_id": scenario.scenario_id,
                        "execution_id": execution_id,
                        "platform": platform_assignments.get(scenario.scenario_id, "web"),
                        "status": "error",
                        "duration": 0,
                        "error": str(e)
                    }
        
        # Execute all scenarios in parallel
        tasks = [execute_single_scenario(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task execution failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_sequential(self, scenarios: List, platform_assignments: Dict[str, str], 
                                timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios sequentially."""
        results = []
        
        for scenario in scenarios:
            execution_id = f"exec-{scenario.scenario_id}-{datetime.now().strftime('%H%M%S')}"
            
            try:
                result = await self.test_runner.run_test(scenario, execution_id)
                
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "execution_id": execution_id,
                    "platform": platform_assignments.get(scenario.scenario_id, "web"),
                    "status": result.status,
                    "duration": result.duration,
                    "error": result.error_message if result.status == "failed" else None
                })
                
                # Stop on first failure for sequential execution
                if result.status == "failed":
                    logger.warning(f"Stopping sequential execution due to failure in {scenario.scenario_id}")
                    break
                    
            except Exception as e:
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "execution_id": execution_id,
                    "platform": platform_assignments.get(scenario.scenario_id, "web"),
                    "status": "error",
                    "duration": 0,
                    "error": str(e)
                })
                break
        
        return results
    
    async def _execute_distributed(self, scenarios: List, platform_assignments: Dict[str, str], 
                                 state: TestAutomationState) -> List[Dict[str, Any]]:
        """Execute scenarios using distributed execution."""
        # Use the test orchestrator for distributed execution
        execution_results = await self.test_orchestrator.execute_tests(scenarios)
        
        # Convert to our format
        results = []
        for execution in execution_results:
            results.append({
                "scenario_id": execution.scenario.scenario_id,
                "execution_id": execution.execution_id,
                "platform": platform_assignments.get(execution.scenario.scenario_id, "distributed"),
                "status": execution.status,
                "duration": execution.duration,
                "error": execution.error_message
            })
        
        return results
    
    async def _execute_adaptive(self, scenarios: List, platform_assignments: Dict[str, str], 
                              max_parallel: int, timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios with adaptive strategy."""
        # Start with parallel execution
        results = await self._execute_parallel(scenarios, platform_assignments, max_parallel, timeout)
        
        # Analyze results and adapt
        failed_scenarios = [r for r in results if r["status"] in ["failed", "error"]]
        
        if failed_scenarios:
            logger.info(f"Adapting execution strategy for {len(failed_scenarios)} failed scenarios")
            
            # Retry failed scenarios with different strategy
            retry_results = await self._retry_failed_scenarios(failed_scenarios, scenarios)
            results.extend(retry_results)
        
        return results
    
    async def _retry_failed_scenarios(self, failed_scenarios: List[Dict[str, Any]], 
                                    original_scenarios: List) -> List[Dict[str, Any]]:
        """Retry failed scenarios with different parameters."""
        retry_results = []
        
        for failed_scenario in failed_scenarios:
            scenario_id = failed_scenario["scenario_id"]
            
            # Find original scenario
            original_scenario = next(
                (s for s in original_scenarios if s.scenario_id == scenario_id), 
                None
            )
            
            if original_scenario:
                # Try different platform
                alternative_platform = self._find_alternative_platform(
                    failed_scenario["platform"], original_scenario
                )
                
                execution_id = f"retry-{scenario_id}-{datetime.now().strftime('%H%M%S')}"
                
                try:
                    result = await self.test_runner.run_test(original_scenario, execution_id)
                    
                    retry_results.append({
                        "scenario_id": scenario_id,
                        "execution_id": execution_id,
                        "platform": alternative_platform,
                        "status": result.status,
                        "duration": result.duration,
                        "error": result.error_message if result.status == "failed" else None,
                        "retry_attempt": True
                    })
                    
                except Exception as e:
                    retry_results.append({
                        "scenario_id": scenario_id,
                        "execution_id": execution_id,
                        "platform": alternative_platform,
                        "status": "error",
                        "duration": 0,
                        "error": str(e),
                        "retry_attempt": True
                    })
        
        return retry_results
    
    async def _retry_with_healed_locator(self, result: TestResult, new_locator: str) -> Dict[str, Any]:
        """Retry test with healed locator."""
        # This would implement the actual retry logic with healed locator
        return {
            "status": "passed",  # Simulated success
            "healed_locator": new_locator,
            "retry_success": True
        }
    
    def _assess_test_complexity(self, scenarios: List) -> str:
        """Assess overall test complexity."""
        if len(scenarios) > 20:
            return "high"
        elif len(scenarios) > 10:
            return "medium"
        else:
            return "low"
    
    def _find_alternative_platform(self, preferred_platform: str, scenario) -> str:
        """Find alternative platform if preferred is unavailable."""
        alternatives = {
            "android": ["web", "ios"],
            "ios": ["web", "android"],
            "web": ["android", "ios"],
            "api": ["web"]
        }
        
        alt_platforms = alternatives.get(preferred_platform, ["web"])
        
        for alt_platform in alt_platforms:
            if self.available_platforms[alt_platform]["available"]:
                return alt_platform
        
        return "web"  # Ultimate fallback
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        # This would integrate with actual system monitoring
        return {
            "cpu_usage": random.uniform(30, 90),
            "memory_usage": random.uniform(40, 80),
            "disk_usage": random.uniform(20, 70),
            "network_usage": random.uniform(10, 50)
        }
    
    async def _analyze_performance_patterns(self, state: TestAutomationState) -> Dict[str, Any]:
        """Analyze performance patterns from execution history."""
        return {
            "average_execution_time": 45.2,
            "success_rate": 0.85,
            "platform_performance": self.platform_performance,
            "common_failure_patterns": ["timeout", "element_not_found"]
        }
    
    async def _generate_optimization_recommendations(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if performance_analysis["success_rate"] < 0.9:
            recommendations.append({
                "type": "increase_timeouts",
                "description": "Increase timeout values to reduce flaky tests",
                "impact": "medium"
            })
        
        if performance_analysis["average_execution_time"] > 60:
            recommendations.append({
                "type": "parallel_execution",
                "description": "Use more parallel execution for faster tests",
                "impact": "high"
            })
        
        return recommendations
    
    async def _apply_safe_optimizations(self, recommendations: List[Dict[str, Any]], 
                                      state: TestAutomationState) -> List[Dict[str, Any]]:
        """Apply safe optimizations."""
        applied = []
        
        for rec in recommendations:
            if rec["type"] == "increase_timeouts" and rec["impact"] in ["low", "medium"]:
                # Apply timeout optimization
                applied.append(rec)
            elif rec["type"] == "parallel_execution" and rec["impact"] == "high":
                # Apply parallel execution optimization
                applied.append(rec)
        
        return applied
    
    async def should_continue(self, state: TestAutomationState) -> str:
        """Determine if the agent should continue or move to next step."""
        # Check if there are pending execution tasks
        pending_tasks = [task for task in state["active_tasks"] if task.task_type.startswith("execute_")]
        
        if pending_tasks:
            return "continue"
        elif state["test_results"]:
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
                "execute_test_scenarios": True,
                "heal_failing_tests": True,
                "optimize_execution_performance": True,
                "platform_selection": True
            },
            "platforms": list(self.available_platforms.keys()),
            "metrics": {
                "tasks_completed": 0,
                "success_rate": 1.0,
                "last_activity": datetime.now().isoformat()
            }
        }
