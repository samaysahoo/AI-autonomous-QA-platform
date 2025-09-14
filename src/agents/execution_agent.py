"""Execution Agent - chooses platform/device and executes tests adaptively."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random

from .base_agent import BaseAgent, AgentCapabilities, MessageType, AgentMessage
from ..test_execution.test_runner import TestRunner
from ..test_execution.test_orchestrator import TestOrchestrator
from ..test_execution.vision_healer import VisionHealer
from ..test_generation.test_scenario import TestScenario, TestFramework, TestType
from ..observability.test_prioritizer import TestPrioritizer

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """Agent responsible for adaptive test execution across platforms and devices."""
    
    def __init__(self, agent_id: str = "execution-001"):
        capabilities = AgentCapabilities(
            can_execute_tests=True,
            supported_platforms=["android", "ios", "web", "api", "desktop"],
            supported_frameworks=["appium", "selenium", "espresso", "xcuitest", "pytest", "jest"],
            max_concurrent_tasks=5
        )
        
        super().__init__(agent_id, "Execution Agent", capabilities)
        
        # Initialize components
        self.test_runner = TestRunner()
        self.test_orchestrator = TestOrchestrator()
        self.vision_healer = VisionHealer()
        self.test_prioritizer = TestPrioritizer()
        
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
        
        # Execution strategies
        self.execution_strategies = {
            "parallel": self._execute_parallel,
            "sequential": self._execute_sequential,
            "adaptive": self._execute_adaptive,
            "distributed": self._execute_distributed
        }
        
        # Performance tracking
        self.execution_history = []
        self.platform_performance = {}
        
        logger.info(f"Execution Agent initialized with {len(self.available_platforms)} platforms")
    
    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this agent can handle a specific task type."""
        return task_type in [
            "execute_test_scenarios",
            "execute_test_suite",
            "adaptive_execution",
            "platform_selection",
            "device_management",
            "heal_failing_tests",
            "performance_optimization"
        ]
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an execution task."""
        task_type = task_data.get("task_type", "")
        
        logger.info(f"Execution Agent processing task: {task_type}")
        
        try:
            if task_type == "execute_test_scenarios":
                return await self._execute_test_scenarios(task_data)
            elif task_type == "execute_test_suite":
                return await self._execute_test_suite(task_data)
            elif task_type == "adaptive_execution":
                return await self._adaptive_execution(task_data)
            elif task_type == "platform_selection":
                return await self._select_optimal_platform(task_data)
            elif task_type == "device_management":
                return await self._manage_devices(task_data)
            elif task_type == "heal_failing_tests":
                return await self._heal_failing_tests(task_data)
            elif task_type == "performance_optimization":
                return await self._optimize_execution_performance(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_type}: {e}")
            raise
    
    async def _execute_test_scenarios(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a set of test scenarios."""
        scenarios_data = task_data.get("scenarios", [])
        execution_strategy = task_data.get("strategy", "adaptive")
        max_parallel = task_data.get("max_parallel", 3)
        timeout = task_data.get("timeout", 1800)  # 30 minutes
        
        # Convert scenario data to TestScenario objects
        scenarios = []
        for scenario_data in scenarios_data:
            scenario = TestScenario.from_dict(scenario_data)
            scenarios.append(scenario)
        
        # Select optimal platforms for execution
        platform_assignments = await self._assign_platforms_to_scenarios(scenarios)
        
        # Execute using selected strategy
        strategy_func = self.execution_strategies.get(execution_strategy, self._execute_adaptive)
        results = await strategy_func(scenarios, platform_assignments, max_parallel, timeout)
        
        # Analyze execution results
        execution_summary = self._analyze_execution_results(results)
        
        return {
            "execution_id": f"exec-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "scenarios_executed": len(scenarios),
            "execution_strategy": execution_strategy,
            "platform_assignments": platform_assignments,
            "results": results,
            "summary": execution_summary,
            "execution_time": execution_summary["total_execution_time"],
            "success_rate": execution_summary["success_rate"]
        }
    
    async def _execute_test_suite(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete test suite."""
        suite_data = task_data.get("test_suite", {})
        scenarios = [TestScenario.from_dict(s) for s in suite_data.get("scenarios", [])]
        suite_type = suite_data.get("suite_type", "regression")
        
        # Determine execution strategy based on suite type
        if suite_type == "smoke":
            strategy = "parallel"
            max_parallel = 5
            timeout = 600  # 10 minutes
        elif suite_type == "regression":
            strategy = "distributed"
            max_parallel = 10
            timeout = 3600  # 1 hour
        else:
            strategy = "adaptive"
            max_parallel = 3
            timeout = 1800  # 30 minutes
        
        # Execute the suite
        return await self._execute_test_scenarios({
            "scenarios": [s.to_dict() for s in scenarios],
            "strategy": strategy,
            "max_parallel": max_parallel,
            "timeout": timeout,
            "suite_type": suite_type
        })
    
    async def _adaptive_execution(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tests using adaptive strategy based on real-time conditions."""
        scenarios_data = task_data.get("scenarios", [])
        initial_strategy = task_data.get("initial_strategy", "parallel")
        
        scenarios = [TestScenario.from_dict(s) for s in scenarios_data]
        
        # Monitor system resources and adjust strategy
        system_metrics = await self._get_system_metrics()
        
        # Adapt execution based on conditions
        if system_metrics["cpu_usage"] > 80:
            strategy = "sequential"
            max_parallel = 1
        elif system_metrics["memory_usage"] > 85:
            strategy = "distributed"
            max_parallel = 2
        else:
            strategy = "parallel"
            max_parallel = min(len(scenarios), 5)
        
        # Execute with adaptive parameters
        platform_assignments = await self._assign_platforms_to_scenarios(scenarios)
        results = await self._execute_adaptive(scenarios, platform_assignments, max_parallel, 1800)
        
        return {
            "execution_id": f"adaptive-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "adaptive_strategy": strategy,
            "system_metrics": system_metrics,
            "results": results,
            "adaptation_reason": self._get_adaptation_reason(system_metrics)
        }
    
    async def _select_optimal_platform(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal platform for test execution."""
        scenario_data = task_data.get("scenario", {})
        scenario = TestScenario.from_dict(scenario_data)
        
        # Analyze scenario requirements
        requirements = self._analyze_scenario_requirements(scenario)
        
        # Score available platforms
        platform_scores = {}
        for platform, platform_info in self.available_platforms.items():
            if platform_info["available"]:  # Only consider available platforms
                score = self._calculate_platform_score(platform, requirements, scenario)
                platform_scores[platform] = score
        
        # Select best platform
        if platform_scores:
            best_platform = max(platform_scores, key=platform_scores.get)
            best_score = platform_scores[best_platform]
        else:
            best_platform = "web"  # Fallback
            best_score = 0.5
        
        # Select specific device/browser
        device_selection = self._select_specific_device(best_platform, requirements)
        
        return {
            "selected_platform": best_platform,
            "platform_score": best_score,
            "device_selection": device_selection,
            "requirements_analysis": requirements,
            "alternative_platforms": sorted(platform_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
        }
    
    async def _assign_platforms_to_scenarios(self, scenarios: List[TestScenario]) -> Dict[str, str]:
        """Assign optimal platforms to test scenarios."""
        assignments = {}
        
        for scenario in scenarios:
            # Determine platform based on scenario characteristics
            if scenario.framework == TestFramework.APPIUM:
                if "mobile" in scenario.tags or "android" in scenario.tags:
                    platform = "android"
                elif "ios" in scenario.tags:
                    platform = "ios"
                else:
                    platform = "android"  # Default for Appium
            elif scenario.framework == TestFramework.SELENIUM:
                platform = "web"
            elif scenario.framework == TestFramework.ESPRESSO:
                platform = "android"
            elif scenario.framework == TestFramework.XCUITEST:
                platform = "ios"
            else:
                platform = "web"  # Fallback
            
            # Check platform availability and adjust if needed
            if not self.available_platforms[platform]["available"]:
                platform = self._find_alternative_platform(platform, scenario)
            
            assignments[scenario.scenario_id] = platform
        
        return assignments
    
    async def _execute_parallel(self, scenarios: List[TestScenario], 
                              platform_assignments: Dict[str, str], 
                              max_parallel: int, timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios in parallel."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single_scenario(scenario: TestScenario, platform: str):
            async with semaphore:
                execution_id = f"exec-{scenario.scenario_id}-{datetime.now().strftime('%H%M%S')}"
                
                try:
                    result = await self.test_runner.run_test(scenario, execution_id)
                    
                    # Update platform performance metrics
                    self._update_platform_performance(platform, result)
                    
                    return {
                        "scenario_id": scenario.scenario_id,
                        "execution_id": execution_id,
                        "platform": platform,
                        "result": result,
                        "status": result.status,
                        "duration": result.duration,
                        "healing_attempts": result.healing_attempts
                    }
                except Exception as e:
                    logger.error(f"Error executing scenario {scenario.scenario_id}: {e}")
                    return {
                        "scenario_id": scenario.scenario_id,
                        "execution_id": execution_id,
                        "platform": platform,
                        "status": "error",
                        "error": str(e)
                    }
        
        # Execute all scenarios in parallel
        tasks = [
            execute_single_scenario(scenario, platform_assignments[scenario.scenario_id])
            for scenario in scenarios
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task execution failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_sequential(self, scenarios: List[TestScenario], 
                                platform_assignments: Dict[str, str], 
                                max_parallel: int, timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios sequentially."""
        results = []
        
        for scenario in scenarios:
            platform = platform_assignments[scenario.scenario_id]
            execution_id = f"exec-{scenario.scenario_id}-{datetime.now().strftime('%H%M%S')}"
            
            try:
                result = await self.test_runner.run_test(scenario, execution_id)
                
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "execution_id": execution_id,
                    "platform": platform,
                    "result": result,
                    "status": result.status,
                    "duration": result.duration
                })
                
                # Stop on first failure for sequential execution
                if result.status == "failed":
                    logger.warning(f"Stopping sequential execution due to failure in {scenario.scenario_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing scenario {scenario.scenario_id}: {e}")
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "execution_id": execution_id,
                    "platform": platform,
                    "status": "error",
                    "error": str(e)
                })
                break
        
        return results
    
    async def _execute_adaptive(self, scenarios: List[TestScenario], 
                              platform_assignments: Dict[str, str], 
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
    
    async def _execute_distributed(self, scenarios: List[TestScenario], 
                                 platform_assignments: Dict[str, str], 
                                 max_parallel: int, timeout: int) -> List[Dict[str, Any]]:
        """Execute scenarios using distributed execution (Kubernetes/Temporal)."""
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
                "error_message": execution.error_message
            })
        
        return results
    
    async def _heal_failing_tests(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Heal failing tests using vision-based healing."""
        failed_tests = task_data.get("failed_tests", [])
        healing_strategy = task_data.get("healing_strategy", "vision_based")
        
        healed_results = []
        
        for failed_test in failed_tests:
            scenario_id = failed_test.get("scenario_id")
            error_message = failed_test.get("error_message")
            screenshots = failed_test.get("screenshots", [])
            
            if screenshots and error_message:
                # Attempt vision-based healing
                healing_result = await self.vision_healer.heal_test(
                    failed_locator=error_message,
                    screenshot_path=screenshots[0],
                    context={"scenario_id": scenario_id}
                )
                
                if healing_result.success:
                    # Retry test with healed locator
                    retry_result = await self._retry_with_healed_locator(
                        failed_test, healing_result.new_locator
                    )
                    healed_results.append({
                        "scenario_id": scenario_id,
                        "healing_success": True,
                        "new_locator": healing_result.new_locator,
                        "retry_result": retry_result
                    })
                else:
                    healed_results.append({
                        "scenario_id": scenario_id,
                        "healing_success": False,
                        "reason": healing_result.error_message
                    })
        
        return {
            "healed_tests": healed_results,
            "healing_strategy": healing_strategy,
            "successful_healings": len([r for r in healed_results if r["healing_success"]]),
            "total_attempts": len(healed_results)
        }
    
    async def _retry_failed_scenarios(self, failed_scenarios: List[Dict], 
                                    original_scenarios: List[TestScenario]) -> List[Dict[str, Any]]:
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
                        "result": result,
                        "status": result.status,
                        "retry_attempt": True
                    })
                    
                except Exception as e:
                    retry_results.append({
                        "scenario_id": scenario_id,
                        "execution_id": execution_id,
                        "platform": alternative_platform,
                        "status": "error",
                        "error": str(e),
                        "retry_attempt": True
                    })
        
        return retry_results
    
    def _analyze_scenario_requirements(self, scenario: TestScenario) -> Dict[str, Any]:
        """Analyze scenario requirements for platform selection."""
        requirements = {
            "framework": scenario.framework.value,
            "test_type": scenario.test_type.value,
            "platform_hints": [],
            "device_requirements": [],
            "browser_requirements": [],
            "api_requirements": []
        }
        
        # Extract requirements from tags and metadata
        for tag in scenario.tags:
            if tag in ["mobile", "android", "ios"]:
                requirements["platform_hints"].append(tag)
            elif tag in ["desktop", "web"]:
                requirements["platform_hints"].append(tag)
            elif tag in ["chrome", "firefox", "safari"]:
                requirements["browser_requirements"].append(tag)
        
        # Extract from metadata
        if scenario.metadata:
            if "platform" in scenario.metadata:
                requirements["platform_hints"].append(scenario.metadata["platform"])
            if "device" in scenario.metadata:
                requirements["device_requirements"].append(scenario.metadata["device"])
        
        return requirements
    
    def _calculate_platform_score(self, platform: str, requirements: Dict[str, Any], 
                                scenario: TestScenario) -> float:
        """Calculate platform suitability score."""
        score = 0.0
        
        # Framework compatibility
        if scenario.framework == TestFramework.APPIUM and platform in ["android", "ios"]:
            score += 0.4
        elif scenario.framework == TestFramework.SELENIUM and platform == "web":
            score += 0.4
        elif scenario.framework == TestFramework.ESPRESSO and platform == "android":
            score += 0.4
        elif scenario.framework == TestFramework.XCUITEST and platform == "ios":
            score += 0.4
        
        # Platform hints matching
        for hint in requirements["platform_hints"]:
            if hint in platform:
                score += 0.2
        
        # Historical performance
        if platform in self.platform_performance:
            perf_score = self.platform_performance[platform].get("success_rate", 0.5)
            score += perf_score * 0.3
        
        # Availability bonus
        if self.available_platforms[platform]["available"]:
            score += 0.1
        
        return min(score, 1.0)
    
    def _select_specific_device(self, platform: str, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Select specific device/browser for platform."""
        platform_info = self.available_platforms[platform]
        
        if platform == "web":
            # Select browser
            preferred_browsers = requirements.get("browser_requirements", [])
            available_browsers = platform_info["available"]
            
            if preferred_browsers:
                selected_browser = next(
                    (b for b in preferred_browsers if b in available_browsers),
                    available_browsers[0]
                )
            else:
                selected_browser = available_browsers[0]
            
            return {
                "type": "browser",
                "name": selected_browser,
                "platform": platform
            }
        
        else:
            # Select device
            available_devices = platform_info["available"]
            selected_device = available_devices[0] if available_devices else "default"
            
            return {
                "type": "device",
                "name": selected_device,
                "platform": platform
            }
    
    def _find_alternative_platform(self, preferred_platform: str, scenario: TestScenario) -> str:
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
    
    def _update_platform_performance(self, platform: str, result: Any):
        """Update platform performance metrics."""
        if platform not in self.platform_performance:
            self.platform_performance[platform] = {
                "total_executions": 0,
                "successful_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0
            }
        
        perf = self.platform_performance[platform]
        perf["total_executions"] += 1
        
        if result.status == "passed":
            perf["successful_executions"] += 1
        
        perf["success_rate"] = perf["successful_executions"] / perf["total_executions"]
        
        if result.duration:
            perf["average_duration"] = (
                (perf["average_duration"] * (perf["total_executions"] - 1) + result.duration) /
                perf["total_executions"]
            )
    
    def _analyze_execution_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution results and provide summary."""
        total_scenarios = len(results)
        successful = len([r for r in results if r["status"] == "passed"])
        failed = len([r for r in results if r["status"] == "failed"])
        errors = len([r for r in results if r["status"] == "error"])
        
        total_duration = sum(r.get("duration", 0) for r in results if r.get("duration"))
        average_duration = total_duration / total_scenarios if total_scenarios > 0 else 0
        
        return {
            "total_scenarios": total_scenarios,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "success_rate": successful / total_scenarios if total_scenarios > 0 else 0,
            "total_execution_time": total_duration,
            "average_duration": average_duration,
            "platform_distribution": self._get_platform_distribution(results)
        }
    
    def _get_platform_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get platform distribution from results."""
        distribution = {}
        for result in results:
            platform = result.get("platform", "unknown")
            distribution[platform] = distribution.get(platform, 0) + 1
        return distribution
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        # This would integrate with actual system monitoring
        return {
            "cpu_usage": random.uniform(30, 90),
            "memory_usage": random.uniform(40, 80),
            "disk_usage": random.uniform(20, 70),
            "network_usage": random.uniform(10, 50)
        }
    
    def _get_adaptation_reason(self, metrics: Dict[str, float]) -> str:
        """Get reason for execution strategy adaptation."""
        if metrics["cpu_usage"] > 80:
            return "High CPU usage detected"
        elif metrics["memory_usage"] > 85:
            return "High memory usage detected"
        else:
            return "Optimal conditions for parallel execution"
    
    async def _retry_with_healed_locator(self, failed_test: Dict[str, Any], new_locator: str) -> Dict[str, Any]:
        """Retry test with healed locator."""
        # This would implement the actual retry logic
        return {
            "status": "retried",
            "healed_locator": new_locator,
            "retry_success": True
        }
    
    async def _manage_devices(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage device availability and allocation."""
        action = task_data.get("action", "status")
        
        if action == "status":
            return {
                "platform_status": {
                    platform: {
                        "available": len(info["available"]),
                        "busy": len(info["busy"]),
                        "total": len(info["devices"]) if "devices" in info else len(info.get("browsers", []))
                    }
                    for platform, info in self.available_platforms.items()
                }
            }
        
        elif action == "allocate":
            platform = task_data.get("platform")
            device = task_data.get("device")
            
            if platform in self.available_platforms:
                platform_info = self.available_platforms[platform]
                if device in platform_info["available"]:
                    platform_info["available"].remove(device)
                    platform_info["busy"].append(device)
                    
                    return {"status": "allocated", "device": device, "platform": platform}
        
        return {"status": "failed", "reason": "Invalid action or device unavailable"}
    
    async def _optimize_execution_performance(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution performance based on historical data."""
        optimization_target = task_data.get("target", "execution_time")
        
        # Analyze historical performance
        if optimization_target == "execution_time":
            # Find fastest platforms
            platform_times = {
                platform: perf.get("average_duration", 0)
                for platform, perf in self.platform_performance.items()
            }
            
            fastest_platforms = sorted(platform_times.items(), key=lambda x: x[1])[:3]
            
            return {
                "optimization_target": optimization_target,
                "fastest_platforms": fastest_platforms,
                "recommendations": [
                    f"Prioritize {platform} for faster execution" 
                    for platform, _ in fastest_platforms
                ]
            }
        
        return {"optimization_target": optimization_target, "status": "completed"}
