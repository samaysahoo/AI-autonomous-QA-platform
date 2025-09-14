"""Test runner for executing individual test scenarios."""

import logging
import asyncio
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import json

from src.test_generation.test_scenario import TestScenario
from .vision_healer import VisionHealer, HealingResult

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a test execution."""
    execution_id: str
    scenario_id: str
    status: str  # passed, failed, skipped, error
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    screenshots: List[str] = None
    logs: List[str] = None
    healing_attempts: int = 0
    
    def __post_init__(self):
        if self.screenshots is None:
            self.screenshots = []
        if self.logs is None:
            self.logs = []


class TestRunner:
    """Executes individual test scenarios."""
    
    def __init__(self):
        self.vision_healer = VisionHealer()
    
    async def run_test(self, scenario: TestScenario, execution_id: str) -> TestResult:
        """Execute a test scenario and return the result."""
        
        result = TestResult(
            execution_id=execution_id,
            scenario_id=scenario.scenario_id,
            status="running",
            start_time=datetime.now()
        )
        
        try:
            # Generate test code
            test_code = await self._generate_test_code(scenario)
            
            # Execute the test
            execution_result = await self._execute_test_code(test_code, scenario)
            
            result.status = execution_result["status"]
            result.error_message = execution_result.get("error_message")
            result.logs = execution_result.get("logs", [])
            result.screenshots = execution_result.get("screenshots", [])
            
            # If test failed, attempt healing
            if result.status == "failed" and result.screenshots:
                healing_result = await self._attempt_healing(
                    scenario, result, execution_result.get("failed_locator")
                )
                
                if healing_result.success:
                    result.healing_attempts += 1
                    # Retry test with healed locator
                    retry_result = await self._retry_with_healed_locator(
                        scenario, healing_result.new_locator
                    )
                    
                    if retry_result["status"] == "passed":
                        result.status = "passed"
                        result.error_message = None
                    else:
                        result.status = "failed"
                        result.error_message = "Healing failed to resolve test failure"
                else:
                    result.status = "failed"
            
        except Exception as e:
            logger.error(f"Error executing test: {e}")
            result.status = "error"
            result.error_message = str(e)
        
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    async def _generate_test_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate executable test code for the scenario."""
        
        from src.test_generation.code_generator import CodeGenerator
        
        code_generator = CodeGenerator()
        return code_generator.generate_test_code(scenario)
    
    async def _execute_test_code(self, test_code: Dict[str, str], 
                               scenario: TestScenario) -> Dict[str, Any]:
        """Execute the generated test code."""
        
        try:
            # Create temporary file for test code
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f"_{test_code['filename']}", 
                delete=False
            ) as f:
                f.write(test_code["code"])
                temp_file = f.name
            
            # Execute based on framework
            if test_code["framework"] == "appium":
                return await self._execute_appium_test(temp_file, scenario)
            elif test_code["framework"] == "selenium":
                return await self._execute_selenium_test(temp_file, scenario)
            elif test_code["framework"] == "espresso":
                return await self._execute_espresso_test(temp_file, scenario)
            elif test_code["framework"] == "xcuitest":
                return await self._execute_xcuitest_test(temp_file, scenario)
            else:
                return {
                    "status": "error",
                    "error_message": f"Unsupported framework: {test_code['framework']}"
                }
        
        except Exception as e:
            logger.error(f"Error executing test code: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
        
        finally:
            # Clean up temporary file
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    async def _execute_appium_test(self, test_file: str, 
                                 scenario: TestScenario) -> Dict[str, Any]:
        """Execute Appium test."""
        
        try:
            # Run pytest with the test file
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=scenario.get_estimated_duration() + 60
            )
            
            # Parse result
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "status": status,
                "error_message": result.stderr if status == "failed" else None,
                "logs": result.stdout.split('\n'),
                "screenshots": self._collect_screenshots()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error_message": "Test execution timeout",
                "logs": [],
                "screenshots": []
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }
    
    async def _execute_selenium_test(self, test_file: str, 
                                   scenario: TestScenario) -> Dict[str, Any]:
        """Execute Selenium test."""
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=scenario.get_estimated_duration() + 60
            )
            
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "status": status,
                "error_message": result.stderr if status == "failed" else None,
                "logs": result.stdout.split('\n'),
                "screenshots": self._collect_screenshots()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }
    
    async def _execute_espresso_test(self, test_file: str, 
                                   scenario: TestScenario) -> Dict[str, Any]:
        """Execute Espresso test."""
        
        try:
            # For Espresso, we would typically use Gradle
            result = subprocess.run(
                ["./gradlew", "connectedAndroidTest"],
                capture_output=True,
                text=True,
                timeout=scenario.get_estimated_duration() + 60
            )
            
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "status": status,
                "error_message": result.stderr if status == "failed" else None,
                "logs": result.stdout.split('\n'),
                "screenshots": []
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }
    
    async def _execute_xcuitest_test(self, test_file: str, 
                                   scenario: TestScenario) -> Dict[str, Any]:
        """Execute XCUITest test."""
        
        try:
            # For XCUITest, we would use xcodebuild
            result = subprocess.run(
                ["xcodebuild", "test", "-scheme", "YourApp", "-destination", "platform=iOS Simulator"],
                capture_output=True,
                text=True,
                timeout=scenario.get_estimated_duration() + 60
            )
            
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "status": status,
                "error_message": result.stderr if status == "failed" else None,
                "logs": result.stdout.split('\n'),
                "screenshots": []
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }
    
    def _collect_screenshots(self) -> List[str]:
        """Collect screenshots from test execution."""
        
        screenshots = []
        
        # Look for screenshots in common directories
        screenshot_dirs = [
            "./screenshots",
            "./test_results/screenshots",
            "/tmp/screenshots"
        ]
        
        for directory in screenshot_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        screenshots.append(os.path.join(directory, file))
        
        return screenshots
    
    async def _attempt_healing(self, scenario: TestScenario, 
                             result: TestResult, 
                             failed_locator: Optional[str]) -> HealingResult:
        """Attempt to heal a failed test using vision-based healing."""
        
        if not result.screenshots or not failed_locator:
            return HealingResult(
                success=False,
                error_message="No screenshots or failed locator available"
            )
        
        # Use the most recent screenshot
        latest_screenshot = result.screenshots[-1]
        
        # Create context from scenario and result
        context = {
            "scenario": scenario.to_dict(),
            "failed_locator": failed_locator,
            "error_message": result.error_message,
            "logs": result.logs
        }
        
        # Attempt healing
        healing_result = await self.vision_healer.heal_test(
            failed_locator=failed_locator,
            screenshot_path=latest_screenshot,
            context=context
        )
        
        return healing_result
    
    async def _retry_with_healed_locator(self, scenario: TestScenario, 
                                       new_locator: str) -> Dict[str, Any]:
        """Retry test execution with the healed locator."""
        
        try:
            # Update scenario with new locator
            # This is a simplified implementation
            # In practice, you'd need to update the specific step that failed
            
            # Generate new test code with healed locator
            test_code = await self._generate_test_code(scenario)
            
            # Execute the test again
            return await self._execute_test_code(test_code, scenario)
            
        except Exception as e:
            logger.error(f"Error retrying test with healed locator: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    async def run_test_suite(self, scenarios: List[TestScenario]) -> List[TestResult]:
        """Run multiple test scenarios and return results."""
        
        results = []
        
        # Run tests in parallel (with some concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tests
        
        async def run_single_test(scenario: TestScenario) -> TestResult:
            async with semaphore:
                execution_id = f"exec_{scenario.scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                return await self.run_test(scenario, execution_id)
        
        # Execute all tests
        tasks = [run_single_test(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to TestResult objects
        filtered_results = []
        for result in results:
            if isinstance(result, TestResult):
                filtered_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Test execution failed with exception: {result}")
                # Create error result
                error_result = TestResult(
                    execution_id="error",
                    scenario_id="unknown",
                    status="error",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(result)
                )
                filtered_results.append(error_result)
        
        return filtered_results
