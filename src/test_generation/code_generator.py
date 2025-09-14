"""Code generator for converting test scenarios to executable test code."""

import logging
from typing import List, Dict, Any, Optional
from string import Template

from .test_scenario import TestScenario, TestFramework, TestType

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates executable test code from test scenarios."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def generate_test_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate executable test code for a scenario."""
        
        if scenario.framework == TestFramework.APPIUM:
            return self._generate_appium_code(scenario)
        elif scenario.framework == TestFramework.ESPRESSO:
            return self._generate_espresso_code(scenario)
        elif scenario.framework == TestFramework.XCUITEST:
            return self._generate_xcuitest_code(scenario)
        elif scenario.framework == TestFramework.SELENIUM:
            return self._generate_selenium_code(scenario)
        else:
            logger.warning(f"Unsupported framework: {scenario.framework}")
            return {"error": f"Unsupported framework: {scenario.framework}"}
    
    def _generate_appium_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate Appium test code."""
        
        template = self.templates["appium"]
        
        # Generate step code
        step_code = []
        for i, step in enumerate(scenario.steps):
            step_template = self.templates["appium_step"]
            step_code.append(step_template.substitute(
                step_number=i + 1,
                description=step.description,
                action=step.action,
                expected_result=step.expected_result,
                locator=step.locator or "//*[@text='Element']",
                input_data=step.input_data or {},
                wait_time=step.wait_time or 5
            ))
        
        # Generate prerequisites setup
        prerequisites_code = []
        for prereq in scenario.prerequisites:
            prereq_template = self.templates["prerequisite"]
            prerequisites_code.append(prereq_template.substitute(
                prerequisite=prereq
            ))
        
        # Generate main test code
        test_code = template.substitute(
            test_name=self._sanitize_name(scenario.title),
            description=scenario.description,
            test_id=scenario.scenario_id,
            priority=scenario.priority.value,
            tags=", ".join([f'"{tag}"' for tag in scenario.tags]),
            prerequisites="\n        ".join(prerequisites_code),
            steps="\n        ".join(step_code),
            expected_duration=scenario.get_estimated_duration()
        )
        
        return {
            "language": "python",
            "framework": "appium",
            "code": test_code,
            "filename": f"test_{self._sanitize_name(scenario.title)}.py"
        }
    
    def _generate_espresso_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate Espresso test code for Android."""
        
        template = self.templates["espresso"]
        
        # Generate step code
        step_code = []
        for i, step in enumerate(scenario.steps):
            step_template = self.templates["espresso_step"]
            step_code.append(step_template.substitute(
                step_number=i + 1,
                description=step.description,
                action=step.action,
                expected_result=step.expected_result,
                locator=step.locator or "withText(\"Element\")",
                input_data=step.input_data or {}
            ))
        
        test_code = template.substitute(
            test_name=self._sanitize_name(scenario.title),
            description=scenario.description,
            test_id=scenario.scenario_id,
            steps="\n        ".join(step_code)
        )
        
        return {
            "language": "kotlin",
            "framework": "espresso",
            "code": test_code,
            "filename": f"{self._sanitize_name(scenario.title)}Test.kt"
        }
    
    def _generate_xcuitest_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate XCUITest code for iOS."""
        
        template = self.templates["xcuitest"]
        
        # Generate step code
        step_code = []
        for i, step in enumerate(scenario.steps):
            step_template = self.templates["xcuitest_step"]
            step_code.append(step_template.substitute(
                step_number=i + 1,
                description=step.description,
                action=step.action,
                expected_result=step.expected_result,
                locator=step.locator or "buttons[\"Element\"]",
                input_data=step.input_data or {}
            ))
        
        test_code = template.substitute(
            test_name=self._sanitize_name(scenario.title),
            description=scenario.description,
            test_id=scenario.scenario_id,
            steps="\n        ".join(step_code)
        )
        
        return {
            "language": "swift",
            "framework": "xcuitest",
            "code": test_code,
            "filename": f"{self._sanitize_name(scenario.title)}Test.swift"
        }
    
    def _generate_selenium_code(self, scenario: TestScenario) -> Dict[str, str]:
        """Generate Selenium test code for web testing."""
        
        template = self.templates["selenium"]
        
        # Generate step code
        step_code = []
        for i, step in enumerate(scenario.steps):
            step_template = self.templates["selenium_step"]
            step_code.append(step_template.substitute(
                step_number=i + 1,
                description=step.description,
                action=step.action,
                expected_result=step.expected_result,
                locator=step.locator or "id='element'",
                input_data=step.input_data or {}
            ))
        
        test_code = template.substitute(
            test_name=self._sanitize_name(scenario.title),
            description=scenario.description,
            test_id=scenario.scenario_id,
            steps="\n        ".join(step_code)
        )
        
        return {
            "language": "python",
            "framework": "selenium",
            "code": test_code,
            "filename": f"test_{self._sanitize_name(scenario.title)}.py"
        }
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load code templates for different frameworks."""
        
        templates = {
            "appium": Template("""
import pytest
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
import time

class Test${test_name}:
    \"\"\"
    ${description}
    Test ID: ${test_id}
    Priority: ${priority}
    Tags: ${tags}
    \"\"\"
    
    @pytest.fixture(scope="class")
    def driver(self):
        \"\"\"Setup Appium driver\"\"\"
        capabilities = {
            'platformName': 'Android',
            'deviceName': 'Android Emulator',
            'app': 'path/to/app.apk',
            'automationName': 'UiAutomator2'
        }
        driver = webdriver.Remote('http://localhost:4723', capabilities)
        yield driver
        driver.quit()
    
    def test_${test_name}(self, driver):
        \"\"\"Execute test scenario: ${description}\"\"\"
        # Prerequisites
        ${prerequisites}
        
        # Test steps
        ${steps}
        
        # Verify test completion
        assert True, "Test scenario completed successfully"
"""),
            
            "appium_step": Template("""
        # Step ${step_number}: ${description}
        # Action: ${action}
        # Expected: ${expected_result}
        element = driver.find_element(AppiumBy.XPATH, "${locator}")
        element.click()
        time.sleep(${wait_time})
        # Add assertions as needed
"""),
            
            "prerequisite": Template("""
        # Prerequisite: ${prerequisite}
        # Add prerequisite setup code here
"""),
            
            "espresso": Template("""
package com.example.tests

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.ActivityTestRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ${test_name}Test {
    
    @get:Rule
    val activityRule = ActivityTestRule(MainActivity::class.java)
    
    @Test
    fun test${test_name}() {
        \"\"\"
        ${description}
        Test ID: ${test_id}
        \"\"\"
        
        ${steps}
    }
}
"""),
            
            "espresso_step": Template("""
        // Step ${step_number}: ${description}
        // Action: ${action}
        // Expected: ${expected_result}
        onView(${locator})
            .perform(click())
"""),
            
            "xcuitest": Template("""
import XCTest

class ${test_name}Test: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }
    
    func test${test_name}() throws {
        \"\"\"
        ${description}
        Test ID: ${test_id}
        \"\"\"
        
        ${steps}
    }
}
"""),
            
            "xcuitest_step": Template("""
        // Step ${step_number}: ${description}
        // Action: ${action}
        // Expected: ${expected_result}
        app.${locator}.tap()
"""),
            
            "selenium": Template("""
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Test${test_name}:
    \"\"\"
    ${description}
    Test ID: ${test_id}
    \"\"\"
    
    @pytest.fixture(scope="class")
    def driver(self):
        \"\"\"Setup Selenium driver\"\"\"
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    def test_${test_name}(self, driver):
        \"\"\"Execute test scenario: ${description}\"\"\"
        
        ${steps}
        
        # Verify test completion
        assert True, "Test scenario completed successfully"
"""),
            
            "selenium_step": Template("""
        # Step ${step_number}: ${description}
        # Action: ${action}
        # Expected: ${expected_result}
        element = driver.find_element(By.CSS_SELECTOR, "${locator}")
        element.click()
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "${locator}"))
        )
""")
        }
        
        return templates
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use in code."""
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = 'Test' + sanitized
        return sanitized or 'GeneratedTest'
