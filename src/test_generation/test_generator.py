"""AI-powered test case generator using LLM + RAG."""

import logging
import uuid
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

from config.settings import get_settings
from src.data_ingestion.vector_store import VectorStoreManager
from .test_scenario import TestScenario, TestType, TestFramework, TestPriority, TestStep

logger = logging.getLogger(__name__)


class TestCaseGenerator:
    """Generates test cases using LLM with RAG context."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.vector_store = VectorStoreManager()
    
    def generate_test_scenarios(
        self, 
        query: str, 
        test_type: TestType = TestType.E2E,
        framework: TestFramework = TestFramework.APPIUM,
        max_scenarios: int = 5
    ) -> List[TestScenario]:
        """Generate test scenarios based on a query using RAG."""
        
        # Retrieve relevant context from vector store
        context_docs = self.vector_store.search_similar(
            query=query,
            n_results=10,
            metadata_filter={"source": {"$in": ["jira", "analytics"]}}
        )
        
        # Prepare context for LLM
        context = self._prepare_context(context_docs)
        
        # Generate test scenarios using LLM
        scenarios = self._generate_with_llm(
            query=query,
            context=context,
            test_type=test_type,
            framework=framework,
            max_scenarios=max_scenarios
        )
        
        logger.info(f"Generated {len(scenarios)} test scenarios for query: {query}")
        return scenarios
    
    def generate_from_bug_report(self, bug_description: str) -> List[TestScenario]:
        """Generate test scenarios specifically for bug reproduction and validation."""
        
        # Search for similar bugs and their fixes
        context_docs = self.vector_store.search_similar(
            query=f"bug fix validation test {bug_description}",
            n_results=8,
            metadata_filter={"metadata.error_type": {"$exists": True}}
        )
        
        context = self._prepare_context(context_docs)
        
        # Generate regression tests
        scenarios = self._generate_regression_tests(
            bug_description=bug_description,
            context=context
        )
        
        return scenarios
    
    def generate_from_user_story(self, story_description: str) -> List[TestScenario]:
        """Generate test scenarios from user story acceptance criteria."""
        
        # Search for similar user stories and their implementations
        context_docs = self.vector_store.search_similar(
            query=f"user story acceptance criteria {story_description}",
            n_results=10,
            metadata_filter={"metadata.type": {"$in": ["Story", "User Story"]}}
        )
        
        context = self._prepare_context(context_docs)
        
        # Generate acceptance test scenarios
        scenarios = self._generate_acceptance_tests(
            story_description=story_description,
            context=context
        )
        
        return scenarios
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents."""
        if not docs:
            return "No relevant context found."
        
        context_parts = []
        for doc in docs[:5]:  # Limit to top 5 most relevant docs
            context_parts.append(f"""
            Document ID: {doc['id']}
            Content: {doc['content'][:500]}...
            Source: {doc['metadata'].get('source', 'unknown')}
            Type: {doc['metadata'].get('type', 'unknown')}
            """)
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(
        self,
        query: str,
        context: str,
        test_type: TestType,
        framework: TestFramework,
        max_scenarios: int
    ) -> List[TestScenario]:
        """Generate test scenarios using OpenAI LLM."""
        
        system_prompt = f"""
        You are an expert test automation engineer. Generate comprehensive test scenarios based on the provided context and query.
        
        Test Framework: {framework.value}
        Test Type: {test_type.value}
        
        Generate up to {max_scenarios} test scenarios that:
        1. Cover the main functionality described
        2. Include edge cases and error conditions
        3. Are specific and actionable
        4. Include proper test steps with expected results
        5. Consider the context from similar issues/stories
        
        Return the scenarios in JSON format with this structure:
        {{
            "scenarios": [
                {{
                    "title": "Test scenario title",
                    "description": "Detailed description",
                    "priority": "critical|high|medium|low",
                    "steps": [
                        {{
                            "step_id": "step_1",
                            "description": "Step description",
                            "action": "Action to perform",
                            "expected_result": "Expected outcome",
                            "locator": "UI element locator (if applicable)",
                            "input_data": {{"key": "value"}},
                            "wait_time": 5
                        }}
                    ],
                    "prerequisites": ["prerequisite1", "prerequisite2"],
                    "tags": ["tag1", "tag2"],
                    "confidence_score": 0.85
                }}
            ]
        }}
        """
        
        user_prompt = f"""
        Query: {query}
        
        Context from similar documents:
        {context}
        
        Please generate test scenarios based on this information.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse LLM response
            scenarios_data = self._parse_llm_response(response.choices[0].message.content)
            
            # Convert to TestScenario objects
            scenarios = []
            for scenario_data in scenarios_data.get("scenarios", []):
                scenario = self._create_test_scenario(
                    scenario_data, test_type, framework, context
                )
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating test scenarios with LLM: {e}")
            return []
    
    def _generate_regression_tests(
        self,
        bug_description: str,
        context: str
    ) -> List[TestScenario]:
        """Generate regression tests for bug fixes."""
        
        system_prompt = """
        You are an expert QA engineer. Generate regression test scenarios to validate bug fixes.
        
        Focus on:
        1. Reproducing the original bug
        2. Validating the fix works
        3. Testing related functionality that might be affected
        4. Edge cases around the bug area
        
        Return scenarios in the same JSON format as before.
        """
        
        user_prompt = f"""
        Bug Description: {bug_description}
        
        Context from similar bugs and fixes:
        {context}
        
        Generate regression test scenarios.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            
            scenarios_data = self._parse_llm_response(response.choices[0].message.content)
            
            scenarios = []
            for scenario_data in scenarios_data.get("scenarios", []):
                scenario = self._create_test_scenario(
                    scenario_data, TestType.E2E, TestFramework.APPIUM, context
                )
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating regression tests: {e}")
            return []
    
    def _generate_acceptance_tests(
        self,
        story_description: str,
        context: str
    ) -> List[TestScenario]:
        """Generate acceptance test scenarios from user stories."""
        
        system_prompt = """
        You are an expert QA engineer. Generate acceptance test scenarios from user stories.
        
        Focus on:
        1. Validating acceptance criteria
        2. Testing user workflows end-to-end
        3. Ensuring business requirements are met
        4. Happy path and alternative flows
        
        Return scenarios in the same JSON format as before.
        """
        
        user_prompt = f"""
        User Story: {story_description}
        
        Context from similar stories and implementations:
        {context}
        
        Generate acceptance test scenarios.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=3500
            )
            
            scenarios_data = self._parse_llm_response(response.choices[0].message.content)
            
            scenarios = []
            for scenario_data in scenarios_data.get("scenarios", []):
                scenario = self._create_test_scenario(
                    scenario_data, TestType.E2E, TestFramework.APPIUM, context
                )
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating acceptance tests: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response JSON."""
        import json
        import re
        
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {response}")
            return {"scenarios": []}
    
    def _create_test_scenario(
        self,
        scenario_data: Dict[str, Any],
        test_type: TestType,
        framework: TestFramework,
        context: str
    ) -> TestScenario:
        """Create TestScenario object from LLM response data."""
        
        scenario_id = str(uuid.uuid4())
        
        # Create test steps
        steps = []
        for step_data in scenario_data.get("steps", []):
            step = TestStep(
                step_id=step_data.get("step_id", f"step_{len(steps) + 1}"),
                description=step_data.get("description", ""),
                action=step_data.get("action", ""),
                expected_result=step_data.get("expected_result", ""),
                locator=step_data.get("locator"),
                input_data=step_data.get("input_data"),
                wait_time=step_data.get("wait_time")
            )
            steps.append(step)
        
        # Determine priority
        priority_str = scenario_data.get("priority", "medium").lower()
        priority_map = {
            "critical": TestPriority.CRITICAL,
            "high": TestPriority.HIGH,
            "medium": TestPriority.MEDIUM,
            "low": TestPriority.LOW
        }
        priority = priority_map.get(priority_str, TestPriority.MEDIUM)
        
        scenario = TestScenario(
            scenario_id=scenario_id,
            title=scenario_data.get("title", "Generated Test Scenario"),
            description=scenario_data.get("description", ""),
            test_type=test_type,
            framework=framework,
            priority=priority,
            steps=steps,
            prerequisites=scenario_data.get("prerequisites", []),
            tags=scenario_data.get("tags", []),
            confidence_score=scenario_data.get("confidence_score"),
            metadata={
                "generated_from_context": context[:200] + "..." if len(context) > 200 else context
            }
        )
        
        return scenario
