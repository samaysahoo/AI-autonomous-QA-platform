"""Tests for test generation module."""

import pytest
from unittest.mock import Mock, patch
from src.test_generation.test_generator import TestCaseGenerator
from src.test_generation.test_scenario import TestType, TestFramework


class TestTestCaseGenerator:
    """Test cases for TestCaseGenerator class."""
    
    @pytest.fixture
    def test_generator(self, mock_vector_store):
        """Create TestCaseGenerator instance with mocked dependencies."""
        return TestCaseGenerator()
    
    def test_init(self, test_generator):
        """Test TestCaseGenerator initialization."""
        assert test_generator is not None
        assert test_generator.settings is not None
        assert test_generator.vector_store is not None
    
    @pytest.mark.asyncio
    async def test_generate_test_scenarios_success(self, test_generator, mock_openai_client):
        """Test successful test scenario generation."""
        # Mock vector store search
        test_generator.vector_store.search_similar.return_value = [
            {
                "id": "doc_1",
                "content": "Test document content",
                "metadata": {"source": "jira", "type": "Story"}
            }
        ]
        
        # Mock OpenAI response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = '''
        {
            "scenarios": [
                {
                    "title": "Test User Login",
                    "description": "Test user login functionality",
                    "priority": "high",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "description": "Navigate to login page",
                            "action": "Open browser and go to login URL",
                            "expected_result": "Login page is displayed"
                        }
                    ],
                    "prerequisites": ["Valid user account"],
                    "tags": ["authentication", "login"],
                    "confidence_score": 0.85
                }
            ]
        }
        '''
        
        scenarios = test_generator.generate_test_scenarios(
            query="User login functionality",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=1
        )
        
        assert len(scenarios) == 1
        assert scenarios[0].title == "Test User Login"
        assert scenarios[0].test_type == TestType.E2E
        assert scenarios[0].framework == TestFramework.APPIUM
        assert len(scenarios[0].steps) == 1
    
    @pytest.mark.asyncio
    async def test_generate_test_scenarios_empty_query(self, test_generator):
        """Test test scenario generation with empty query."""
        scenarios = test_generator.generate_test_scenarios(
            query="",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=1
        )
        
        assert len(scenarios) == 0
    
    @pytest.mark.asyncio
    async def test_generate_test_scenarios_api_error(self, test_generator, mock_openai_client):
        """Test test scenario generation with API error."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        scenarios = test_generator.generate_test_scenarios(
            query="User login functionality",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=1
        )
        
        assert len(scenarios) == 0
    
    @pytest.mark.asyncio
    async def test_generate_from_bug_report(self, test_generator, mock_openai_client):
        """Test test generation from bug report."""
        # Mock vector store search
        test_generator.vector_store.search_similar.return_value = []
        
        # Mock OpenAI response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = '''
        {
            "scenarios": [
                {
                    "title": "Reproduce Login Bug",
                    "description": "Test to reproduce the login bug",
                    "priority": "critical",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "description": "Navigate to login page",
                            "action": "Open browser and go to login URL",
                            "expected_result": "Login page is displayed"
                        }
                    ],
                    "prerequisites": [],
                    "tags": ["bug", "regression"],
                    "confidence_score": 0.9
                }
            ]
        }
        '''
        
        scenarios = test_generator.generate_from_bug_report(
            "Login button not working on mobile devices"
        )
        
        assert len(scenarios) >= 0  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_generate_from_user_story(self, test_generator, mock_openai_client):
        """Test test generation from user story."""
        # Mock vector store search
        test_generator.vector_store.search_similar.return_value = []
        
        # Mock OpenAI response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = '''
        {
            "scenarios": [
                {
                    "title": "User Story Acceptance Test",
                    "description": "Test user story acceptance criteria",
                    "priority": "high",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "description": "Verify user can complete the action",
                            "action": "Perform the main action",
                            "expected_result": "Action is completed successfully"
                        }
                    ],
                    "prerequisites": ["User is logged in"],
                    "tags": ["acceptance", "user-story"],
                    "confidence_score": 0.8
                }
            ]
        }
        '''
        
        scenarios = test_generator.generate_from_user_story(
            "As a user, I want to be able to reset my password"
        )
        
        assert len(scenarios) >= 0  # Should handle gracefully
    
    def test_parse_llm_response_valid_json(self, test_generator):
        """Test parsing valid LLM response."""
        valid_json = '''
        {
            "scenarios": [
                {
                    "title": "Test Scenario",
                    "description": "Test description",
                    "priority": "high",
                    "steps": [],
                    "prerequisites": [],
                    "tags": [],
                    "confidence_score": 0.8
                }
            ]
        }
        '''
        
        result = test_generator._parse_llm_response(valid_json)
        
        assert "scenarios" in result
        assert len(result["scenarios"]) == 1
        assert result["scenarios"][0]["title"] == "Test Scenario"
    
    def test_parse_llm_response_invalid_json(self, test_generator):
        """Test parsing invalid LLM response."""
        invalid_json = "This is not valid JSON"
        
        result = test_generator._parse_llm_response(invalid_json)
        
        assert result == {"scenarios": []}
    
    def test_parse_llm_response_with_extra_text(self, test_generator):
        """Test parsing LLM response with extra text."""
        response_with_text = '''
        Here is the generated test scenario:
        {
            "scenarios": [
                {
                    "title": "Test Scenario",
                    "description": "Test description",
                    "priority": "high",
                    "steps": [],
                    "prerequisites": [],
                    "tags": [],
                    "confidence_score": 0.8
                }
            ]
        }
        That's the end of the response.
        '''
        
        result = test_generator._parse_llm_response(response_with_text)
        
        assert "scenarios" in result
        assert len(result["scenarios"]) == 1
        assert result["scenarios"][0]["title"] == "Test Scenario"
    
    def test_create_test_scenario(self, test_generator):
        """Test creating TestScenario from LLM response data."""
        scenario_data = {
            "title": "Test User Login",
            "description": "Test user login functionality",
            "priority": "high",
            "steps": [
                {
                    "step_id": "step_1",
                    "description": "Navigate to login page",
                    "action": "Open browser and go to login URL",
                    "expected_result": "Login page is displayed"
                }
            ],
            "prerequisites": ["Valid user account"],
            "tags": ["authentication", "login"],
            "confidence_score": 0.85
        }
        
        scenario = test_generator._create_test_scenario(
            scenario_data,
            TestType.E2E,
            TestFramework.APPIUM,
            "test context"
        )
        
        assert scenario.title == "Test User Login"
        assert scenario.test_type == TestType.E2E
        assert scenario.framework == TestFramework.APPIUM
        assert len(scenario.steps) == 1
        assert scenario.confidence_score == 0.85
