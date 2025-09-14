"""Pytest configuration and fixtures for AI Test Automation Platform."""

import pytest
import asyncio
import tempfile
import os
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch

from config.settings import Settings
from src.data_ingestion.vector_store import VectorStoreManager
from src.test_generation.test_scenario import TestScenario, TestType, TestFramework, TestPriority


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Settings(
        openai_api_key="test-key",
        jira_url="https://test.atlassian.net",
        jira_username="test-user",
        jira_api_token="test-token",
        datadog_api_key="test-datadog-key",
        datadog_app_key="test-datadog-app",
        sentry_dsn="https://test@sentry.io/test",
        chroma_persist_directory="./test_data/chroma_db",
        faiss_index_path="./test_data/faiss_index",
        database_url="sqlite:///./test_data/test.db",
        redis_url="redis://localhost:6379/1",
        temporal_host="localhost:7233",
        kubeconfig_path="~/.kube/config",
        appium_server_url="http://localhost:4723",
        default_test_timeout=300,
        max_concurrent_tests=5,
        prometheus_port=8000,
        log_level="DEBUG"
    )
    return settings


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    with patch('src.data_ingestion.vector_store.VectorStoreManager') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.get_collection_stats.return_value = {
            'total_documents': 10,
            'chroma_count': 10,
            'faiss_count': 10
        }
        mock_instance.search_similar.return_value = []
        mock_instance.add_documents.return_value = True
        yield mock_instance


@pytest.fixture
def sample_test_scenario():
    """Sample test scenario for testing."""
    from src.test_generation.test_scenario import TestStep
    
    steps = [
        TestStep(
            step_id="step_1",
            description="Navigate to login page",
            action="Open browser and navigate to login URL",
            expected_result="Login page is displayed",
            locator="//a[@href='/login']"
        ),
        TestStep(
            step_id="step_2",
            description="Enter email address",
            action="Enter valid email in email field",
            expected_result="Email is entered successfully",
            locator="//input[@id='email']"
        ),
        TestStep(
            step_id="step_3",
            description="Enter password",
            action="Enter valid password in password field",
            expected_result="Password is entered successfully",
            locator="//input[@id='password']"
        ),
        TestStep(
            step_id="step_4",
            description="Click login button",
            action="Click the login button",
            expected_result="User is logged in successfully",
            locator="//button[@type='submit']"
        )
    ]
    
    scenario = TestScenario(
        scenario_id="test_scenario_1",
        title="User Login Flow",
        description="Test the complete user login process with valid credentials",
        test_type=TestType.E2E,
        framework=TestFramework.APPIUM,
        priority=TestPriority.HIGH,
        steps=steps,
        prerequisites=["Valid user account exists", "Application is running"],
        tags=["authentication", "login", "e2e"],
        expected_duration=120,
        source_documents=["doc_1", "doc_2"],
        generated_by="ai_test_generator",
        confidence_score=0.85,
        metadata={
            "components": ["authentication"],
            "browser": "chrome",
            "platform": "mobile"
        }
    )
    
    return scenario


@pytest.fixture
def sample_failure_data():
    """Sample failure data for testing."""
    return [
        {
            "test_name": "test_user_login",
            "error_message": "Element not found: login button",
            "stack_trace": "NoSuchElementException at LoginPage.java:45",
            "duration": 30,
            "timestamp": "2024-01-15T10:30:00Z",
            "environment": "staging",
            "browser": "chrome",
            "error_type": "NoSuchElementException",
            "severity": "Medium"
        },
        {
            "test_name": "test_user_registration",
            "error_message": "Element not found: register button",
            "stack_trace": "NoSuchElementException at RegistrationPage.java:67",
            "duration": 45,
            "timestamp": "2024-01-15T10:35:00Z",
            "environment": "staging",
            "browser": "chrome",
            "error_type": "NoSuchElementException",
            "severity": "Medium"
        }
    ]


@pytest.fixture
def sample_crash_events():
    """Sample crash events for testing."""
    from src.data_ingestion.analytics_ingestor import CrashEvent
    
    return [
        CrashEvent(
            event_id="crash_1",
            timestamp="2024-01-15T10:30:00Z",
            error_type="AuthenticationError",
            error_message="Invalid credentials provided",
            stack_trace="AuthenticationError at auth.py:45",
            user_id="user123",
            session_id="session456",
            device_info={"os": "iOS", "version": "15.0"},
            app_version="1.2.3",
            severity="Medium",
            tags={"component": "authentication"}
        ),
        CrashEvent(
            event_id="crash_2",
            timestamp="2024-01-15T10:35:00Z",
            error_type="DatabaseConnectionError",
            error_message="Connection pool exhausted",
            stack_trace="DatabaseConnectionError at db.py:123",
            user_id="user456",
            session_id="session789",
            device_info={"os": "Android", "version": "11.0"},
            app_version="1.2.3",
            severity="High",
            tags={"component": "database"}
        )
    ]


@pytest.fixture
def sample_jira_tickets():
    """Sample Jira tickets for testing."""
    from src.data_ingestion.jira_ingestor import JiraTicket
    
    return [
        JiraTicket(
            key="PROJ-123",
            summary="Implement user authentication",
            description="As a user, I want to be able to log in to the application",
            issue_type="Story",
            status="In Progress",
            priority="High",
            assignee="John Doe",
            created="2024-01-10T09:00:00Z",
            updated="2024-01-15T14:30:00Z",
            labels=["authentication", "user-management"],
            components=["auth"],
            epic_link="PROJ-100",
            story_points=5
        ),
        JiraTicket(
            key="PROJ-124",
            summary="Fix login button not working",
            description="Login button is not responding to clicks",
            issue_type="Bug",
            status="Open",
            priority="Critical",
            assignee="Jane Smith",
            created="2024-01-12T11:00:00Z",
            updated="2024-01-15T16:00:00Z",
            labels=["bug", "authentication"],
            components=["ui", "auth"],
            epic_link=None,
            story_points=None
        )
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Mock chat completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
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
        
        mock_instance.chat.completions.create.return_value = mock_response
        yield mock_instance


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing."""
    with patch('kubernetes.client.BatchV1Api') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Mock job creation response
        mock_response = Mock()
        mock_instance.create_namespaced_job.return_value = mock_response
        
        yield mock_instance


@pytest.fixture
def mock_temporal_client():
    """Mock Temporal client for testing."""
    with patch('temporalio.client.Client') as mock:
        mock_instance = Mock()
        mock.connect.return_value = mock_instance
        
        # Mock workflow start
        mock_handle = Mock()
        mock_instance.start_workflow.return_value = mock_handle
        
        yield mock_instance


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    
    # Clean up any test files created during tests
    test_dirs = ["./test_data", "./logs", "./screenshots", "./test_results"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
