"""Demo mode configuration and management."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .synthetic_data_generator import SyntheticDataGenerator
from .jira_mock import JiraMockData
from .datadog_mock import DatadogMockData
from .sentry_mock import SentryMockData
from .github_mock import GitHubMockData

logger = logging.getLogger(__name__)


class DemoMode:
    """Demo mode manager for the AI Test Automation Platform."""
    
    def __init__(self, demo_data_dir: str = "demo_data"):
        """Initialize demo mode."""
        self.demo_data_dir = Path(demo_data_dir)
        self.demo_data_dir.mkdir(exist_ok=True)
        
        # Initialize synthetic data generators
        self.synthetic_generator = SyntheticDataGenerator(seed=42)
        
        # Initialize mock APIs
        self.jira_mock = JiraMockData(use_synthetic_data=True)
        self.datadog_mock = DatadogMockData(use_synthetic_data=True)
        self.sentry_mock = SentryMockData(use_synthetic_data=True)
        self.github_mock = GitHubMockData(use_synthetic_data=True)
        
        # Demo data cache
        self._demo_data_cache = {}
        self._cache_timestamp = None
        
        logger.info("DemoMode initialized")
    
    def is_demo_mode(self) -> bool:
        """Check if demo mode is enabled."""
        return True  # Always enabled for demo mode
    
    def generate_demo_dataset(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """Generate comprehensive demo dataset."""
        
        # Check if we have cached data
        if not force_regenerate and self._demo_data_cache and self._cache_timestamp:
            cache_age = datetime.now() - self._cache_timestamp
            if cache_age.total_seconds() < 3600:  # Cache valid for 1 hour
                logger.info("Using cached demo data")
                return self._demo_data_cache
        
        logger.info("Generating comprehensive demo dataset...")
        
        # Generate all demo data
        demo_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_records": 0,
                "data_sources": ["jira", "datadog", "sentry", "github", "test_executions", "user_feedback"]
            },
            "jira": {
                "tickets": self.jira_mock.generate_jira_tickets(100),
                "projects": self.jira_mock.get_projects(),
                "statistics": self.jira_mock.get_issue_statistics()
            },
            "datadog": {
                "metrics": self.datadog_mock.generate_datadog_metrics(2000),
                "logs": self.datadog_mock.get_logs(limit=500),
                "events": self.datadog_mock.get_events(),
                "dashboards": self.datadog_mock.get_dashboards(),
                "monitors": self.datadog_mock.get_monitors(),
                "service_map": self.datadog_mock.get_service_map()
            },
            "sentry": {
                "events": self.sentry_mock.generate_sentry_events(500),
                "issues": self.sentry_mock.get_issues(limit=50),
                "releases": self.sentry_mock.get_releases(limit=20),
                "projects": self.sentry_mock.get_projects(),
                "teams": self.sentry_mock.get_teams(),
                "alerts": self.sentry_mock.get_alerts(limit=20),
                "performance_data": self.sentry_mock.get_performance_data()
            },
            "github": {
                "repositories": self.github_mock.get_repositories(),
                "commits": self.github_mock.get_commits("demo-org", "demo-backend", limit=100),
                "pull_requests": self.github_mock.get_pull_requests("demo-org", "demo-backend", limit=30),
                "issues": self.github_mock.get_issues("demo-org", "demo-backend", limit=30),
                "releases": self.github_mock.get_releases("demo-org", "demo-backend", limit=10),
                "workflow_runs": self.github_mock.get_workflow_runs("demo-org", "demo-backend", limit=20),
                "branches": self.github_mock.get_branches("demo-org", "demo-backend", limit=15)
            },
            "test_executions": self.synthetic_generator.generate_test_execution_data(800),
            "user_feedback": self.synthetic_generator.generate_user_feedback(100),
            "analytics_events": self._generate_analytics_events(),
            "system_metrics": self._generate_system_metrics()
        }
        
        # Calculate total records
        total_records = 0
        for source, data in demo_data.items():
            if source != "metadata":
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            total_records += len(value)
                elif isinstance(data, list):
                    total_records += len(data)
        
        demo_data["metadata"]["total_records"] = total_records
        
        # Cache the data
        self._demo_data_cache = demo_data
        self._cache_timestamp = datetime.now()
        
        # Save to file
        self._save_demo_data(demo_data)
        
        logger.info(f"Generated demo dataset with {total_records} total records")
        return demo_data
    
    def get_demo_data(self, data_type: str = None) -> Any:
        """Get demo data by type."""
        if not self._demo_data_cache:
            self.generate_demo_dataset()
        
        if data_type is None:
            return self._demo_data_cache
        
        return self._demo_data_cache.get(data_type, {})
    
    def get_jira_demo_data(self) -> Dict[str, Any]:
        """Get Jira demo data."""
        return self.get_demo_data("jira")
    
    def get_datadog_demo_data(self) -> Dict[str, Any]:
        """Get Datadog demo data."""
        return self.get_demo_data("datadog")
    
    def get_sentry_demo_data(self) -> Dict[str, Any]:
        """Get Sentry demo data."""
        return self.get_demo_data("sentry")
    
    def get_github_demo_data(self) -> Dict[str, Any]:
        """Get GitHub demo data."""
        return self.get_demo_data("github")
    
    def get_test_execution_demo_data(self) -> List[Dict[str, Any]]:
        """Get test execution demo data."""
        return self.get_demo_data("test_executions")
    
    def get_user_feedback_demo_data(self) -> List[Dict[str, Any]]:
        """Get user feedback demo data."""
        return self.get_demo_data("user_feedback")
    
    def get_analytics_events_demo_data(self) -> List[Dict[str, Any]]:
        """Get analytics events demo data."""
        return self.get_demo_data("analytics_events")
    
    def get_system_metrics_demo_data(self) -> Dict[str, Any]:
        """Get system metrics demo data."""
        return self.get_demo_data("system_metrics")
    
    def create_demo_workflow_input(self, workflow_type: str = "e2e") -> Dict[str, Any]:
        """Create demo input data for workflows."""
        
        if workflow_type == "e2e":
            return {
                "change_type": "feature_implementation",
                "diff_content": """
+def new_authentication_feature():
+    # New OAuth2 authentication implementation
+    return authenticate_with_oauth2()
+    
+def validate_user_permissions(user_id):
+    # Validate user permissions for new feature
+    return check_user_access(user_id)
+    
+class AuthService:
+    def __init__(self):
+        self.oauth2_client = OAuth2Client()
+    
+    def authenticate_user(self, credentials):
+        # Authenticate user with OAuth2
+        token = self.oauth2_client.get_token(credentials)
+        return self.validate_token(token)
+    
+    def validate_token(self, token):
+        # Validate JWT token
+        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                """,
                "changed_files": [
                    "src/auth/oauth2_service.py",
                    "src/auth/permission_validator.py", 
                    "src/auth/jwt_handler.py",
                    "tests/auth/test_oauth2.py"
                ],
                "commit_metadata": {
                    "message": "feat: implement OAuth2 authentication with permission validation",
                    "author": "demo.developer@example.com",
                    "timestamp": datetime.now().isoformat(),
                    "branch": "feature/oauth2-auth",
                    "pull_request": "#123"
                },
                "requirements": [
                    "OAuth2 authentication must be secure and follow industry standards",
                    "User permissions must be validated for each request",
                    "Feature must be testable across all platforms (web, mobile, API)",
                    "Performance impact should be minimal (<100ms overhead)",
                    "Must support multiple OAuth2 providers (Google, GitHub, Microsoft)",
                    "JWT tokens should have configurable expiration times",
                    "Error handling must provide clear feedback to users"
                ],
                "jira_tickets": self.jira_mock.generate_jira_tickets(5),
                "historical_failures": [
                    {
                        "component": "authentication",
                        "description": "Login timeout on mobile devices",
                        "frequency": 15,
                        "last_occurrence": (datetime.now() - timedelta(days=7)).isoformat()
                    },
                    {
                        "component": "permissions",
                        "description": "Permission validation fails for admin users",
                        "frequency": 3,
                        "last_occurrence": (datetime.now() - timedelta(days=14)).isoformat()
                    }
                ]
            }
        
        elif workflow_type == "bug_triage":
            return {
                "test_results": [
                    {
                        "scenario_id": "auth-test-001",
                        "status": "failed",
                        "error_message": "Element not found: login_button",
                        "execution_time": 45.2,
                        "screenshots": ["screenshot_auth_fail_001.png"],
                        "platform": "android",
                        "browser": None,
                        "device": "Samsung Galaxy S21",
                        "environment": "staging"
                    },
                    {
                        "scenario_id": "auth-test-002",
                        "status": "failed", 
                        "error_message": "Element not found: login_button",
                        "execution_time": 43.8,
                        "screenshots": ["screenshot_auth_fail_002.png"],
                        "platform": "android",
                        "browser": None,
                        "device": "Pixel 6",
                        "environment": "staging"
                    },
                    {
                        "scenario_id": "auth-test-003",
                        "status": "failed",
                        "error_message": "Timeout waiting for element: submit_form",
                        "execution_time": 120.0,
                        "screenshots": ["screenshot_auth_timeout_001.png"],
                        "platform": "web",
                        "browser": "Chrome",
                        "device": None,
                        "environment": "production"
                    },
                    {
                        "scenario_id": "payment-test-001",
                        "status": "failed",
                        "error_message": "Payment gateway connection failed",
                        "execution_time": 30.5,
                        "screenshots": ["screenshot_payment_fail_001.png"],
                        "platform": "web",
                        "browser": "Firefox",
                        "device": None,
                        "environment": "staging"
                    },
                    {
                        "scenario_id": "notification-test-001",
                        "status": "passed",
                        "execution_time": 12.3,
                        "platform": "api",
                        "browser": None,
                        "device": None,
                        "environment": "production"
                    }
                ],
                "clustering_method": "auto",
                "min_cluster_size": 2
            }
        
        elif workflow_type == "performance_optimization":
            return {
                "performance_data": {
                    "current_metrics": {
                        "success_rate": 0.85,
                        "average_execution_time": 65.2,
                        "resource_utilization": 0.78,
                        "memory_usage": 0.82,
                        "cpu_usage": 0.65
                    },
                    "optimization_targets": ["execution_time", "success_rate", "resource_utilization"],
                    "historical_data": {
                        "previous_success_rate": 0.82,
                        "previous_execution_time": 72.1,
                        "previous_resource_utilization": 0.85
                    },
                    "trend_analysis": {
                        "execution_time_trend": "increasing",
                        "success_rate_trend": "stable",
                        "resource_usage_trend": "decreasing"
                    }
                }
            }
        
        else:
            return {}
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get demo data statistics."""
        demo_data = self.get_demo_data()
        
        stats = {
            "total_records": demo_data["metadata"]["total_records"],
            "data_sources": demo_data["metadata"]["data_sources"],
            "breakdown": {
                "jira_tickets": len(demo_data["jira"]["tickets"]),
                "datadog_metrics": len(demo_data["datadog"]["metrics"]),
                "sentry_events": len(demo_data["sentry"]["events"]),
                "github_commits": len(demo_data["github"]["commits"]),
                "test_executions": len(demo_data["test_executions"]),
                "user_feedback": len(demo_data["user_feedback"]),
                "analytics_events": len(demo_data["analytics_events"])
            },
            "generated_at": demo_data["metadata"]["generated_at"],
            "cache_age_minutes": (datetime.now() - datetime.fromisoformat(demo_data["metadata"]["generated_at"])).total_seconds() / 60
        }
        
        return stats
    
    def _generate_analytics_events(self) -> List[Dict[str, Any]]:
        """Generate analytics events."""
        events = []
        event_types = [
            "user_login", "user_logout", "page_view", "button_click", "form_submit",
            "file_upload", "search_query", "purchase_complete", "error_occurred", "feature_used"
        ]
        
        for i in range(500):
            event_time = datetime.now() - timedelta(hours=random.randint(0, 720))
            
            event = {
                "event_id": f"analytics-{random.randint(10000, 99999)}",
                "timestamp": event_time.isoformat(),
                "event_type": random.choice(event_types),
                "user_id": random.randint(1000, 9999),
                "session_id": f"session-{random.randint(100000, 999999)}",
                "properties": {
                    "page_url": f"https://demo.example.com/{random.choice(['home', 'login', 'dashboard', 'profile'])}",
                    "user_agent": f"{random.choice(['Chrome', 'Firefox', 'Safari'])}/120.0",
                    "device_type": random.choice(["desktop", "mobile", "tablet"]),
                    "country": random.choice(["US", "GB", "DE", "FR", "CA"]),
                    "duration": random.uniform(1.0, 300.0) if random.choice(event_types) == "page_view" else None
                }
            }
            
            events.append(event)
        
        return events
    
    def _generate_system_metrics(self) -> Dict[str, Any]:
        """Generate system metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": random.uniform(20.0, 80.0),
            "memory_usage": random.uniform(30.0, 90.0),
            "disk_usage": random.uniform(40.0, 85.0),
            "network_io": {
                "bytes_sent": random.randint(1000000, 10000000),
                "bytes_received": random.randint(2000000, 20000000)
            },
            "active_connections": random.randint(50, 500),
            "request_rate": random.uniform(10.0, 1000.0),
            "error_rate": random.uniform(0.01, 0.05),
            "response_time_p95": random.uniform(0.1, 2.0),
            "database_connections": random.randint(10, 100),
            "cache_hit_rate": random.uniform(0.8, 0.99)
        }
    
    def _save_demo_data(self, demo_data: Dict[str, Any]) -> None:
        """Save demo data to file."""
        try:
            demo_file = self.demo_data_dir / "demo_dataset.json"
            
            # Remove large data that's not needed for persistence
            save_data = demo_data.copy()
            save_data["metadata"]["saved_to_file"] = True
            save_data["metadata"]["file_size_mb"] = 0  # Will be calculated
            
            with open(demo_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            file_size = demo_file.stat().st_size / (1024 * 1024)  # MB
            save_data["metadata"]["file_size_mb"] = round(file_size, 2)
            
            logger.info(f"Demo data saved to {demo_file} ({file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save demo data: {e}")
    
    def load_demo_data_from_file(self) -> Optional[Dict[str, Any]]:
        """Load demo data from file."""
        try:
            demo_file = self.demo_data_dir / "demo_dataset.json"
            
            if not demo_file.exists():
                logger.info("No cached demo data file found")
                return None
            
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)
            
            logger.info(f"Demo data loaded from {demo_file}")
            return demo_data
            
        except Exception as e:
            logger.error(f"Failed to load demo data: {e}")
            return None
    
    def clear_demo_cache(self) -> None:
        """Clear demo data cache."""
        self._demo_data_cache = {}
        self._cache_timestamp = None
        logger.info("Demo data cache cleared")
    
    def get_mock_apis(self) -> Dict[str, Any]:
        """Get mock API instances."""
        return {
            "jira": self.jira_mock,
            "datadog": self.datadog_mock,
            "sentry": self.sentry_mock,
            "github": self.github_mock
        }
