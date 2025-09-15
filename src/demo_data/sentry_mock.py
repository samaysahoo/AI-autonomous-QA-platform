"""Mock Sentry API for demo mode."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class SentryMockData:
    """Mock Sentry API for demo mode testing."""
    
    def __init__(self, use_synthetic_data: bool = True):
        """Initialize Sentry mock data."""
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_generator = SyntheticDataGenerator(seed=42) if use_synthetic_data else None
        
        # Cache for demo data
        self._cached_events = None
        
        logger.info("SentryMockData initialized for demo mode")
    
    def get_events(self,
                  start_time: datetime = None,
                  end_time: datetime = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get error events from Sentry."""
        if not self.use_synthetic_data:
            return []
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        # Generate synthetic events if not cached
        if self._cached_events is None:
            self._cached_events = self.synthetic_generator.generate_sentry_events(500)
        
        # Filter events by time range
        filtered_events = []
        for event in self._cached_events:
            event_time = datetime.fromisoformat(event["timestamp"])
            if start_time <= event_time <= end_time:
                filtered_events.append(event)
        
        return filtered_events[:limit]
    
    def get_issues(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get error issues from Sentry."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic issues based on error types
        issues = []
        error_types = [
            "ElementNotFound", "TimeoutException", "ConnectionError", "AuthenticationError",
            "ValidationError", "PermissionDenied", "RateLimitExceeded", "ServiceUnavailable"
        ]
        
        for i, error_type in enumerate(error_types):
            issue = {
                "id": f"issue-{i+1}",
                "title": f"{error_type}: {self._generate_issue_title(error_type)}",
                "level": random.choice(["error", "warning", "info"]),
                "status": random.choice(["unresolved", "resolved", "ignored"]),
                "first_seen": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "last_seen": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "count": random.randint(1, 1000),
                "user_count": random.randint(1, 100),
                "culprit": f"src/{random.choice(['auth', 'payment', 'notification'])}/service.py",
                "permalink": f"https://demo.sentry.io/issues/{i+1}/",
                "project": {
                    "id": "1",
                    "name": "demo-project",
                    "slug": "demo-project"
                },
                "type": "error",
                "metadata": {
                    "type": error_type,
                    "value": self._generate_error_description(error_type)
                },
                "tags": [
                    {"key": "environment", "value": random.choice(["production", "staging", "development"])},
                    {"key": "release", "value": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"},
                    {"key": "platform", "value": random.choice(["python", "javascript", "java", "csharp"])},
                    {"key": "browser", "value": random.choice(["Chrome", "Firefox", "Safari", "Edge"])}
                ]
            }
            
            issues.append(issue)
        
        return issues[:limit]
    
    def get_issue_events(self, issue_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get events for a specific issue."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic events for the issue
        events = []
        
        for i in range(min(limit, 20)):
            event_time = datetime.now() - timedelta(hours=random.randint(1, 168))
            
            event = {
                "id": f"event-{random.randint(10000, 99999)}",
                "timestamp": event_time.isoformat(),
                "level": random.choice(["error", "warning", "info"]),
                "message": self._generate_error_message(),
                "exception": {
                    "values": [{
                        "type": "DemoError",
                        "value": "This is a demo error for testing purposes",
                        "stacktrace": {
                            "frames": [
                                {
                                    "filename": "demo_service.py",
                                    "function": "process_request",
                                    "lineno": random.randint(10, 100),
                                    "context_line": "result = api_call()",
                                    "pre_context": ["def process_request():", "    try:"],
                                    "post_context": ["    except Exception as e:", "        raise"]
                                },
                                {
                                    "filename": "api_client.py", 
                                    "function": "api_call",
                                    "lineno": random.randint(20, 80),
                                    "context_line": "raise ConnectionError()",
                                    "pre_context": ["def api_call():", "    response = requests.get(url)"],
                                    "post_context": ["    return response", ""]
                                }
                            ]
                        }
                    }]
                },
                "tags": {
                    "environment": random.choice(["production", "staging"]),
                    "release": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "user": f"user{random.randint(1, 100)}@example.com",
                    "browser": random.choice(["Chrome", "Firefox", "Safari"]),
                    "platform": "python"
                },
                "user": {
                    "id": random.randint(1000, 9999),
                    "email": f"user{random.randint(1, 100)}@example.com"
                },
                "request": {
                    "url": f"https://demo.example.com/api/{random.choice(['users', 'payments', 'notifications'])}",
                    "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                    "headers": {
                        "User-Agent": f"{random.choice(['Chrome', 'Firefox', 'Safari'])}/120.0"
                    }
                },
                "extra": {
                    "request_id": f"req-{random.randint(10000, 99999)}",
                    "user_agent": f"{random.choice(['Chrome', 'Firefox', 'Safari'])}/120.0",
                    "ip_address": f"192.168.1.{random.randint(1, 254)}"
                }
            }
            
            events.append(event)
        
        return events
    
    def get_releases(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get releases from Sentry."""
        releases = []
        
        for i in range(min(limit, 20)):
            release_date = datetime.now() - timedelta(days=random.randint(1, 90))
            
            release = {
                "version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "short_version": f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "date_created": release_date.isoformat(),
                "date_released": release_date.isoformat(),
                "new_groups": random.randint(0, 10),
                "deploy_count": random.randint(1, 5),
                "commit_count": random.randint(5, 50),
                "last_commit": {
                    "id": f"commit-{random.randint(100000, 999999)}",
                    "message": random.choice([
                        "Fix authentication bug",
                        "Add new payment feature", 
                        "Improve error handling",
                        "Update dependencies",
                        "Optimize performance"
                    ]),
                    "author": {
                        "name": f"Developer {random.randint(1, 10)}",
                        "email": f"dev{random.randint(1, 10)}@example.com"
                    }
                },
                "projects": [
                    {
                        "id": "1",
                        "name": "demo-project",
                        "slug": "demo-project"
                    }
                ],
                "url": f"https://demo.sentry.io/releases/v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}/"
            }
            
            releases.append(release)
        
        return releases
    
    def get_teams(self) -> List[Dict[str, Any]]:
        """Get teams from Sentry."""
        teams = [
            {
                "id": "1",
                "name": "Backend Team",
                "slug": "backend-team",
                "date_created": (datetime.now() - timedelta(days=365)).isoformat(),
                "is_member": True,
                "member_count": 5
            },
            {
                "id": "2", 
                "name": "Frontend Team",
                "slug": "frontend-team",
                "date_created": (datetime.now() - timedelta(days=300)).isoformat(),
                "is_member": True,
                "member_count": 4
            },
            {
                "id": "3",
                "name": "Mobile Team", 
                "slug": "mobile-team",
                "date_created": (datetime.now() - timedelta(days=200)).isoformat(),
                "is_member": False,
                "member_count": 3
            }
        ]
        
        return teams
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get projects from Sentry."""
        projects = [
            {
                "id": "1",
                "name": "Demo Backend",
                "slug": "demo-backend",
                "platform": "python",
                "date_created": (datetime.now() - timedelta(days=365)).isoformat(),
                "is_member": True,
                "features": ["alerts", "releases", "performance"],
                "status": "active"
            },
            {
                "id": "2",
                "name": "Demo Frontend",
                "slug": "demo-frontend", 
                "platform": "javascript",
                "date_created": (datetime.now() - timedelta(days=300)).isoformat(),
                "is_member": True,
                "features": ["alerts", "releases", "performance"],
                "status": "active"
            },
            {
                "id": "3",
                "name": "Demo Mobile",
                "slug": "demo-mobile",
                "platform": "react-native",
                "date_created": (datetime.now() - timedelta(days=200)).isoformat(),
                "is_member": False,
                "features": ["alerts", "releases"],
                "status": "active"
            }
        ]
        
        return projects
    
    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get alerts from Sentry."""
        alerts = []
        
        alert_types = ["error", "performance", "issue", "metric"]
        
        for i in range(min(limit, 20)):
            alert_time = datetime.now() - timedelta(hours=random.randint(1, 168))
            
            alert = {
                "id": f"alert-{random.randint(10000, 99999)}",
                "title": self._generate_alert_title(),
                "message": self._generate_alert_message(),
                "type": random.choice(alert_types),
                "status": random.choice(["triggered", "resolved", "muted"]),
                "date_created": alert_time.isoformat(),
                "date_updated": (alert_time + timedelta(minutes=random.randint(1, 60))).isoformat(),
                "project": {
                    "id": "1",
                    "name": "demo-backend",
                    "slug": "demo-backend"
                },
                "conditions": [
                    {
                        "id": "1",
                        "alert_rule_id": f"rule-{random.randint(1, 10)}",
                        "label": "Error rate is above 5%",
                        "threshold": 5,
                        "comparison": "greater_than"
                    }
                ],
                "triggered_at": alert_time.isoformat(),
                "resolved_at": (alert_time + timedelta(minutes=random.randint(10, 120))).isoformat() if random.choice([True, False]) else None
            }
            
            alerts.append(alert)
        
        return alerts
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get alert rules from Sentry."""
        rules = [
            {
                "id": "1",
                "name": "High Error Rate",
                "conditions": [
                    {
                        "id": "1",
                        "label": "Error rate is above 5%",
                        "threshold": 5,
                        "comparison": "greater_than",
                        "time_window": 300
                    }
                ],
                "actions": [
                    {
                        "type": "email",
                        "target": "team@example.com"
                    },
                    {
                        "type": "slack",
                        "target": "#alerts"
                    }
                ],
                "frequency": 300,
                "date_created": (datetime.now() - timedelta(days=30)).isoformat(),
                "date_updated": datetime.now().isoformat()
            },
            {
                "id": "2",
                "name": "Performance Degradation",
                "conditions": [
                    {
                        "id": "2",
                        "label": "Average response time above 2s",
                        "threshold": 2,
                        "comparison": "greater_than",
                        "time_window": 600
                    }
                ],
                "actions": [
                    {
                        "type": "email",
                        "target": "devops@example.com"
                    }
                ],
                "frequency": 600,
                "date_created": (datetime.now() - timedelta(days=20)).isoformat(),
                "date_updated": datetime.now().isoformat()
            }
        ]
        
        return rules
    
    def get_performance_data(self,
                           start_time: datetime = None,
                           end_time: datetime = None) -> Dict[str, Any]:
        """Get performance data from Sentry."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        # Generate synthetic performance data
        transactions = []
        
        for i in range(100):
            transaction_time = start_time + timedelta(minutes=random.randint(0, int((end_time - start_time).total_seconds() / 60)))
            
            transaction = {
                "id": f"transaction-{random.randint(10000, 99999)}",
                "timestamp": transaction_time.isoformat(),
                "transaction": f"/api/{random.choice(['users', 'payments', 'notifications'])}",
                "duration": random.uniform(0.001, 5.0),
                "status": random.choice(["ok", "cancelled", "internal_error", "aborted", "deadline_exceeded"]),
                "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "tags": {
                    "environment": random.choice(["production", "staging"]),
                    "release": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "user": f"user{random.randint(1, 100)}@example.com"
                },
                "user": {
                    "id": random.randint(1000, 9999),
                    "email": f"user{random.randint(1, 100)}@example.com"
                }
            }
            
            transactions.append(transaction)
        
        return {
            "transactions": transactions,
            "summary": {
                "total_transactions": len(transactions),
                "avg_duration": sum(t["duration"] for t in transactions) / len(transactions),
                "error_rate": len([t for t in transactions if t["status"] != "ok"]) / len(transactions),
                "p95_duration": sorted([t["duration"] for t in transactions])[int(len(transactions) * 0.95)]
            }
        }
    
    # Helper methods
    
    def _generate_issue_title(self, error_type: str) -> str:
        """Generate issue title."""
        titles = {
            "ElementNotFound": "Element not found in UI",
            "TimeoutException": "Operation timed out",
            "ConnectionError": "Network connection failed",
            "AuthenticationError": "Authentication failed",
            "ValidationError": "Input validation failed",
            "PermissionDenied": "Access denied",
            "RateLimitExceeded": "Rate limit exceeded",
            "ServiceUnavailable": "Service unavailable"
        }
        return titles.get(error_type, "Unknown error occurred")
    
    def _generate_error_description(self, error_type: str) -> str:
        """Generate error description."""
        descriptions = {
            "ElementNotFound": "The specified UI element could not be located",
            "TimeoutException": "The operation exceeded the maximum allowed time",
            "ConnectionError": "Unable to establish network connection",
            "AuthenticationError": "Invalid credentials provided",
            "ValidationError": "Input data validation failed",
            "PermissionDenied": "Insufficient permissions for operation",
            "RateLimitExceeded": "API rate limit has been exceeded",
            "ServiceUnavailable": "The requested service is currently unavailable"
        }
        return descriptions.get(error_type, "An unexpected error occurred")
    
    def _generate_error_message(self) -> str:
        """Generate error message."""
        messages = [
            "Element not found: login_button",
            "Timeout waiting for element to be visible",
            "Connection failed to external service",
            "Invalid authentication token",
            "Validation failed for user input",
            "Permission denied for resource access",
            "Rate limit exceeded for API calls",
            "Service temporarily unavailable"
        ]
        return random.choice(messages)
    
    def _generate_alert_title(self) -> str:
        """Generate alert title."""
        titles = [
            "High Error Rate Detected",
            "Performance Degradation Alert",
            "New Issue Created",
            "Memory Usage High",
            "Response Time Slow",
            "Database Connection Failed",
            "External Service Down",
            "SSL Certificate Expiring"
        ]
        return random.choice(titles)
    
    def _generate_alert_message(self) -> str:
        """Generate alert message."""
        messages = [
            "Error rate has exceeded the threshold of 5%",
            "Average response time is above 2 seconds",
            "A new critical issue has been created",
            "Memory usage has reached 85%",
            "API response time has increased significantly",
            "Database connection pool is exhausted",
            "External payment service is not responding",
            "SSL certificate will expire in 7 days"
        ]
        return random.choice(messages)
