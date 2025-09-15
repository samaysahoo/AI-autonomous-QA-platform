"""Synthetic data generator for demo mode testing."""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates realistic synthetic data for testing the AI Test Automation Platform."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the synthetic data generator."""
        if seed:
            random.seed(seed)
        
        # Sample data for realistic generation
        self.sample_users = [
            "john.doe@example.com", "jane.smith@example.com", "mike.wilson@example.com",
            "sarah.jones@example.com", "david.brown@example.com", "lisa.garcia@example.com",
            "alex.miller@example.com", "emma.davis@example.com", "chris.rodriguez@example.com",
            "amanda.taylor@example.com"
        ]
        
        self.sample_components = [
            "authentication", "user_management", "payment_processing", "notification_service",
            "file_upload", "search_engine", "recommendation_engine", "analytics_service",
            "api_gateway", "database_layer", "cache_service", "logging_service"
        ]
        
        self.sample_platforms = ["android", "ios", "web", "api"]
        
        self.sample_test_types = ["unit", "integration", "e2e", "performance", "security"]
        
        self.sample_error_types = [
            "ElementNotFound", "TimeoutException", "ConnectionError", "AuthenticationError",
            "ValidationError", "PermissionDenied", "RateLimitExceeded", "ServiceUnavailable",
            "DatabaseError", "NetworkError", "ConfigurationError", "ResourceExhausted"
        ]
        
        self.sample_browsers = ["chrome", "firefox", "safari", "edge"]
        
        self.sample_devices = [
            "iPhone 12", "iPhone 13", "iPhone 14", "Samsung Galaxy S21", "Samsung Galaxy S22",
            "Pixel 6", "Pixel 7", "iPad Air", "iPad Pro", "MacBook Pro", "Dell XPS", "Surface Pro"
        ]
        
        self.sample_environments = ["staging", "production", "development", "testing"]
        
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_jira_tickets(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic Jira tickets."""
        tickets = []
        
        issue_types = ["Story", "Bug", "Task", "Epic", "Sub-task"]
        priorities = ["Critical", "High", "Medium", "Low"]
        statuses = ["Open", "In Progress", "Code Review", "Testing", "Done", "Closed"]
        labels = ["frontend", "backend", "mobile", "api", "database", "security", "performance", "ui/ux"]
        
        for i in range(count):
            created_date = datetime.now() - timedelta(days=random.randint(1, 365))
            updated_date = created_date + timedelta(days=random.randint(0, 30))
            
            issue_type = random.choice(issue_types)
            priority = random.choice(priorities)
            
            if issue_type == "Story":
                title = f"As a user, I want to {self._generate_user_story_action()} so that {self._generate_user_story_benefit()}"
                description = self._generate_user_story_description()
            elif issue_type == "Bug":
                title = f"Bug: {self._generate_bug_description()}"
                description = self._generate_bug_report()
            else:
                title = f"Task: {self._generate_task_title()}"
                description = self._generate_task_description()
            
            ticket = {
                "id": f"DEMO-{1000 + i}",
                "key": f"DEMO-{1000 + i}",
                "summary": title,
                "description": description,
                "issue_type": issue_type,
                "priority": priority,
                "status": random.choice(statuses),
                "assignee": random.choice(self.sample_users),
                "reporter": random.choice(self.sample_users),
                "created": created_date.isoformat(),
                "updated": updated_date.isoformat(),
                "labels": random.sample(labels, random.randint(1, 3)),
                "components": random.sample(self.sample_components, random.randint(1, 2)),
                "story_points": random.randint(1, 8) if issue_type == "Story" else None,
                "fix_version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}" if issue_type == "Bug" else None,
                "environment": random.choice(self.sample_environments)
            }
            
            tickets.append(ticket)
        
        logger.info(f"Generated {count} Jira tickets")
        return tickets
    
    def generate_datadog_metrics(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate realistic Datadog metrics."""
        metrics = []
        
        metric_types = [
            "application.requests.count",
            "application.requests.duration",
            "application.errors.count",
            "application.memory.usage",
            "application.cpu.usage",
            "database.queries.count",
            "database.queries.duration",
            "cache.hits.count",
            "cache.misses.count",
            "external.api.calls.count"
        ]
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
            metric_name = random.choice(metric_types)
            
            # Generate realistic values based on metric type
            if "count" in metric_name:
                value = random.randint(1, 10000)
            elif "duration" in metric_name:
                value = random.uniform(0.001, 5.0)
            elif "usage" in metric_name:
                value = random.uniform(10.0, 95.0)
            else:
                value = random.uniform(0.0, 100.0)
            
            metric = {
                "timestamp": timestamp.isoformat(),
                "metric_name": metric_name,
                "value": value,
                "tags": {
                    "environment": random.choice(self.sample_environments),
                    "service": random.choice(self.sample_components),
                    "host": f"host-{random.randint(1, 100)}",
                    "version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}"
                }
            }
            
            metrics.append(metric)
        
        logger.info(f"Generated {count} Datadog metrics")
        return metrics
    
    def generate_sentry_events(self, count: int = 200) -> List[Dict[str, Any]]:
        """Generate realistic Sentry error events."""
        events = []
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(hours=random.randint(0, 720))
            error_type = random.choice(self.sample_error_types)
            
            event = {
                "event_id": f"demo-{random.randint(100000, 999999)}",
                "timestamp": timestamp.isoformat(),
                "level": random.choice(["error", "warning", "info", "debug"]),
                "message": self._generate_error_message(error_type),
                "exception": {
                    "type": error_type,
                    "value": self._generate_error_description(error_type),
                    "stacktrace": self._generate_stacktrace()
                },
                "tags": {
                    "environment": random.choice(self.sample_environments),
                    "release": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "user": random.choice(self.sample_users),
                    "browser": random.choice(self.sample_browsers),
                    "platform": random.choice(self.sample_platforms)
                },
                "contexts": {
                    "runtime": {
                        "name": "Python",
                        "version": "3.9.7"
                    },
                    "browser": {
                        "name": random.choice(self.sample_browsers),
                        "version": f"{random.randint(90, 120)}.0"
                    }
                },
                "user": {
                    "id": random.randint(1000, 9999),
                    "email": random.choice(self.sample_users)
                },
                "fingerprint": [f"demo-error-{random.randint(1, 10)}"],
                "culprit": f"src/{random.choice(self.sample_components)}/service.py",
                "logger": "demo.logger"
            }
            
            events.append(event)
        
        logger.info(f"Generated {count} Sentry events")
        return events
    
    def generate_github_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic GitHub events (commits, PRs, issues)."""
        events = []
        
        for i in range(count):
            event_type = random.choice(["commit", "pull_request", "issue", "release"])
            timestamp = datetime.now() - timedelta(hours=random.randint(0, 168))
            
            if event_type == "commit":
                event = self._generate_commit_event(timestamp)
            elif event_type == "pull_request":
                event = self._generate_pull_request_event(timestamp)
            elif event_type == "issue":
                event = self._generate_issue_event(timestamp)
            else:
                event = self._generate_release_event(timestamp)
            
            events.append(event)
        
        logger.info(f"Generated {count} GitHub events")
        return events
    
    def generate_test_execution_data(self, count: int = 300) -> List[Dict[str, Any]]:
        """Generate realistic test execution data."""
        executions = []
        
        test_names = [
            "test_user_login", "test_user_registration", "test_payment_processing",
            "test_file_upload", "test_search_functionality", "test_notification_delivery",
            "test_api_endpoints", "test_database_operations", "test_cache_performance",
            "test_security_validation", "test_mobile_responsiveness", "test_cross_browser_compatibility"
        ]
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
            test_name = random.choice(test_names)
            status = random.choice(["passed", "failed", "skipped", "error"])
            
            execution = {
                "execution_id": f"exec-{random.randint(10000, 99999)}",
                "test_name": test_name,
                "test_type": random.choice(self.sample_test_types),
                "platform": random.choice(self.sample_platforms),
                "environment": random.choice(self.sample_environments),
                "status": status,
                "duration": random.uniform(0.1, 120.0),
                "timestamp": timestamp.isoformat(),
                "browser": random.choice(self.sample_browsers) if random.choice(self.sample_platforms) == "web" else None,
                "device": random.choice(self.sample_devices) if random.choice(self.sample_platforms) in ["android", "ios"] else None,
                "error_message": self._generate_test_error() if status == "failed" else None,
                "screenshots": [f"screenshot-{random.randint(1, 100)}.png"] if status == "failed" else [],
                "tags": random.sample(["smoke", "regression", "critical", "performance"], random.randint(1, 2))
            }
            
            executions.append(execution)
        
        logger.info(f"Generated {count} test execution records")
        return executions
    
    def generate_user_feedback(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic user feedback data."""
        feedback = []
        
        feedback_types = ["bug_report", "feature_request", "improvement_suggestion", "test_feedback"]
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 90))
            feedback_type = random.choice(feedback_types)
            
            fb = {
                "feedback_id": f"fb-{random.randint(10000, 99999)}",
                "type": feedback_type,
                "title": self._generate_feedback_title(feedback_type),
                "description": self._generate_feedback_description(feedback_type),
                "user_email": random.choice(self.sample_users),
                "timestamp": timestamp.isoformat(),
                "priority": random.choice(["low", "medium", "high", "critical"]),
                "status": random.choice(["new", "in_review", "accepted", "rejected", "implemented"]),
                "tags": random.sample(["ui", "performance", "bug", "enhancement"], random.randint(1, 2)),
                "related_tests": [f"test-{random.randint(1, 20)}" for _ in range(random.randint(0, 3))]
            }
            
            feedback.append(fb)
        
        logger.info(f"Generated {count} user feedback records")
        return feedback
    
    # Helper methods for generating realistic content
    
    def _generate_user_story_action(self) -> str:
        """Generate user story action."""
        actions = [
            "login to the system", "upload a file", "search for products", "make a payment",
            "receive notifications", "view my profile", "download reports", "share content",
            "manage my settings", "track my orders", "rate products", "contact support"
        ]
        return random.choice(actions)
    
    def _generate_user_story_benefit(self) -> str:
        """Generate user story benefit."""
        benefits = [
            "I can access my account securely", "I can manage my files efficiently",
            "I can find what I'm looking for", "I can complete my purchases",
            "I stay informed about updates", "I can customize my experience",
            "I can get the information I need", "I can collaborate with others",
            "I can control my preferences", "I can monitor my transactions",
            "I can provide feedback", "I can get help when needed"
        ]
        return random.choice(benefits)
    
    def _generate_user_story_description(self) -> str:
        """Generate user story description."""
        return f"""
As a {random.choice(['user', 'customer', 'administrator', 'developer'])}, 
I want to {self._generate_user_story_action()} 
so that {self._generate_user_story_benefit()}.

Acceptance Criteria:
- The feature should work on {random.choice(['mobile', 'desktop', 'all platforms'])}
- Response time should be under {random.randint(1, 5)} seconds
- The feature should be accessible to users with disabilities
- Error handling should provide clear feedback to users
        """.strip()
    
    def _generate_bug_description(self) -> str:
        """Generate bug description."""
        descriptions = [
            "Login button not responding on mobile devices",
            "File upload fails for files larger than 10MB",
            "Search results not displaying correctly in Firefox",
            "Payment processing timeout after 30 seconds",
            "Notification emails not being sent",
            "Profile picture not updating after upload",
            "Reports download as corrupted files",
            "Share button missing from content pages",
            "Settings changes not being saved",
            "Order tracking shows incorrect status",
            "Product ratings not being recorded",
            "Support chat window not opening"
        ]
        return random.choice(descriptions)
    
    def _generate_bug_report(self) -> str:
        """Generate detailed bug report."""
        return f"""
**Bug Description:**
{self._generate_bug_description()}

**Steps to Reproduce:**
1. Navigate to the application
2. Perform the following actions:
   - {random.choice(['Click on', 'Enter', 'Select', 'Upload'])} {random.choice(['the button', 'data', 'an option', 'a file'])}
   - Wait for the system to process
   - Observe the error

**Expected Behavior:**
The system should {random.choice(['respond correctly', 'display the expected result', 'process the request', 'show success message'])}

**Actual Behavior:**
The system {random.choice(['does not respond', 'shows an error message', 'crashes', 'behaves unexpectedly'])}

**Environment:**
- Browser: {random.choice(self.sample_browsers)}
- Device: {random.choice(self.sample_devices)}
- Environment: {random.choice(self.sample_environments)}
- Version: v{random.randint(1, 3)}.{random.randint(0, 9)}

**Additional Information:**
This issue affects {random.choice(['all users', 'some users', 'specific user groups'])} and has a {random.choice(['high', 'medium', 'low'])} impact on user experience.
        """.strip()
    
    def _generate_task_title(self) -> str:
        """Generate task title."""
        titles = [
            "Implement user authentication", "Optimize database queries", "Add logging functionality",
            "Update API documentation", "Refactor legacy code", "Add unit tests",
            "Improve error handling", "Update dependencies", "Enhance security measures",
            "Optimize performance", "Add monitoring", "Implement caching"
        ]
        return random.choice(titles)
    
    def _generate_task_description(self) -> str:
        """Generate task description."""
        return f"""
Task: {self._generate_task_title()}

**Objective:**
{random.choice(['Improve system performance', 'Enhance user experience', 'Fix technical debt', 'Add new functionality', 'Strengthen security'])}

**Requirements:**
- Must be compatible with existing systems
- Should follow established coding standards
- Requires proper testing and documentation
- Must meet performance benchmarks

**Deliverables:**
- Implementation of the requested changes
- Unit tests with at least {random.randint(80, 95)}% coverage
- Updated documentation
- Performance benchmarks

**Estimated Effort:** {random.randint(2, 16)} hours
        """.strip()
    
    def _generate_error_message(self, error_type: str) -> str:
        """Generate error message."""
        messages = {
            "ElementNotFound": "Element not found: login_button",
            "TimeoutException": "Timeout waiting for element to be visible",
            "ConnectionError": "Failed to establish connection to server",
            "AuthenticationError": "Invalid credentials provided",
            "ValidationError": "Input validation failed",
            "PermissionDenied": "User does not have required permissions",
            "RateLimitExceeded": "API rate limit exceeded",
            "ServiceUnavailable": "Service temporarily unavailable",
            "DatabaseError": "Database connection failed",
            "NetworkError": "Network request failed",
            "ConfigurationError": "Invalid configuration detected",
            "ResourceExhausted": "System resources exhausted"
        }
        return messages.get(error_type, "An unexpected error occurred")
    
    def _generate_error_description(self, error_type: str) -> str:
        """Generate error description."""
        descriptions = {
            "ElementNotFound": "The specified UI element could not be located on the page",
            "TimeoutException": "The operation timed out while waiting for a response",
            "ConnectionError": "Unable to establish a network connection",
            "AuthenticationError": "The provided authentication credentials are invalid",
            "ValidationError": "The input data does not meet validation requirements",
            "PermissionDenied": "The current user lacks the necessary permissions",
            "RateLimitExceeded": "Too many requests have been made in a short time period",
            "ServiceUnavailable": "The requested service is currently unavailable",
            "DatabaseError": "An error occurred while accessing the database",
            "NetworkError": "A network-related error occurred",
            "ConfigurationError": "The system configuration contains invalid settings",
            "ResourceExhausted": "The system has run out of available resources"
        }
        return descriptions.get(error_type, "An unexpected system error occurred")
    
    def _generate_stacktrace(self) -> str:
        """Generate realistic stacktrace."""
        return f"""
Traceback (most recent call last):
  File "/app/src/{random.choice(self.sample_components)}/service.py", line {random.randint(10, 100)}, in {random.choice(['process_request', 'handle_data', 'execute_task'])}
    {random.choice(['result = api_call()', 'data = process_input()', 'response = send_request()'])}
  File "/app/src/api/client.py", line {random.randint(20, 80)}, in {random.choice(['make_request', 'send_data', 'call_endpoint'])}
    {random.choice(['raise ConnectionError()', 'raise TimeoutError()', 'raise ValidationError()'])}
{random.choice(self.sample_error_types)}: {self._generate_error_description(random.choice(self.sample_error_types))}
        """.strip()
    
    def _generate_commit_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate GitHub commit event."""
        return {
            "type": "commit",
            "timestamp": timestamp.isoformat(),
            "commit_id": f"demo{random.randint(100000, 999999)}",
            "message": random.choice([
                "Fix login validation issue",
                "Add new payment processing feature",
                "Optimize database queries",
                "Update API documentation",
                "Refactor user service",
                "Add error handling for file uploads",
                "Implement caching mechanism",
                "Fix mobile responsive layout"
            ]),
            "author": random.choice(self.sample_users),
            "files_changed": random.randint(1, 10),
            "lines_added": random.randint(10, 200),
            "lines_removed": random.randint(1, 50),
            "branch": random.choice(["main", "develop", "feature/payment", "hotfix/login"])
        }
    
    def _generate_pull_request_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate GitHub pull request event."""
        return {
            "type": "pull_request",
            "timestamp": timestamp.isoformat(),
            "pr_id": random.randint(100, 999),
            "title": random.choice([
                "Add user authentication feature",
                "Fix payment processing bug",
                "Improve API performance",
                "Update mobile UI components",
                "Add comprehensive test coverage",
                "Implement caching strategy",
                "Refactor database layer",
                "Enhance error handling"
            ]),
            "description": f"""
This PR {random.choice(['implements', 'fixes', 'improves', 'refactors'])} the {random.choice(self.sample_components)} functionality.

Changes:
- {random.choice(['Added new features', 'Fixed bugs', 'Improved performance', 'Updated documentation'])}
- {random.choice(['Enhanced security', 'Optimized queries', 'Improved UI', 'Added tests'])}

Testing:
- All existing tests pass
- New tests added for the changes
- Manual testing completed
            """.strip(),
            "author": random.choice(self.sample_users),
            "status": random.choice(["open", "closed", "merged"]),
            "files_changed": random.randint(1, 15),
            "commits": random.randint(1, 8),
            "reviewers": random.sample(self.sample_users, random.randint(1, 3))
        }
    
    def _generate_issue_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate GitHub issue event."""
        return {
            "type": "issue",
            "timestamp": timestamp.isoformat(),
            "issue_id": random.randint(1000, 9999),
            "title": random.choice([
                "Bug: Login fails on mobile devices",
                "Feature: Add dark mode support",
                "Enhancement: Improve search performance",
                "Bug: File upload timeout issue",
                "Feature: Add user preferences",
                "Enhancement: Better error messages",
                "Bug: API rate limiting problems",
                "Feature: Add export functionality"
            ]),
            "description": self._generate_bug_report(),
            "author": random.choice(self.sample_users),
            "status": random.choice(["open", "closed"]),
            "labels": random.sample(["bug", "enhancement", "feature", "documentation"], random.randint(1, 2)),
            "assignees": random.sample(self.sample_users, random.randint(0, 2))
        }
    
    def _generate_release_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate GitHub release event."""
        version = f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        return {
            "type": "release",
            "timestamp": timestamp.isoformat(),
            "version": version,
            "title": f"Release {version}",
            "description": f"""
## What's New in {version}

### Features
- {random.choice(['New user authentication system', 'Enhanced payment processing', 'Improved mobile experience'])}
- {random.choice(['Advanced search functionality', 'Better error handling', 'Performance optimizations'])}

### Bug Fixes
- Fixed login issues on mobile devices
- Resolved file upload timeout problems
- Corrected API rate limiting behavior

### Improvements
- Enhanced security measures
- Optimized database queries
- Improved user interface
            """.strip(),
            "author": random.choice(self.sample_users),
            "prerelease": random.choice([True, False]),
            "draft": False
        }
    
    def _generate_test_error(self) -> str:
        """Generate test error message."""
        errors = [
            "Element not found: login_button",
            "Timeout waiting for element to be visible",
            "Assertion failed: expected 'Welcome' but got 'Error'",
            "Page load timeout exceeded",
            "Network connection failed",
            "Invalid credentials provided",
            "Element not clickable",
            "Text not found on page",
            "Image comparison failed",
            "API response validation failed"
        ]
        return random.choice(errors)
    
    def _generate_feedback_title(self, feedback_type: str) -> str:
        """Generate feedback title."""
        titles = {
            "bug_report": "Bug: Login issue on mobile",
            "feature_request": "Feature: Add dark mode",
            "improvement_suggestion": "Improvement: Better error messages",
            "test_feedback": "Test feedback: Performance test results"
        }
        return titles.get(feedback_type, "General feedback")
    
    def _generate_feedback_description(self, feedback_type: str) -> str:
        """Generate feedback description."""
        descriptions = {
            "bug_report": "I'm experiencing issues with the login functionality on my mobile device. The login button doesn't respond when tapped.",
            "feature_request": "It would be great to have a dark mode option for the application. This would help reduce eye strain during night usage.",
            "improvement_suggestion": "The error messages could be more descriptive to help users understand what went wrong and how to fix it.",
            "test_feedback": "The performance tests are running well, but I noticed some flakiness in the mobile tests on slower devices."
        }
        return descriptions.get(feedback_type, "General feedback about the system.")
    
    def generate_all_demo_data(self) -> Dict[str, Any]:
        """Generate all demo data types."""
        logger.info("Generating comprehensive demo dataset...")
        
        demo_data = {
            "jira_tickets": self.generate_jira_tickets(75),
            "datadog_metrics": self.generate_datadog_metrics(1500),
            "sentry_events": self.generate_sentry_events(300),
            "github_events": self.generate_github_events(150),
            "test_executions": self.generate_test_execution_data(500),
            "user_feedback": self.generate_user_feedback(75),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_records": 0,
                "data_types": ["jira", "datadog", "sentry", "github", "test_executions", "user_feedback"]
            }
        }
        
        # Calculate total records
        total_records = sum(len(records) for key, records in demo_data.items() if key != "metadata")
        demo_data["metadata"]["total_records"] = total_records
        
        logger.info(f"Generated {total_records} total demo records")
        return demo_data
