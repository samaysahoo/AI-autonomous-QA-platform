"""Mock Jira API for demo mode."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class JiraMockData:
    """Mock Jira API for demo mode testing."""
    
    def __init__(self, use_synthetic_data: bool = True):
        """Initialize Jira mock data."""
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_generator = SyntheticDataGenerator(seed=42) if use_synthetic_data else None
        
        # Cache for demo data
        self._cached_tickets = None
        self._cached_projects = None
        
        logger.info("JiraMockData initialized for demo mode")
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get available Jira projects."""
        if self._cached_projects is None:
            self._cached_projects = [
                {
                    "id": "10000",
                    "key": "DEMO",
                    "name": "Demo Project",
                    "description": "Demo project for AI Test Automation Platform",
                    "lead": "demo.lead@example.com",
                    "projectTypeKey": "software",
                    "avatarUrls": {
                        "16x16": "https://example.com/avatar.png",
                        "24x24": "https://example.com/avatar.png",
                        "32x32": "https://example.com/avatar.png",
                        "48x48": "https://example.com/avatar.png"
                    }
                },
                {
                    "id": "10001", 
                    "key": "MOBILE",
                    "name": "Mobile App",
                    "description": "Mobile application development",
                    "lead": "mobile.lead@example.com",
                    "projectTypeKey": "software",
                    "avatarUrls": {
                        "16x16": "https://example.com/mobile-avatar.png",
                        "24x24": "https://example.com/mobile-avatar.png",
                        "32x32": "https://example.com/mobile-avatar.png",
                        "48x48": "https://example.com/mobile-avatar.png"
                    }
                }
            ]
        
        return self._cached_projects
    
    def search_issues(self, jql: str = "", max_results: int = 50) -> Dict[str, Any]:
        """Search for Jira issues using JQL."""
        if not self.use_synthetic_data:
            return {"issues": [], "total": 0}
        
        # Generate synthetic tickets if not cached
        if self._cached_tickets is None:
            self._cached_tickets = self.synthetic_generator.generate_jira_tickets(100)
        
        # Simple JQL parsing for demo
        filtered_tickets = self._cached_tickets.copy()
        
        # Apply basic JQL filters
        if "type = Bug" in jql:
            filtered_tickets = [t for t in filtered_tickets if t["issue_type"] == "Bug"]
        elif "type = Story" in jql:
            filtered_tickets = [t for t in filtered_tickets if t["issue_type"] == "Story"]
        elif "priority = Critical" in jql:
            filtered_tickets = [t for t in filtered_tickets if t["priority"] == "Critical"]
        elif "status = Open" in jql:
            filtered_tickets = [t for t in filtered_tickets if t["status"] == "Open"]
        
        # Apply max_results limit
        filtered_tickets = filtered_tickets[:max_results]
        
        # Convert to Jira API format
        issues = []
        for ticket in filtered_tickets:
            issue = {
                "id": ticket["id"],
                "key": ticket["key"],
                "fields": {
                    "summary": ticket["summary"],
                    "description": ticket["description"],
                    "issuetype": {"name": ticket["issue_type"]},
                    "priority": {"name": ticket["priority"]},
                    "status": {"name": ticket["status"]},
                    "assignee": {"emailAddress": ticket["assignee"]},
                    "reporter": {"emailAddress": ticket["reporter"]},
                    "created": ticket["created"],
                    "updated": ticket["updated"],
                    "labels": ticket["labels"],
                    "components": [{"name": comp} for comp in ticket["components"]],
                    "fixVersions": [{"name": ticket["fix_version"]}] if ticket["fix_version"] else [],
                    "customfield_10001": ticket["story_points"]  # Story points field
                }
            }
            issues.append(issue)
        
        return {
            "issues": issues,
            "total": len(filtered_tickets),
            "maxResults": max_results,
            "startAt": 0
        }
    
    def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Get a specific Jira issue."""
        if not self.use_synthetic_data:
            return None
        
        # Generate synthetic tickets if not cached
        if self._cached_tickets is None:
            self._cached_tickets = self.synthetic_generator.generate_jira_tickets(100)
        
        # Find the issue
        for ticket in self._cached_tickets:
            if ticket["key"] == issue_key:
                return {
                    "id": ticket["id"],
                    "key": ticket["key"],
                    "fields": {
                        "summary": ticket["summary"],
                        "description": ticket["description"],
                        "issuetype": {"name": ticket["issue_type"]},
                        "priority": {"name": ticket["priority"]},
                        "status": {"name": ticket["status"]},
                        "assignee": {"emailAddress": ticket["assignee"]},
                        "reporter": {"emailAddress": ticket["reporter"]},
                        "created": ticket["created"],
                        "updated": ticket["updated"],
                        "labels": ticket["labels"],
                        "components": [{"name": comp} for comp in ticket["components"]],
                        "fixVersions": [{"name": ticket["fix_version"]}] if ticket["fix_version"] else [],
                        "customfield_10001": ticket["story_points"]
                    }
                }
        
        return None
    
    def get_issue_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Get comments for a Jira issue."""
        if not self.use_synthetic_data:
            return []
        
        # Generate some sample comments
        comments = [
            {
                "id": "10000",
                "author": {"emailAddress": "reviewer@example.com"},
                "body": "This looks good to me. Ready for testing.",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat()
            },
            {
                "id": "10001", 
                "author": {"emailAddress": "tester@example.com"},
                "body": "I've tested this feature and it works as expected. All test cases pass.",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat()
            }
        ]
        
        return comments
    
    def get_issue_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """Get available transitions for a Jira issue."""
        transitions = [
            {
                "id": "11",
                "name": "To Do",
                "to": {"name": "To Do"}
            },
            {
                "id": "21", 
                "name": "In Progress",
                "to": {"name": "In Progress"}
            },
            {
                "id": "31",
                "name": "Done",
                "to": {"name": "Done"}
            }
        ]
        
        return transitions
    
    def transition_issue(self, issue_key: str, transition_id: str, comment: str = "") -> bool:
        """Transition a Jira issue."""
        # Mock successful transition
        logger.info(f"Mock transition: {issue_key} -> transition {transition_id}")
        return True
    
    def create_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Jira issue."""
        # Mock issue creation
        new_issue = {
            "id": "100000",
            "key": f"DEMO-{len(self._cached_tickets or []) + 1000}",
            "fields": {
                "summary": issue_data.get("summary", "New Issue"),
                "description": issue_data.get("description", ""),
                "issuetype": {"name": issue_data.get("issue_type", "Task")},
                "priority": {"name": issue_data.get("priority", "Medium")},
                "project": {"key": "DEMO"}
            }
        }
        
        logger.info(f"Mock created issue: {new_issue['key']}")
        return new_issue
    
    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to a Jira issue."""
        new_comment = {
            "id": "10002",
            "author": {"emailAddress": "system@example.com"},
            "body": comment,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat()
        }
        
        logger.info(f"Mock added comment to {issue_key}")
        return new_comment
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        users = {
            "demo.user": {
                "accountId": "demo123",
                "displayName": "Demo User",
                "emailAddress": "demo.user@example.com",
                "active": True
            },
            "test.user": {
                "accountId": "test456", 
                "displayName": "Test User",
                "emailAddress": "test.user@example.com",
                "active": True
            }
        }
        
        return users.get(username)
    
    def get_fields(self) -> List[Dict[str, Any]]:
        """Get available Jira fields."""
        fields = [
            {
                "id": "summary",
                "name": "Summary",
                "custom": False,
                "schema": {"type": "string", "system": "summary"}
            },
            {
                "id": "description",
                "name": "Description", 
                "custom": False,
                "schema": {"type": "string", "system": "description"}
            },
            {
                "id": "issuetype",
                "name": "Issue Type",
                "custom": False,
                "schema": {"type": "issuetype", "system": "issuetype"}
            },
            {
                "id": "priority",
                "name": "Priority",
                "custom": False,
                "schema": {"type": "priority", "system": "priority"}
            },
            {
                "id": "status",
                "name": "Status",
                "custom": False,
                "schema": {"type": "status", "system": "status"}
            },
            {
                "id": "customfield_10001",
                "name": "Story Points",
                "custom": True,
                "schema": {"type": "number"}
            }
        ]
        
        return fields
    
    def get_issue_types(self) -> List[Dict[str, Any]]:
        """Get available issue types."""
        issue_types = [
            {"id": "1", "name": "Bug", "description": "A problem which impairs product functions"},
            {"id": "2", "name": "Story", "description": "A user story"},
            {"id": "3", "name": "Task", "description": "A task that needs to be done"},
            {"id": "4", "name": "Epic", "description": "A large body of work"},
            {"id": "5", "name": "Sub-task", "description": "A sub-task of a parent issue"}
        ]
        
        return issue_types
    
    def get_priorities(self) -> List[Dict[str, Any]]:
        """Get available priorities."""
        priorities = [
            {"id": "1", "name": "Critical", "description": "Critical priority"},
            {"id": "2", "name": "High", "description": "High priority"},
            {"id": "3", "name": "Medium", "description": "Medium priority"},
            {"id": "4", "name": "Low", "description": "Low priority"}
        ]
        
        return priorities
    
    def get_statuses(self) -> List[Dict[str, Any]]:
        """Get available statuses."""
        statuses = [
            {"id": "1", "name": "Open", "description": "Issue is open"},
            {"id": "2", "name": "In Progress", "description": "Issue is being worked on"},
            {"id": "3", "name": "Code Review", "description": "Code is under review"},
            {"id": "4", "name": "Testing", "description": "Issue is being tested"},
            {"id": "5", "name": "Done", "description": "Issue is complete"},
            {"id": "6", "name": "Closed", "description": "Issue is closed"}
        ]
        
        return statuses
    
    def get_components(self) -> List[Dict[str, Any]]:
        """Get available components."""
        components = [
            {"id": "10000", "name": "Authentication", "description": "User authentication system"},
            {"id": "10001", "name": "Payment", "description": "Payment processing system"},
            {"id": "10002", "name": "Notification", "description": "Notification service"},
            {"id": "10003", "name": "File Upload", "description": "File upload functionality"},
            {"id": "10004", "name": "Search", "description": "Search engine"},
            {"id": "10005", "name": "API", "description": "API endpoints"},
            {"id": "10006", "name": "Database", "description": "Database layer"},
            {"id": "10007", "name": "Cache", "description": "Caching service"}
        ]
        
        return components
    
    def search_issues_by_component(self, component: str) -> List[Dict[str, Any]]:
        """Search issues by component."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic tickets if not cached
        if self._cached_tickets is None:
            self._cached_tickets = self.synthetic_generator.generate_jira_tickets(100)
        
        # Filter by component
        filtered_tickets = [
            ticket for ticket in self._cached_tickets 
            if component.lower() in [comp.lower() for comp in ticket["components"]]
        ]
        
        return filtered_tickets
    
    def get_issue_statistics(self) -> Dict[str, Any]:
        """Get issue statistics for demo."""
        if not self.use_synthetic_data:
            return {}
        
        # Generate synthetic tickets if not cached
        if self._cached_tickets is None:
            self._cached_tickets = self.synthetic_generator.generate_jira_tickets(100)
        
        stats = {
            "total_issues": len(self._cached_tickets),
            "by_type": {},
            "by_priority": {},
            "by_status": {},
            "by_assignee": {},
            "created_last_30_days": 0,
            "resolved_last_30_days": 0
        }
        
        # Calculate statistics
        for ticket in self._cached_tickets:
            # By type
            stats["by_type"][ticket["issue_type"]] = stats["by_type"].get(ticket["issue_type"], 0) + 1
            
            # By priority
            stats["by_priority"][ticket["priority"]] = stats["by_priority"].get(ticket["priority"], 0) + 1
            
            # By status
            stats["by_status"][ticket["status"]] = stats["by_status"].get(ticket["status"], 0) + 1
            
            # By assignee
            stats["by_assignee"][ticket["assignee"]] = stats["by_assignee"].get(ticket["assignee"], 0) + 1
            
            # Recent activity (mock)
            if ticket["status"] in ["Done", "Closed"]:
                stats["resolved_last_30_days"] += 1
            stats["created_last_30_days"] += 1
        
        return stats
