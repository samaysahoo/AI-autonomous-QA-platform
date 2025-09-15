"""Jira data ingestion module for pulling specs, user stories, and bug tickets."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from jira import JIRA
from dataclasses import dataclass

from config.settings import get_settings
from ..demo_data.demo_mode import DemoMode

logger = logging.getLogger(__name__)


@dataclass
class JiraTicket:
    """Represents a Jira ticket with relevant metadata."""
    key: str
    summary: str
    description: str
    issue_type: str
    status: str
    priority: str
    assignee: Optional[str]
    created: datetime
    updated: datetime
    labels: List[str]
    components: List[str]
    epic_link: Optional[str]
    story_points: Optional[int]


class JiraIngestor:
    """Handles data ingestion from Jira API."""
    
    def __init__(self, demo_mode: bool = False):
        self.settings = get_settings()
        self.demo_mode = demo_mode
        self.demo_mode_manager = DemoMode() if demo_mode else None
        
        if not demo_mode:
            self.jira = JIRA(
                server=self.settings.jira_url,
                basic_auth=(self.settings.jira_username, self.settings.jira_api_token)
            )
        else:
            self.jira = None
    
    def fetch_tickets_by_type(self, issue_types: List[str], days_back: int = 30) -> List[JiraTicket]:
        """Fetch tickets of specific types from the last N days."""
        if self.demo_mode:
            return self._fetch_demo_tickets_by_type(issue_types, days_back)
        
        jql_query = self._build_jql_query(issue_types, days_back)
        
        try:
            issues = self.jira.search_issues(
                jql_query,
                expand=['changelog'],
                maxResults=1000
            )
            
            tickets = []
            for issue in issues:
                ticket = self._parse_jira_issue(issue)
                tickets.append(ticket)
            
            logger.info(f"Fetched {len(tickets)} tickets of types: {issue_types}")
            return tickets
            
        except Exception as e:
            logger.error(f"Error fetching Jira tickets: {e}")
            return []
    
    def fetch_specifications(self, days_back: int = 90) -> List[JiraTicket]:
        """Fetch specification tickets (typically Epic, Story, or specific labels)."""
        spec_types = ['Epic', 'Story', 'Requirement']
        spec_labels = ['spec', 'specification', 'requirement']
        
        tickets = self.fetch_tickets_by_type(spec_types, days_back)
        
        # Filter by labels if needed
        spec_tickets = [
            ticket for ticket in tickets 
            if any(label.lower() in spec_labels for label in ticket.labels)
        ]
        
        return spec_tickets
    
    def fetch_user_stories(self, days_back: int = 60) -> List[JiraTicket]:
        """Fetch user story tickets."""
        story_types = ['Story', 'User Story']
        tickets = self.fetch_tickets_by_type(story_types, days_back)
        
        # Filter for user stories specifically
        user_stories = [
            ticket for ticket in tickets
            if 'user story' in ticket.description.lower() or 
               'as a' in ticket.description.lower()
        ]
        
        return user_stories
    
    def fetch_bug_tickets(self, days_back: int = 30) -> List[JiraTicket]:
        """Fetch bug tickets for test case generation."""
        bug_types = ['Bug', 'Defect']
        tickets = self.fetch_tickets_by_type(bug_types, days_back)
        
        # Prioritize high-priority bugs
        high_priority_bugs = [
            ticket for ticket in tickets
            if ticket.priority in ['Highest', 'High', 'Critical']
        ]
        
        return high_priority_bugs
    
    def _build_jql_query(self, issue_types: List[str], days_back: int) -> str:
        """Build JQL query for fetching tickets."""
        type_filter = ' OR '.join([f'issuetype = "{t}"' for t in issue_types])
        date_filter = f'updated >= -{days_back}d'
        
        return f'({type_filter}) AND {date_filter} ORDER BY updated DESC'
    
    def _parse_jira_issue(self, issue) -> JiraTicket:
        """Parse a Jira issue into a JiraTicket object."""
        return JiraTicket(
            key=issue.key,
            summary=issue.fields.summary or "",
            description=issue.fields.description or "",
            issue_type=issue.fields.issuetype.name,
            status=issue.fields.status.name,
            priority=issue.fields.priority.name if issue.fields.priority else "None",
            assignee=issue.fields.assignee.displayName if issue.fields.assignee else None,
            created=datetime.strptime(issue.fields.created, "%Y-%m-%dT%H:%M:%S.%f%z"),
            updated=datetime.strptime(issue.fields.updated, "%Y-%m-%dT%H:%M:%S.%f%z"),
            labels=issue.fields.labels or [],
            components=[comp.name for comp in (issue.fields.components or [])],
            epic_link=issue.fields.customfield_10014 if hasattr(issue.fields, 'customfield_10014') else None,
            story_points=issue.fields.customfield_10016 if hasattr(issue.fields, 'customfield_10016') else None
        )
    
    def get_ticket_content_for_indexing(self, ticket: JiraTicket) -> Dict[str, Any]:
        """Convert ticket to format suitable for vector store indexing."""
        content = f"""
        Title: {ticket.summary}
        Type: {ticket.issue_type}
        Status: {ticket.status}
        Priority: {ticket.priority}
        Description: {ticket.description}
        Components: {', '.join(ticket.components)}
        Labels: {', '.join(ticket.labels)}
        """
        
        return {
            "id": ticket.key,
            "content": content.strip(),
            "metadata": {
                "type": ticket.issue_type,
                "status": ticket.status,
                "priority": ticket.priority,
                "assignee": ticket.assignee,
                "created": ticket.created.isoformat(),
                "updated": ticket.updated.isoformat(),
                "labels": ticket.labels,
                "components": ticket.components,
                "epic_link": ticket.epic_link,
                "story_points": ticket.story_points,
                "source": "jira"
            }
        }
    
    def _fetch_demo_tickets_by_type(self, issue_types: List[str], days_back: int = 30) -> List[JiraTicket]:
        """Fetch demo tickets by type."""
        if not self.demo_mode_manager:
            return []
        
        # Get demo data
        demo_data = self.demo_mode_manager.get_jira_demo_data()
        demo_tickets = demo_data.get("tickets", [])
        
        # Filter by issue types
        filtered_tickets = [
            ticket for ticket in demo_tickets 
            if ticket.get("issue_type") in issue_types
        ]
        
        # Convert to JiraTicket objects
        tickets = []
        for demo_ticket in filtered_tickets:
            ticket = JiraTicket(
                key=demo_ticket["key"],
                summary=demo_ticket["summary"],
                description=demo_ticket["description"],
                issue_type=demo_ticket["issue_type"],
                status=demo_ticket["status"],
                priority=demo_ticket["priority"],
                assignee=demo_ticket["assignee"],
                reporter=demo_ticket["reporter"],
                created=datetime.fromisoformat(demo_ticket["created"]),
                updated=datetime.fromisoformat(demo_ticket["updated"]),
                labels=demo_ticket.get("labels", []),
                components=demo_ticket.get("components", []),
                epic_link=demo_ticket.get("epic_link"),
                story_points=demo_ticket.get("story_points")
            )
            tickets.append(ticket)
        
        logger.info(f"[DEMO MODE] Fetched {len(tickets)} demo tickets of types: {issue_types}")
        return tickets
