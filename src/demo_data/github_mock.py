"""Mock GitHub API for demo mode."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class GitHubMockData:
    """Mock GitHub API for demo mode testing."""
    
    def __init__(self, use_synthetic_data: bool = True):
        """Initialize GitHub mock data."""
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_generator = SyntheticDataGenerator(seed=42) if use_synthetic_data else None
        
        # Cache for demo data
        self._cached_events = None
        self._cached_repositories = None
        
        logger.info("GitHubMockData initialized for demo mode")
    
    def get_repositories(self) -> List[Dict[str, Any]]:
        """Get repositories from GitHub."""
        if self._cached_repositories is None:
            self._cached_repositories = [
                {
                    "id": 1,
                    "name": "demo-backend",
                    "full_name": "demo-org/demo-backend",
                    "description": "Demo backend application for AI Test Automation Platform",
                    "private": False,
                    "html_url": "https://github.com/demo-org/demo-backend",
                    "clone_url": "https://github.com/demo-org/demo-backend.git",
                    "created_at": (datetime.now() - timedelta(days=365)).isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "pushed_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "language": "Python",
                    "stargazers_count": random.randint(10, 100),
                    "forks_count": random.randint(5, 50),
                    "open_issues_count": random.randint(0, 20),
                    "default_branch": "main",
                    "topics": ["python", "backend", "api", "testing"],
                    "owner": {
                        "login": "demo-org",
                        "id": 12345,
                        "type": "Organization"
                    }
                },
                {
                    "id": 2,
                    "name": "demo-frontend",
                    "full_name": "demo-org/demo-frontend",
                    "description": "Demo frontend application",
                    "private": False,
                    "html_url": "https://github.com/demo-org/demo-frontend",
                    "clone_url": "https://github.com/demo-org/demo-frontend.git",
                    "created_at": (datetime.now() - timedelta(days=300)).isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "pushed_at": (datetime.now() - timedelta(hours=5)).isoformat(),
                    "language": "JavaScript",
                    "stargazers_count": random.randint(15, 80),
                    "forks_count": random.randint(3, 30),
                    "open_issues_count": random.randint(0, 15),
                    "default_branch": "main",
                    "topics": ["javascript", "react", "frontend", "ui"],
                    "owner": {
                        "login": "demo-org",
                        "id": 12345,
                        "type": "Organization"
                    }
                },
                {
                    "id": 3,
                    "name": "demo-mobile",
                    "full_name": "demo-org/demo-mobile",
                    "description": "Demo mobile application",
                    "private": False,
                    "html_url": "https://github.com/demo-org/demo-mobile",
                    "clone_url": "https://github.com/demo-org/demo-mobile.git",
                    "created_at": (datetime.now() - timedelta(days=200)).isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "pushed_at": (datetime.now() - timedelta(hours=1)).isoformat(),
                    "language": "TypeScript",
                    "stargazers_count": random.randint(8, 60),
                    "forks_count": random.randint(2, 25),
                    "open_issues_count": random.randint(0, 10),
                    "default_branch": "main",
                    "topics": ["typescript", "react-native", "mobile", "ios", "android"],
                    "owner": {
                        "login": "demo-org",
                        "id": 12345,
                        "type": "Organization"
                    }
                }
            ]
        
        return self._cached_repositories
    
    def get_commits(self, 
                   owner: str,
                   repo: str,
                   branch: str = "main",
                   since: datetime = None,
                   until: datetime = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get commits from a repository."""
        if not self.use_synthetic_data:
            return []
        
        if since is None:
            since = datetime.now() - timedelta(days=30)
        if until is None:
            until = datetime.now()
        
        # Generate synthetic commits
        commits = []
        
        for i in range(min(limit, 50)):
            commit_time = since + timedelta(days=random.randint(0, int((until - since).total_seconds() / 86400)))
            
            commit = {
                "sha": f"demo{random.randint(100000, 999999)}",
                "commit": {
                    "message": self._generate_commit_message(),
                    "author": {
                        "name": random.choice(["John Doe", "Jane Smith", "Mike Wilson", "Sarah Jones"]),
                        "email": random.choice(["john@example.com", "jane@example.com", "mike@example.com", "sarah@example.com"]),
                        "date": commit_time.isoformat()
                    },
                    "committer": {
                        "name": random.choice(["John Doe", "Jane Smith", "Mike Wilson", "Sarah Jones"]),
                        "email": random.choice(["john@example.com", "jane@example.com", "mike@example.com", "sarah@example.com"]),
                        "date": commit_time.isoformat()
                    }
                },
                "author": {
                    "login": random.choice(["johndoe", "janesmith", "mikewilson", "sarahjones"]),
                    "id": random.randint(1000, 9999),
                    "avatar_url": f"https://github.com/identicons/{random.choice(['johndoe', 'janesmith'])}.png"
                },
                "committer": {
                    "login": random.choice(["johndoe", "janesmith", "mikewilson", "sarahjones"]),
                    "id": random.randint(1000, 9999),
                    "avatar_url": f"https://github.com/identicons/{random.choice(['johndoe', 'janesmith'])}.png"
                },
                "html_url": f"https://github.com/{owner}/{repo}/commit/demo{random.randint(100000, 999999)}",
                "stats": {
                    "additions": random.randint(1, 100),
                    "deletions": random.randint(0, 50),
                    "total": random.randint(1, 150)
                },
                "files": self._generate_commit_files()
            }
            
            commits.append(commit)
        
        return commits
    
    def get_pull_requests(self,
                         owner: str,
                         repo: str,
                         state: str = "all",
                         limit: int = 50) -> List[Dict[str, Any]]:
        """Get pull requests from a repository."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic pull requests
        pull_requests = []
        
        for i in range(min(limit, 30)):
            created_at = datetime.now() - timedelta(days=random.randint(1, 90))
            updated_at = created_at + timedelta(days=random.randint(0, 7))
            
            pr_state = random.choice(["open", "closed", "merged"]) if state == "all" else state
            
            pr = {
                "id": random.randint(1000, 9999),
                "number": random.randint(1, 500),
                "title": self._generate_pr_title(),
                "body": self._generate_pr_description(),
                "state": pr_state,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "closed_at": (updated_at + timedelta(days=1)).isoformat() if pr_state in ["closed", "merged"] else None,
                "merged_at": (updated_at + timedelta(days=1)).isoformat() if pr_state == "merged" else None,
                "html_url": f"https://github.com/{owner}/{repo}/pull/{random.randint(1, 500)}",
                "diff_url": f"https://github.com/{owner}/{repo}/pull/{random.randint(1, 500)}.diff",
                "patch_url": f"https://github.com/{owner}/{repo}/pull/{random.randint(1, 500)}.patch",
                "user": {
                    "login": random.choice(["johndoe", "janesmith", "mikewilson", "sarahjones"]),
                    "id": random.randint(1000, 9999),
                    "avatar_url": f"https://github.com/identicons/{random.choice(['johndoe', 'janesmith'])}.png"
                },
                "head": {
                    "ref": f"feature/{random.choice(['auth', 'payment', 'notification', 'ui'])}-{random.randint(1, 10)}",
                    "sha": f"head{random.randint(100000, 999999)}",
                    "repo": {
                        "name": repo,
                        "full_name": f"{owner}/{repo}",
                        "html_url": f"https://github.com/{owner}/{repo}"
                    }
                },
                "base": {
                    "ref": "main",
                    "sha": f"base{random.randint(100000, 999999)}",
                    "repo": {
                        "name": repo,
                        "full_name": f"{owner}/{repo}",
                        "html_url": f"https://github.com/{owner}/{repo}"
                    }
                },
                "additions": random.randint(10, 500),
                "deletions": random.randint(0, 200),
                "changed_files": random.randint(1, 20),
                "commits": random.randint(1, 10),
                "comments": random.randint(0, 10),
                "review_comments": random.randint(0, 20),
                "labels": self._generate_pr_labels(),
                "assignees": self._generate_pr_assignees(),
                "requested_reviewers": self._generate_pr_reviewers()
            }
            
            pull_requests.append(pr)
        
        return pull_requests
    
    def get_issues(self,
                  owner: str,
                  repo: str,
                  state: str = "all",
                  limit: int = 50) -> List[Dict[str, Any]]:
        """Get issues from a repository."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic issues
        issues = []
        
        for i in range(min(limit, 30)):
            created_at = datetime.now() - timedelta(days=random.randint(1, 180))
            updated_at = created_at + timedelta(days=random.randint(0, 30))
            
            issue_state = random.choice(["open", "closed"]) if state == "all" else state
            
            issue = {
                "id": random.randint(10000, 99999),
                "number": random.randint(1, 1000),
                "title": self._generate_issue_title(),
                "body": self._generate_issue_body(),
                "state": issue_state,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "closed_at": (updated_at + timedelta(days=1)).isoformat() if issue_state == "closed" else None,
                "html_url": f"https://github.com/{owner}/{repo}/issues/{random.randint(1, 1000)}",
                "user": {
                    "login": random.choice(["johndoe", "janesmith", "mikewilson", "sarahjones"]),
                    "id": random.randint(1000, 9999),
                    "avatar_url": f"https://github.com/identicons/{random.choice(['johndoe', 'janesmith'])}.png"
                },
                "labels": self._generate_issue_labels(),
                "assignees": self._generate_issue_assignees(),
                "comments": random.randint(0, 10),
                "pull_request": None  # Not a PR
            }
            
            issues.append(issue)
        
        return issues
    
    def get_releases(self,
                    owner: str,
                    repo: str,
                    limit: int = 20) -> List[Dict[str, Any]]:
        """Get releases from a repository."""
        releases = []
        
        for i in range(min(limit, 10)):
            published_at = datetime.now() - timedelta(days=random.randint(1, 365))
            
            release = {
                "id": random.randint(10000, 99999),
                "tag_name": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "name": f"Release {random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "body": self._generate_release_body(),
                "draft": random.choice([True, False]),
                "prerelease": random.choice([True, False]),
                "created_at": published_at.isoformat(),
                "published_at": published_at.isoformat(),
                "html_url": f"https://github.com/{owner}/{repo}/releases/tag/v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "author": {
                    "login": random.choice(["johndoe", "janesmith", "mikewilson", "sarahjones"]),
                    "id": random.randint(1000, 9999),
                    "avatar_url": f"https://github.com/identicons/{random.choice(['johndoe', 'janesmith'])}.png"
                },
                "assets": self._generate_release_assets()
            }
            
            releases.append(release)
        
        return releases
    
    def get_workflow_runs(self,
                         owner: str,
                         repo: str,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Get workflow runs from a repository."""
        workflow_runs = []
        
        for i in range(min(limit, 20)):
            created_at = datetime.now() - timedelta(hours=random.randint(1, 168))
            updated_at = created_at + timedelta(minutes=random.randint(1, 60))
            
            workflow_run = {
                "id": random.randint(100000, 999999),
                "name": random.choice([
                    "CI/CD Pipeline",
                    "Test Suite",
                    "Build and Deploy",
                    "Security Scan",
                    "Performance Tests"
                ]),
                "status": random.choice(["completed", "in_progress", "queued", "failed"]),
                "conclusion": random.choice(["success", "failure", "cancelled", "skipped"]) if random.choice([True, False]) else None,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "run_started_at": created_at.isoformat(),
                "html_url": f"https://github.com/{owner}/{repo}/actions/runs/{random.randint(100000, 999999)}",
                "head_branch": random.choice(["main", "develop", f"feature-{random.randint(1, 10)}"]),
                "head_sha": f"sha{random.randint(100000, 999999)}",
                "event": random.choice(["push", "pull_request", "schedule", "workflow_dispatch"]),
                "jobs": self._generate_workflow_jobs()
            }
            
            workflow_runs.append(workflow_run)
        
        return workflow_runs
    
    def get_branches(self,
                    owner: str,
                    repo: str,
                    limit: int = 20) -> List[Dict[str, Any]]:
        """Get branches from a repository."""
        branches = [
            {
                "name": "main",
                "commit": {
                    "sha": f"main{random.randint(100000, 999999)}",
                    "url": f"https://github.com/{owner}/{repo}/commits/main{random.randint(100000, 999999)}"
                },
                "protected": True
            },
            {
                "name": "develop",
                "commit": {
                    "sha": f"develop{random.randint(100000, 999999)}",
                    "url": f"https://github.com/{owner}/{repo}/commits/develop{random.randint(100000, 999999)}"
                },
                "protected": False
            }
        ]
        
        # Add feature branches
        for i in range(min(limit - 2, 10)):
            branch = {
                "name": f"feature/{random.choice(['auth', 'payment', 'notification', 'ui'])}-{random.randint(1, 10)}",
                "commit": {
                    "sha": f"feature{random.randint(100000, 999999)}",
                    "url": f"https://github.com/{owner}/{repo}/commits/feature{random.randint(100000, 999999)}"
                },
                "protected": False
            }
            branches.append(branch)
        
        return branches
    
    # Helper methods
    
    def _generate_commit_message(self) -> str:
        """Generate realistic commit message."""
        messages = [
            "Fix authentication bug in login flow",
            "Add new payment processing feature",
            "Improve error handling for API endpoints",
            "Update dependencies to latest versions",
            "Refactor user service for better performance",
            "Add comprehensive test coverage",
            "Fix mobile responsive layout issues",
            "Implement caching mechanism for better performance",
            "Update API documentation",
            "Fix security vulnerability in input validation",
            "Add dark mode support",
            "Optimize database queries",
            "Fix memory leak in notification service",
            "Add logging for better debugging",
            "Implement rate limiting for API calls"
        ]
        return random.choice(messages)
    
    def _generate_commit_files(self) -> List[Dict[str, Any]]:
        """Generate commit files."""
        files = []
        file_types = [".py", ".js", ".ts", ".json", ".md", ".yml", ".yaml"]
        
        for i in range(random.randint(1, 5)):
            file = {
                "filename": f"src/{random.choice(['auth', 'payment', 'notification', 'api'])}/{random.choice(['service', 'controller', 'model', 'util'])}{random.choice(file_types)}",
                "additions": random.randint(1, 50),
                "deletions": random.randint(0, 20),
                "changes": random.randint(1, 70),
                "status": random.choice(["added", "modified", "removed", "renamed"])
            }
            files.append(file)
        
        return files
    
    def _generate_pr_title(self) -> str:
        """Generate PR title."""
        titles = [
            "Add user authentication feature",
            "Fix payment processing bug",
            "Improve API performance",
            "Update mobile UI components",
            "Add comprehensive test coverage",
            "Implement caching strategy",
            "Refactor database layer",
            "Enhance error handling",
            "Add dark mode support",
            "Optimize memory usage",
            "Fix security vulnerabilities",
            "Update documentation",
            "Add logging improvements",
            "Implement rate limiting",
            "Fix mobile responsive issues"
        ]
        return random.choice(titles)
    
    def _generate_pr_description(self) -> str:
        """Generate PR description."""
        return f"""
## Description
This PR {random.choice(['implements', 'fixes', 'improves', 'refactors'])} the {random.choice(['authentication', 'payment processing', 'API performance', 'UI components'])} functionality.

## Changes
- {random.choice(['Added new features', 'Fixed bugs', 'Improved performance', 'Updated documentation'])}
- {random.choice(['Enhanced security', 'Optimized queries', 'Improved UI', 'Added tests'])}

## Testing
- [ ] All existing tests pass
- [ ] New tests added for the changes
- [ ] Manual testing completed
- [ ] Performance testing done

## Screenshots (if applicable)
<!-- Add screenshots here -->

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes introduced
        """.strip()
    
    def _generate_pr_labels(self) -> List[Dict[str, Any]]:
        """Generate PR labels."""
        labels = [
            {"name": "enhancement", "color": "a2eeef"},
            {"name": "bug", "color": "d73a4a"},
            {"name": "documentation", "color": "0075ca"},
            {"name": "performance", "color": "7057ff"},
            {"name": "security", "color": "e99695"}
        ]
        
        return random.sample(labels, random.randint(1, 3))
    
    def _generate_pr_assignees(self) -> List[Dict[str, Any]]:
        """Generate PR assignees."""
        assignees = [
            {
                "login": "johndoe",
                "id": 1234,
                "avatar_url": "https://github.com/identicons/johndoe.png"
            },
            {
                "login": "janesmith", 
                "id": 5678,
                "avatar_url": "https://github.com/identicons/janesmith.png"
            }
        ]
        
        return random.sample(assignees, random.randint(0, 2))
    
    def _generate_pr_reviewers(self) -> List[Dict[str, Any]]:
        """Generate PR reviewers."""
        reviewers = [
            {
                "login": "mikewilson",
                "id": 9012,
                "avatar_url": "https://github.com/identicons/mikewilson.png"
            },
            {
                "login": "sarahjones",
                "id": 3456,
                "avatar_url": "https://github.com/identicons/sarahjones.png"
            }
        ]
        
        return random.sample(reviewers, random.randint(1, 2))
    
    def _generate_issue_title(self) -> str:
        """Generate issue title."""
        titles = [
            "Bug: Login fails on mobile devices",
            "Feature: Add dark mode support",
            "Enhancement: Improve search performance",
            "Bug: File upload timeout issue",
            "Feature: Add user preferences",
            "Enhancement: Better error messages",
            "Bug: API rate limiting problems",
            "Feature: Add export functionality",
            "Bug: Memory leak in notification service",
            "Enhancement: Add comprehensive logging"
        ]
        return random.choice(titles)
    
    def _generate_issue_body(self) -> str:
        """Generate issue body."""
        return f"""
## Bug Description
{random.choice([
    "Users are experiencing login failures on mobile devices",
    "File uploads are timing out after 30 seconds",
    "API rate limiting is not working correctly",
    "Memory usage increases over time without garbage collection"
])}

## Steps to Reproduce
1. Navigate to the application
2. Perform the following actions:
   - {random.choice(['Try to login', 'Upload a file', 'Make API calls', 'Use the application for extended time'])}
   - Wait for the issue to occur

## Expected Behavior
The system should {random.choice(['allow successful login', 'complete file upload', 'respect rate limits', 'maintain stable memory usage'])}

## Actual Behavior
The system {random.choice(['shows login error', 'times out during upload', 'allows unlimited requests', 'memory usage keeps increasing'])}

## Environment
- Browser: {random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'])}
- Device: {random.choice(['iPhone', 'Android', 'Desktop'])}
- Version: v{random.randint(1, 3)}.{random.randint(0, 9)}

## Additional Context
This issue affects {random.choice(['all users', 'some users', 'mobile users only', 'users with large files'])} and has a {random.choice(['high', 'medium', 'low'])} impact.
        """.strip()
    
    def _generate_issue_labels(self) -> List[Dict[str, Any]]:
        """Generate issue labels."""
        labels = [
            {"name": "bug", "color": "d73a4a"},
            {"name": "enhancement", "color": "a2eeef"},
            {"name": "feature", "color": "7057ff"},
            {"name": "documentation", "color": "0075ca"},
            {"name": "good first issue", "color": "7057ff"},
            {"name": "help wanted", "color": "008672"}
        ]
        
        return random.sample(labels, random.randint(1, 3))
    
    def _generate_issue_assignees(self) -> List[Dict[str, Any]]:
        """Generate issue assignees."""
        assignees = [
            {
                "login": "johndoe",
                "id": 1234,
                "avatar_url": "https://github.com/identicons/johndoe.png"
            }
        ]
        
        return random.sample(assignees, random.randint(0, 1))
    
    def _generate_release_body(self) -> str:
        """Generate release body."""
        return f"""
## What's New in {random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}

### Features
- {random.choice(['New user authentication system', 'Enhanced payment processing', 'Improved mobile experience'])}
- {random.choice(['Advanced search functionality', 'Better error handling', 'Performance optimizations'])}

### Bug Fixes
- Fixed login issues on mobile devices
- Resolved file upload timeout problems
- Corrected API rate limiting behavior
- Fixed memory leak in notification service

### Improvements
- Enhanced security measures
- Optimized database queries
- Improved user interface
- Added comprehensive logging

### Breaking Changes
{random.choice(['None', 'API endpoint changes', 'Database schema updates', 'Configuration file format changes'])}

## Installation
```bash
pip install demo-app=={random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}
```

## Migration Guide
{random.choice(['No migration required', 'See migration guide in documentation', 'Update configuration files', 'Database migration required'])}
        """.strip()
    
    def _generate_release_assets(self) -> List[Dict[str, Any]]:
        """Generate release assets."""
        assets = [
            {
                "name": f"demo-app-{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}.tar.gz",
                "size": random.randint(1000000, 10000000),
                "download_count": random.randint(10, 1000),
                "content_type": "application/gzip"
            },
            {
                "name": f"demo-app-{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}.zip",
                "size": random.randint(1000000, 15000000),
                "download_count": random.randint(5, 500),
                "content_type": "application/zip"
            }
        ]
        
        return random.sample(assets, random.randint(0, 2))
    
    def _generate_workflow_jobs(self) -> List[Dict[str, Any]]:
        """Generate workflow jobs."""
        jobs = [
            {
                "id": random.randint(100000, 999999),
                "name": "test",
                "status": random.choice(["completed", "in_progress", "queued", "failed"]),
                "conclusion": random.choice(["success", "failure", "cancelled"]) if random.choice([True, False]) else None,
                "steps": [
                    {
                        "name": "Checkout code",
                        "status": "completed",
                        "conclusion": "success"
                    },
                    {
                        "name": "Setup Python",
                        "status": "completed", 
                        "conclusion": "success"
                    },
                    {
                        "name": "Install dependencies",
                        "status": "completed",
                        "conclusion": "success"
                    },
                    {
                        "name": "Run tests",
                        "status": random.choice(["completed", "in_progress", "failed"]),
                        "conclusion": random.choice(["success", "failure"]) if random.choice([True, False]) else None
                    }
                ]
            }
        ]
        
        return jobs
