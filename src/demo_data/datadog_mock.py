"""Mock Datadog API for demo mode."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class DatadogMockData:
    """Mock Datadog API for demo mode testing."""
    
    def __init__(self, use_synthetic_data: bool = True):
        """Initialize Datadog mock data."""
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_generator = SyntheticDataGenerator(seed=42) if use_synthetic_data else None
        
        # Cache for demo data
        self._cached_metrics = None
        self._cached_logs = None
        self._cached_events = None
        
        logger.info("DatadogMockData initialized for demo mode")
    
    def get_metrics(self, 
                   metric_name: str,
                   start_time: datetime,
                   end_time: datetime,
                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metrics from Datadog."""
        if not self.use_synthetic_data:
            return []
        
        # Generate synthetic metrics if not cached
        if self._cached_metrics is None:
            self._cached_metrics = self.synthetic_generator.generate_datadog_metrics(2000)
        
        # Filter metrics by name and time range
        filtered_metrics = []
        for metric in self._cached_metrics:
            if metric["metric_name"] == metric_name:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if start_time <= metric_time <= end_time:
                    # Apply tag filters
                    if tags:
                        metric_tags = metric.get("tags", {})
                        if any(tag in str(metric_tags.values()) for tag in tags):
                            filtered_metrics.append(metric)
                    else:
                        filtered_metrics.append(metric)
        
        return filtered_metrics
    
    def get_logs(self,
                query: str = "",
                start_time: datetime = None,
                end_time: datetime = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs from Datadog."""
        if not self.use_synthetic_data:
            return []
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        # Generate synthetic logs
        logs = []
        log_levels = ["ERROR", "WARN", "INFO", "DEBUG"]
        services = ["auth-service", "payment-service", "notification-service", "api-gateway"]
        
        for i in range(min(limit, 200)):
            log_time = start_time + timedelta(minutes=random.randint(0, int((end_time - start_time).total_seconds() / 60)))
            
            log = {
                "timestamp": log_time.isoformat(),
                "level": random.choice(log_levels),
                "service": random.choice(services),
                "message": self._generate_log_message(),
                "tags": {
                    "environment": random.choice(["production", "staging", "development"]),
                    "host": f"host-{random.randint(1, 100)}",
                    "version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}"
                },
                "attributes": {
                    "request_id": f"req-{random.randint(10000, 99999)}",
                    "user_id": random.randint(1000, 9999),
                    "response_time": random.uniform(0.001, 5.0),
                    "status_code": random.choice([200, 201, 400, 401, 403, 404, 500])
                }
            }
            
            logs.append(log)
        
        # Apply query filters
        if query:
            filtered_logs = []
            for log in logs:
                if any(term.lower() in log["message"].lower() for term in query.split()):
                    filtered_logs.append(log)
            logs = filtered_logs
        
        return logs[:limit]
    
    def get_events(self,
                  start_time: datetime = None,
                  end_time: datetime = None,
                  tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get events from Datadog."""
        if not self.use_synthetic_data:
            return []
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        # Generate synthetic events
        events = []
        event_types = ["deployment", "alert", "custom", "error", "performance"]
        
        for i in range(50):
            event_time = start_time + timedelta(minutes=random.randint(0, int((end_time - start_time).total_seconds() / 60)))
            
            event = {
                "id": f"event-{random.randint(10000, 99999)}",
                "timestamp": event_time.isoformat(),
                "title": self._generate_event_title(),
                "text": self._generate_event_description(),
                "alert_type": random.choice(["info", "warning", "error", "success"]),
                "event_type": random.choice(event_types),
                "priority": random.choice(["normal", "low", "high"]),
                "tags": {
                    "environment": random.choice(["production", "staging", "development"]),
                    "service": random.choice(["auth-service", "payment-service", "api-gateway"]),
                    "team": random.choice(["backend", "frontend", "mobile", "devops"])
                },
                "source": "demo-system"
            }
            
            events.append(event)
        
        return events
    
    def get_dashboards(self) -> List[Dict[str, Any]]:
        """Get available dashboards."""
        dashboards = [
            {
                "id": "demo-dashboard-1",
                "title": "Application Performance",
                "description": "Overview of application performance metrics",
                "created": (datetime.now() - timedelta(days=30)).isoformat(),
                "modified": datetime.now().isoformat(),
                "author": "demo.user@example.com",
                "url": "https://app.datadoghq.com/dashboard/demo-dashboard-1"
            },
            {
                "id": "demo-dashboard-2",
                "title": "Error Monitoring",
                "description": "Error rates and exception tracking",
                "created": (datetime.now() - timedelta(days=15)).isoformat(),
                "modified": datetime.now().isoformat(),
                "author": "demo.user@example.com",
                "url": "https://app.datadoghq.com/dashboard/demo-dashboard-2"
            },
            {
                "id": "demo-dashboard-3",
                "title": "Infrastructure Metrics",
                "description": "Server and infrastructure monitoring",
                "created": (datetime.now() - timedelta(days=60)).isoformat(),
                "modified": datetime.now().isoformat(),
                "author": "demo.user@example.com",
                "url": "https://app.datadoghq.com/dashboard/demo-dashboard-3"
            }
        ]
        
        return dashboards
    
    def get_monitors(self) -> List[Dict[str, Any]]:
        """Get available monitors."""
        monitors = [
            {
                "id": "monitor-1",
                "name": "High Error Rate",
                "type": "metric alert",
                "query": "avg(last_5m):avg:application.errors.count{*} > 10",
                "message": "Error rate is above threshold",
                "options": {
                    "thresholds": {"critical": 10, "warning": 5},
                    "notify_no_data": False,
                    "timeout_h": 0
                },
                "overall_state": random.choice(["OK", "WARN", "CRITICAL", "NO DATA"]),
                "created": (datetime.now() - timedelta(days=45)).isoformat(),
                "modified": datetime.now().isoformat()
            },
            {
                "id": "monitor-2",
                "name": "Response Time Alert",
                "type": "metric alert",
                "query": "avg(last_10m):avg:application.requests.duration{*} > 2",
                "message": "Response time is too high",
                "options": {
                    "thresholds": {"critical": 2, "warning": 1},
                    "notify_no_data": False,
                    "timeout_h": 0
                },
                "overall_state": random.choice(["OK", "WARN", "CRITICAL"]),
                "created": (datetime.now() - timedelta(days=30)).isoformat(),
                "modified": datetime.now().isoformat()
            },
            {
                "id": "monitor-3",
                "name": "Memory Usage",
                "type": "metric alert",
                "query": "avg(last_5m):avg:application.memory.usage{*} > 85",
                "message": "Memory usage is high",
                "options": {
                    "thresholds": {"critical": 85, "warning": 75},
                    "notify_no_data": False,
                    "timeout_h": 0
                },
                "overall_state": random.choice(["OK", "WARN", "CRITICAL"]),
                "created": (datetime.now() - timedelta(days=20)).isoformat(),
                "modified": datetime.now().isoformat()
            }
        ]
        
        return monitors
    
    def get_alert_history(self, monitor_id: str) -> List[Dict[str, Any]]:
        """Get alert history for a monitor."""
        alerts = []
        
        for i in range(random.randint(5, 20)):
            alert_time = datetime.now() - timedelta(hours=random.randint(1, 168))
            
            alert = {
                "id": f"alert-{random.randint(10000, 99999)}",
                "monitor_id": monitor_id,
                "timestamp": alert_time.isoformat(),
                "state": random.choice(["OK", "WARN", "CRITICAL", "NO DATA"]),
                "message": "Monitor threshold exceeded",
                "tags": {
                    "environment": random.choice(["production", "staging"]),
                    "service": random.choice(["auth-service", "payment-service", "api-gateway"])
                }
            }
            
            alerts.append(alert)
        
        return alerts
    
    def get_service_map(self) -> Dict[str, Any]:
        """Get service dependency map."""
        services = [
            {"name": "frontend", "type": "web", "dependencies": ["api-gateway"]},
            {"name": "api-gateway", "type": "gateway", "dependencies": ["auth-service", "payment-service", "notification-service"]},
            {"name": "auth-service", "type": "service", "dependencies": ["database"]},
            {"name": "payment-service", "type": "service", "dependencies": ["database", "external-payment-api"]},
            {"name": "notification-service", "type": "service", "dependencies": ["database", "email-service"]},
            {"name": "database", "type": "database", "dependencies": []},
            {"name": "external-payment-api", "type": "external", "dependencies": []},
            {"name": "email-service", "type": "external", "dependencies": []}
        ]
        
        return {
            "services": services,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_apm_traces(self,
                      service: str,
                      start_time: datetime = None,
                      end_time: datetime = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get APM traces."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        traces = []
        
        for i in range(min(limit, 50)):
            trace_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
            
            trace = {
                "trace_id": f"trace-{random.randint(100000, 999999)}",
                "span_id": f"span-{random.randint(10000, 99999)}",
                "timestamp": trace_time.isoformat(),
                "service": service,
                "operation": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "resource": f"/api/{random.choice(['users', 'payments', 'notifications'])}",
                "duration": random.uniform(0.001, 5.0),
                "status": random.choice(["ok", "error"]),
                "tags": {
                    "http.status_code": random.choice([200, 201, 400, 401, 403, 404, 500]),
                    "http.method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                    "environment": random.choice(["production", "staging"])
                }
            }
            
            traces.append(trace)
        
        return traces
    
    def get_synthetics_tests(self) -> List[Dict[str, Any]]:
        """Get synthetic monitoring tests."""
        tests = [
            {
                "test_id": "synthetic-test-1",
                "name": "Homepage Availability",
                "type": "api",
                "url": "https://demo.example.com/",
                "status": random.choice(["OK", "NOK"]),
                "last_check": datetime.now().isoformat(),
                "frequency": 300,
                "locations": ["aws:us-east-1", "aws:eu-west-1"],
                "tags": ["critical", "homepage"]
            },
            {
                "test_id": "synthetic-test-2",
                "name": "Login Flow",
                "type": "browser",
                "url": "https://demo.example.com/login",
                "status": random.choice(["OK", "NOK"]),
                "last_check": datetime.now().isoformat(),
                "frequency": 600,
                "locations": ["aws:us-west-2", "aws:ap-southeast-1"],
                "tags": ["critical", "authentication"]
            },
            {
                "test_id": "synthetic-test-3",
                "name": "API Health Check",
                "type": "api",
                "url": "https://demo.example.com/api/health",
                "status": random.choice(["OK", "NOK"]),
                "last_check": datetime.now().isoformat(),
                "frequency": 60,
                "locations": ["aws:us-east-1", "aws:eu-west-1", "aws:ap-northeast-1"],
                "tags": ["health", "api"]
            }
        ]
        
        return tests
    
    def get_rum_events(self,
                      start_time: datetime = None,
                      end_time: datetime = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get Real User Monitoring events."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        events = []
        
        for i in range(min(limit, 100)):
            event_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
            
            event = {
                "event_id": f"rum-{random.randint(10000, 99999)}",
                "timestamp": event_time.isoformat(),
                "type": random.choice(["view", "action", "error", "resource"]),
                "page": {
                    "url": f"https://demo.example.com/{random.choice(['home', 'login', 'dashboard', 'profile'])}",
                    "title": f"Demo {random.choice(['Home', 'Login', 'Dashboard', 'Profile'])}"
                },
                "user": {
                    "id": random.randint(1000, 9999),
                    "email": f"user{random.randint(1, 100)}@example.com"
                },
                "browser": {
                    "name": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
                    "version": f"{random.randint(90, 120)}.0"
                },
                "device": {
                    "type": random.choice(["desktop", "mobile", "tablet"]),
                    "brand": random.choice(["Apple", "Samsung", "Google", "Microsoft"])
                },
                "geo": {
                    "country": random.choice(["US", "GB", "DE", "FR", "CA"]),
                    "city": random.choice(["New York", "London", "Berlin", "Paris", "Toronto"])
                }
            }
            
            events.append(event)
        
        return events
    
    # Helper methods
    
    def _generate_log_message(self) -> str:
        """Generate realistic log message."""
        messages = [
            "User authentication successful",
            "Payment processing completed",
            "Database connection established",
            "Cache hit for key: user_profile_123",
            "API request processed in 150ms",
            "File upload completed successfully",
            "Email notification sent",
            "Search query executed",
            "User session expired",
            "Rate limit exceeded for user",
            "Database query failed",
            "External API call timeout",
            "Memory usage at 85%",
            "Disk space low on /var/log",
            "SSL certificate expiring soon"
        ]
        return random.choice(messages)
    
    def _generate_event_title(self) -> str:
        """Generate event title."""
        titles = [
            "New deployment to production",
            "High error rate detected",
            "Performance degradation alert",
            "Service dependency changed",
            "Configuration update applied",
            "Security scan completed",
            "Backup operation finished",
            "Certificate renewal required",
            "Resource scaling triggered",
            "Maintenance window started"
        ]
        return random.choice(titles)
    
    def _generate_event_description(self) -> str:
        """Generate event description."""
        descriptions = [
            "A new version has been deployed to the production environment",
            "Error rate has exceeded the threshold for the last 5 minutes",
            "Response time has increased significantly across multiple services",
            "Service dependency map has been updated with new connections",
            "Configuration changes have been applied and are now active",
            "Security vulnerability scan has completed with no critical issues found",
            "Automated backup process has finished successfully",
            "SSL certificate will expire in 30 days and needs renewal",
            "Auto-scaling has been triggered due to increased load",
            "Scheduled maintenance window has started for system updates"
        ]
        return random.choice(descriptions)
