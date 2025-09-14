"""Analytics data ingestion module for pulling crash logs and usage data."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from datadog import initialize, api
import sentry_sdk
from dataclasses import dataclass

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CrashEvent:
    """Represents a crash event from monitoring systems."""
    event_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    user_id: Optional[str]
    session_id: Optional[str]
    device_info: Dict[str, Any]
    app_version: str
    severity: str
    tags: Dict[str, str]


@dataclass
class UsageEvent:
    """Represents a usage analytics event."""
    event_id: str
    timestamp: datetime
    event_name: str
    user_id: Optional[str]
    session_id: Optional[str]
    properties: Dict[str, Any]
    page_url: Optional[str]
    app_version: str


class AnalyticsIngestor:
    """Handles data ingestion from analytics and monitoring systems."""
    
    def __init__(self):
        self.settings = get_settings()
        self._initialize_datadog()
        self._initialize_sentry()
    
    def _initialize_datadog(self):
        """Initialize Datadog API client."""
        try:
            initialize(
                api_key=self.settings.datadog_api_key,
                app_key=self.settings.datadog_app_key,
                api_host=self.settings.datadog_site
            )
            logger.info("Datadog client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Datadog client: {e}")
    
    def _initialize_sentry(self):
        """Initialize Sentry SDK."""
        if self.settings.sentry_dsn:
            try:
                sentry_sdk.init(dsn=self.settings.sentry_dsn)
                logger.info("Sentry client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Sentry client: {e}")
    
    def fetch_crash_events(self, hours_back: int = 24) -> List[CrashEvent]:
        """Fetch crash events from Datadog and Sentry."""
        crash_events = []
        
        # Fetch from Datadog
        datadog_crashes = self._fetch_datadog_crashes(hours_back)
        crash_events.extend(datadog_crashes)
        
        # Fetch from Sentry (if available)
        if self.settings.sentry_dsn:
            sentry_crashes = self._fetch_sentry_crashes(hours_back)
            crash_events.extend(sentry_crashes)
        
        logger.info(f"Fetched {len(crash_events)} crash events")
        return crash_events
    
    def fetch_usage_events(self, hours_back: int = 24) -> List[UsageEvent]:
        """Fetch usage analytics events."""
        try:
            # Query Datadog for custom events
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = "events('source:analytics').rollup('count').by('event_name')"
            
            response = api.Event.query(
                start=int(start_time.timestamp()),
                end=int(end_time.timestamp()),
                query=query
            )
            
            usage_events = []
            for event in response.get('events', []):
                usage_event = UsageEvent(
                    event_id=event.get('id', ''),
                    timestamp=datetime.fromtimestamp(event['date_happened']),
                    event_name=event.get('title', ''),
                    user_id=event.get('tags', {}).get('user_id'),
                    session_id=event.get('tags', {}).get('session_id'),
                    properties=event.get('text', {}),
                    page_url=event.get('url'),
                    app_version=event.get('tags', {}).get('app_version', 'unknown')
                )
                usage_events.append(usage_event)
            
            logger.info(f"Fetched {len(usage_events)} usage events")
            return usage_events
            
        except Exception as e:
            logger.error(f"Error fetching usage events: {e}")
            return []
    
    def _fetch_datadog_crashes(self, hours_back: int) -> List[CrashEvent]:
        """Fetch crash events from Datadog."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Query for error logs
            query = "source:application status:error"
            
            response = api.Log.query(
                start=int(start_time.timestamp()),
                end=int(end_time.timestamp()),
                query=query,
                limit=1000
            )
            
            crash_events = []
            for log in response.get('logs', []):
                crash_event = CrashEvent(
                    event_id=log.get('id', ''),
                    timestamp=datetime.fromtimestamp(log['timestamp'] / 1000),
                    error_type=log.get('attributes', {}).get('error.kind', 'Unknown'),
                    error_message=log.get('message', ''),
                    stack_trace=log.get('attributes', {}).get('error.stack', ''),
                    user_id=log.get('attributes', {}).get('user_id'),
                    session_id=log.get('attributes', {}).get('session_id'),
                    device_info=log.get('attributes', {}).get('device', {}),
                    app_version=log.get('attributes', {}).get('version', 'unknown'),
                    severity=log.get('level', 'error'),
                    tags=log.get('attributes', {})
                )
                crash_events.append(crash_event)
            
            return crash_events
            
        except Exception as e:
            logger.error(f"Error fetching Datadog crashes: {e}")
            return []
    
    def _fetch_sentry_crashes(self, hours_back: int) -> List[CrashEvent]:
        """Fetch crash events from Sentry."""
        # Note: This would require Sentry API integration
        # For now, return empty list as placeholder
        logger.info("Sentry crash fetching not implemented yet")
        return []
    
    def get_crash_content_for_indexing(self, crash: CrashEvent) -> Dict[str, Any]:
        """Convert crash event to format suitable for vector store indexing."""
        content = f"""
        Error Type: {crash.error_type}
        Error Message: {crash.error_message}
        Stack Trace: {crash.stack_trace}
        Device Info: {crash.device_info}
        App Version: {crash.app_version}
        Severity: {crash.severity}
        """
        
        return {
            "id": crash.event_id,
            "content": content.strip(),
            "metadata": {
                "error_type": crash.error_type,
                "severity": crash.severity,
                "app_version": crash.app_version,
                "user_id": crash.user_id,
                "session_id": crash.session_id,
                "timestamp": crash.timestamp.isoformat(),
                "device_info": crash.device_info,
                "tags": crash.tags,
                "source": "analytics"
            }
        }
    
    def get_usage_content_for_indexing(self, usage: UsageEvent) -> Dict[str, Any]:
        """Convert usage event to format suitable for vector store indexing."""
        content = f"""
        Event Name: {usage.event_name}
        Properties: {usage.properties}
        Page URL: {usage.page_url}
        App Version: {usage.app_version}
        """
        
        return {
            "id": usage.event_id,
            "content": content.strip(),
            "metadata": {
                "event_name": usage.event_name,
                "app_version": usage.app_version,
                "user_id": usage.user_id,
                "session_id": usage.session_id,
                "timestamp": usage.timestamp.isoformat(),
                "properties": usage.properties,
                "page_url": usage.page_url,
                "source": "analytics"
            }
        }
