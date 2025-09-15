"""Demo mode configuration for the AI Test Automation Platform."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DemoConfig:
    """Configuration for demo mode."""
    
    # Demo mode settings
    enabled: bool = True
    use_synthetic_data: bool = True
    cache_demo_data: bool = True
    demo_data_dir: str = "demo_data"
    
    # Synthetic data settings
    jira_tickets_count: int = 100
    datadog_metrics_count: int = 2000
    sentry_events_count: int = 500
    github_events_count: int = 150
    test_executions_count: int = 800
    user_feedback_count: int = 100
    analytics_events_count: int = 500
    
    # Mock API settings
    mock_jira: bool = True
    mock_datadog: bool = True
    mock_sentry: bool = True
    mock_github: bool = True
    
    # Demo workflow settings
    demo_workflow_timeout: int = 300  # seconds
    max_demo_workflows: int = 5
    demo_workflow_parallel: bool = False
    
    # Demo data refresh settings
    refresh_interval_hours: int = 24
    force_refresh: bool = False
    
    # Demo logging settings
    demo_log_level: str = "INFO"
    show_demo_banners: bool = True
    
    # Demo API settings
    demo_api_port: int = 8001
    demo_api_host: str = "0.0.0.0"
    
    @classmethod
    def from_env(cls) -> 'DemoConfig':
        """Create demo config from environment variables."""
        return cls(
            enabled=os.getenv('DEMO_MODE', 'true').lower() == 'true',
            use_synthetic_data=os.getenv('USE_SYNTHETIC_DATA', 'true').lower() == 'true',
            cache_demo_data=os.getenv('CACHE_DEMO_DATA', 'true').lower() == 'true',
            demo_data_dir=os.getenv('DEMO_DATA_DIR', 'demo_data'),
            
            jira_tickets_count=int(os.getenv('DEMO_JIRA_TICKETS', '100')),
            datadog_metrics_count=int(os.getenv('DEMO_DATADOG_METRICS', '2000')),
            sentry_events_count=int(os.getenv('DEMO_SENTRY_EVENTS', '500')),
            github_events_count=int(os.getenv('DEMO_GITHUB_EVENTS', '150')),
            test_executions_count=int(os.getenv('DEMO_TEST_EXECUTIONS', '800')),
            user_feedback_count=int(os.getenv('DEMO_USER_FEEDBACK', '100')),
            analytics_events_count=int(os.getenv('DEMO_ANALYTICS_EVENTS', '500')),
            
            mock_jira=os.getenv('DEMO_MOCK_JIRA', 'true').lower() == 'true',
            mock_datadog=os.getenv('DEMO_MOCK_DATADOG', 'true').lower() == 'true',
            mock_sentry=os.getenv('DEMO_MOCK_SENTRY', 'true').lower() == 'true',
            mock_github=os.getenv('DEMO_MOCK_GITHUB', 'true').lower() == 'true',
            
            demo_workflow_timeout=int(os.getenv('DEMO_WORKFLOW_TIMEOUT', '300')),
            max_demo_workflows=int(os.getenv('DEMO_MAX_WORKFLOWS', '5')),
            demo_workflow_parallel=os.getenv('DEMO_WORKFLOW_PARALLEL', 'false').lower() == 'true',
            
            refresh_interval_hours=int(os.getenv('DEMO_REFRESH_INTERVAL', '24')),
            force_refresh=os.getenv('DEMO_FORCE_REFRESH', 'false').lower() == 'true',
            
            demo_log_level=os.getenv('DEMO_LOG_LEVEL', 'INFO'),
            show_demo_banners=os.getenv('DEMO_SHOW_BANNERS', 'true').lower() == 'true',
            
            demo_api_port=int(os.getenv('DEMO_API_PORT', '8001')),
            demo_api_host=os.getenv('DEMO_API_HOST', '0.0.0.0')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'use_synthetic_data': self.use_synthetic_data,
            'cache_demo_data': self.cache_demo_data,
            'demo_data_dir': self.demo_data_dir,
            'jira_tickets_count': self.jira_tickets_count,
            'datadog_metrics_count': self.datadog_metrics_count,
            'sentry_events_count': self.sentry_events_count,
            'github_events_count': self.github_events_count,
            'test_executions_count': self.test_executions_count,
            'user_feedback_count': self.user_feedback_count,
            'analytics_events_count': self.analytics_events_count,
            'mock_jira': self.mock_jira,
            'mock_datadog': self.mock_datadog,
            'mock_sentry': self.mock_sentry,
            'mock_github': self.mock_github,
            'demo_workflow_timeout': self.demo_workflow_timeout,
            'max_demo_workflows': self.max_demo_workflows,
            'demo_workflow_parallel': self.demo_workflow_parallel,
            'refresh_interval_hours': self.refresh_interval_hours,
            'force_refresh': self.force_refresh,
            'demo_log_level': self.demo_log_level,
            'show_demo_banners': self.show_demo_banners,
            'demo_api_port': self.demo_api_port,
            'demo_api_host': self.demo_api_host
        }


# Global demo config instance
demo_config = DemoConfig.from_env()


def get_demo_config() -> DemoConfig:
    """Get the global demo configuration."""
    return demo_config


def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return demo_config.enabled


def get_demo_banner() -> str:
    """Get demo mode banner."""
    if not demo_config.show_demo_banners:
        return ""
    
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🎯 DEMO MODE ENABLED 🎯                           ║
║                                                                              ║
║  This system is running in DEMO MODE with synthetic data for testing.       ║
║  All data is generated and does not connect to real external services.      ║
║                                                                              ║
║  Features:                                                                   ║
║  • 🤖 AI-Powered Test Generation                                            ║
║  • 🔧 Multi-Agent LangGraph System                                          ║
║  • 📊 Comprehensive Data Ingestion                                          ║
║  • 🔍 Self-Healing Test Execution                                           ║
║  • 📈 Continuous Learning & Optimization                                    ║
║                                                                              ║
║  To disable demo mode, set DEMO_MODE=false in environment variables.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    return banner.strip()


def print_demo_banner():
    """Print demo mode banner."""
    if demo_config.show_demo_banners and demo_config.enabled:
        print(get_demo_banner())
        print()
