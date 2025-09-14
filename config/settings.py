"""Configuration settings for the AI Test Automation Platform."""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    
    # Jira Configuration
    jira_url: str
    jira_username: str
    jira_api_token: str
    
    # Datadog Configuration
    datadog_api_key: str
    datadog_app_key: str
    datadog_site: str = "datadoghq.com"
    
    # Sentry Configuration
    sentry_dsn: Optional[str] = None
    
    # Vector Store Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    faiss_index_path: str = "./data/faiss_index"
    
    # Database Configuration
    database_url: str = "sqlite:///./data/test_automation.db"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Temporal Configuration
    temporal_host: str = "localhost:7233"
    
    # Kubernetes Configuration
    kubeconfig_path: str = "~/.kube/config"
    
    # Test Execution Configuration
    appium_server_url: str = "http://localhost:4723"
    default_test_timeout: int = 300
    max_concurrent_tests: int = 5
    
    # Monitoring
    prometheus_port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
