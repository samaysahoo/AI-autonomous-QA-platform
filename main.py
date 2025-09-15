"""Main entry point for the AI Test Automation Platform."""

import logging
import asyncio
from typing import Optional
import uvicorn

from config.settings import get_settings
from src.data_ingestion import JiraIngestor, AnalyticsIngestor, VectorStoreManager
from src.dashboard.dashboard_api import app
from src.langgraph_agents.api import app as langgraph_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize the AI Test Automation Platform."""
    
    logger.info("Initializing AI Test Automation Platform...")
    
    try:
        # Initialize vector store
        vector_store = VectorStoreManager()
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store initialized with {stats.get('total_documents', 0)} documents")
        
        # Initialize data ingestion components
        jira_ingestor = JiraIngestor()
        analytics_ingestor = AnalyticsIngestor()
        logger.info("Data ingestion components initialized")
        
        # Test vector store connectivity
        test_docs = [
            {
                "id": "test_doc_1",
                "content": "Test document for system initialization",
                "metadata": {"source": "system", "type": "test"}
            }
        ]
        
        vector_store.add_documents(test_docs)
        logger.info("Vector store connectivity test passed")
        
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


def run_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard API server."""
    
    logger.info(f"Starting dashboard server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


def run_langgraph_api(host: str = "0.0.0.0", port: int = 8001):
    """Run the LangGraph multi-agent API server."""
    
    logger.info(f"Starting LangGraph multi-agent server on {host}:{port}")
    
    uvicorn.run(
        langgraph_app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


async def run_data_ingestion():
    """Run data ingestion pipeline."""
    
    logger.info("Starting data ingestion pipeline...")
    
    try:
        # Initialize components
        jira_ingestor = JiraIngestor()
        analytics_ingestor = AnalyticsIngestor()
        vector_store = VectorStoreManager()
        
        # Ingest Jira data
        logger.info("Ingesting Jira data...")
        specs = jira_ingestor.fetch_specifications(days_back=30)
        stories = jira_ingestor.fetch_user_stories(days_back=30)
        bugs = jira_ingestor.fetch_bug_tickets(days_back=30)
        
        jira_docs = []
        for spec in specs:
            jira_docs.append(jira_ingestor.get_ticket_content_for_indexing(spec))
        for story in stories:
            jira_docs.append(jira_ingestor.get_ticket_content_for_indexing(story))
        for bug in bugs:
            jira_docs.append(jira_ingestor.get_ticket_content_for_indexing(bug))
        
        if jira_docs:
            vector_store.add_documents(jira_docs)
            logger.info(f"Indexed {len(jira_docs)} Jira documents")
        
        # Ingest analytics data
        logger.info("Ingesting analytics data...")
        crash_events = analytics_ingestor.fetch_crash_events(hours_back=24)
        usage_events = analytics_ingestor.fetch_usage_events(hours_back=24)
        
        analytics_docs = []
        for crash in crash_events:
            analytics_docs.append(analytics_ingestor.get_crash_content_for_indexing(crash))
        for usage in usage_events:
            analytics_docs.append(analytics_ingestor.get_usage_content_for_indexing(usage))
        
        if analytics_docs:
            vector_store.add_documents(analytics_docs)
            logger.info(f"Indexed {len(analytics_docs)} analytics documents")
        
        logger.info("Data ingestion pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")


async def run_test_generation_example():
    """Run example test generation."""
    
    logger.info("Running test generation example...")
    
    try:
        from src.test_generation import TestCaseGenerator
        from src.test_generation.test_scenario import TestType, TestFramework
        
        test_generator = TestCaseGenerator()
        
        # Generate test scenarios
        scenarios = test_generator.generate_test_scenarios(
            query="User login functionality with email and password",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=3
        )
        
        logger.info(f"Generated {len(scenarios)} test scenarios")
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Scenario {i+1}: {scenario.title}")
            logger.info(f"  Steps: {scenario.get_step_count()}")
            logger.info(f"  Priority: {scenario.priority.value}")
            logger.info(f"  Duration: {scenario.get_estimated_duration()}s")
        
        # Generate code for first scenario
        if scenarios:
            from src.test_generation import CodeGenerator
            code_generator = CodeGenerator()
            test_code = code_generator.generate_test_code(scenarios[0])
            
            logger.info(f"Generated test code: {test_code['filename']}")
            logger.info(f"Framework: {test_code['framework']}")
        
        logger.info("Test generation example completed successfully")
        
    except Exception as e:
        logger.error(f"Test generation example failed: {e}")


async def run_risk_analysis_example():
    """Run example risk analysis."""
    
    logger.info("Running risk analysis example...")
    
    try:
        from src.observability import RiskAnalyzer
        
        risk_analyzer = RiskAnalyzer()
        
        # Example diff content
        diff_content = """
        + def authenticate_user(username, password):
        +     if username == "admin" and password == "password":
        +         return True
        +     return False
        
        - def old_auth_method():
        -     # Old authentication logic
        -     pass
        """
        
        changed_files = ["src/auth.py", "src/security.py"]
        commit_metadata = {
            "hash": "abc123",
            "message": "Update authentication logic",
            "author": "developer@example.com"
        }
        
        # Analyze risk
        risk_score = risk_analyzer.analyze_code_change_risk(
            diff_content=diff_content,
            changed_files=changed_files,
            commit_metadata=commit_metadata
        )
        
        logger.info(f"Risk analysis completed:")
        logger.info(f"  Risk Level: {risk_score.risk_level:.2f}")
        logger.info(f"  Confidence: {risk_score.confidence:.2f}")
        logger.info(f"  Components: {', '.join(risk_score.affected_areas)}")
        logger.info(f"  Recommendations: {len(risk_score.recommendations)}")
        
        for i, rec in enumerate(risk_score.recommendations):
            logger.info(f"    {i+1}. {rec}")
        
        logger.info("Risk analysis example completed successfully")
        
    except Exception as e:
        logger.error(f"Risk analysis example failed: {e}")


def main():
    """Main entry point."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [command]")
        print("Commands:")
        print("  dashboard    - Run the dashboard API server")
        print("  langgraph    - Run the LangGraph multi-agent API server")
        print("  ingest       - Run data ingestion pipeline")
        print("  generate     - Run test generation example")
        print("  risk         - Run risk analysis example")
        print("  demo         - Run LangGraph multi-agent demo")
        print("  demo-full    - Run complete system demo with synthetic data")
        print("  init         - Initialize system")
        return
    
    command = sys.argv[1].lower()
    
    if command == "dashboard":
        # Initialize system first
        if not asyncio.run(initialize_system()):
            logger.error("Failed to initialize system")
            return
        
        # Run dashboard
        run_dashboard()
    
    elif command == "ingest":
        asyncio.run(run_data_ingestion())
    
    elif command == "generate":
        asyncio.run(run_test_generation_example())
    
    elif command == "risk":
        asyncio.run(run_risk_analysis_example())
    
    elif command == "langgraph":
        # Initialize system first
        if not asyncio.run(initialize_system()):
            logger.error("Failed to initialize system")
            return
        
        # Run LangGraph API
        run_langgraph_api()
    
    elif command == "demo":
        # Run LangGraph demo
        from scripts.demo_langgraph_system import main as demo_main
        asyncio.run(demo_main())
    
    elif command == "demo-full":
        # Run complete system demo with synthetic data
        from scripts.demo_full_system import main as demo_full_main
        asyncio.run(demo_full_main())
    
    elif command == "init":
        success = asyncio.run(initialize_system())
        if success:
            print("System initialized successfully")
        else:
            print("System initialization failed")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
