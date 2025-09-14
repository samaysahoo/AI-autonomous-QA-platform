"""FastAPI dashboard for the AI Test Automation Platform."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.settings import get_settings
from src.data_ingestion import JiraIngestor, AnalyticsIngestor, VectorStoreManager
from src.test_generation import TestCaseGenerator, CodeGenerator
from src.test_execution import TestOrchestrator, TestRunner
from src.observability import RiskAnalyzer, TestPrioritizer, CodeDiffAnalyzer
from .failure_clusterer import FailureClusterer
from .root_cause_analyzer import RootCauseAnalyzer
from .feedback_loop import FeedbackLoop, HumanFeedback

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Test Automation Platform",
    description="Comprehensive AI-powered testing automation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
settings = get_settings()

# Data ingestion components
jira_ingestor = JiraIngestor()
analytics_ingestor = AnalyticsIngestor()
vector_store = VectorStoreManager()

# Test generation components
test_generator = TestCaseGenerator()
code_generator = CodeGenerator()

# Test execution components
test_orchestrator = TestOrchestrator()
test_runner = TestRunner()

# Observability components
risk_analyzer = RiskAnalyzer()
test_prioritizer = TestPrioritizer()
code_diff_analyzer = CodeDiffAnalyzer()

# Dashboard components
failure_clusterer = FailureClusterer()
root_cause_analyzer = RootCauseAnalyzer()
feedback_loop = FeedbackLoop()


# Pydantic models for API
class TestGenerationRequest(BaseModel):
    query: str
    test_type: str = "e2e"
    framework: str = "appium"
    max_scenarios: int = 5


class TestExecutionRequest(BaseModel):
    scenario_ids: List[str]
    execution_mode: str = "parallel"


class RiskAnalysisRequest(BaseModel):
    diff_content: str
    changed_files: List[str]
    commit_metadata: Dict[str, Any]


class FeedbackSubmission(BaseModel):
    test_result_id: str
    feedback_type: str
    rating: int
    comments: str
    corrections: Dict[str, Any]
    reviewer: str


# API Routes

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "AI Test Automation Platform",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check component health
        vector_stats = vector_store.get_collection_stats()
        
        return {
            "status": "healthy",
            "components": {
                "vector_store": "healthy" if vector_stats.get('total_documents', 0) >= 0 else "warning",
                "test_generator": "healthy",
                "test_orchestrator": "healthy",
                "risk_analyzer": "healthy"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Data Ingestion Endpoints

@app.post("/api/ingest/jira")
async def ingest_jira_data(background_tasks: BackgroundTasks):
    """Ingest data from Jira."""
    try:
        background_tasks.add_task(_ingest_jira_data_task)
        return {"message": "Jira data ingestion started", "status": "processing"}
    except Exception as e:
        logger.error(f"Error starting Jira ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/analytics")
async def ingest_analytics_data(background_tasks: BackgroundTasks):
    """Ingest analytics and crash data."""
    try:
        background_tasks.add_task(_ingest_analytics_data_task)
        return {"message": "Analytics data ingestion started", "status": "processing"}
    except Exception as e:
        logger.error(f"Error starting analytics ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Test Generation Endpoints

@app.post("/api/test-generation/generate")
async def generate_test_scenarios(request: TestGenerationRequest):
    """Generate test scenarios based on query."""
    try:
        from src.test_generation.test_scenario import TestType, TestFramework
        
        test_type = TestType(request.test_type)
        framework = TestFramework(request.framework)
        
        scenarios = test_generator.generate_test_scenarios(
            query=request.query,
            test_type=test_type,
            framework=framework,
            max_scenarios=request.max_scenarios
        )
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "count": len(scenarios),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating test scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-generation/generate-code/{scenario_id}")
async def generate_test_code(scenario_id: str):
    """Generate executable code for a test scenario."""
    try:
        # This would fetch the scenario from storage
        # For now, return a placeholder
        return {
            "scenario_id": scenario_id,
            "code": "// Generated test code would appear here",
            "language": "python",
            "framework": "appium",
            "filename": f"test_{scenario_id}.py"
        }
    except Exception as e:
        logger.error(f"Error generating test code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Test Execution Endpoints

@app.post("/api/test-execution/run")
async def execute_tests(request: TestExecutionRequest, background_tasks: BackgroundTasks):
    """Execute test scenarios."""
    try:
        background_tasks.add_task(_execute_tests_task, request.scenario_ids, request.execution_mode)
        return {
            "message": "Test execution started",
            "scenario_ids": request.scenario_ids,
            "execution_mode": request.execution_mode,
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Error starting test execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-execution/status/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get test execution status."""
    try:
        # This would fetch from actual execution tracking
        return {
            "execution_id": execution_id,
            "status": "completed",
            "progress": 100,
            "results": {
                "passed": 8,
                "failed": 2,
                "skipped": 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Observability Endpoints

@app.post("/api/observability/analyze-risk")
async def analyze_code_risk(request: RiskAnalysisRequest):
    """Analyze risk level of code changes."""
    try:
        risk_score = risk_analyzer.analyze_code_change_risk(
            diff_content=request.diff_content,
            changed_files=request.changed_files,
            commit_metadata=request.commit_metadata
        )
        
        return {
            "risk_score": risk_score.risk_level,
            "confidence": risk_score.confidence,
            "recommendations": risk_score.recommendations,
            "affected_components": risk_score.affected_areas,
            "factors": risk_score.factors
        }
    except Exception as e:
        logger.error(f"Error analyzing code risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/observability/crash-patterns")
async def get_crash_patterns():
    """Get analyzed crash patterns."""
    try:
        # Fetch crash events and analyze patterns
        crash_events = analytics_ingestor.fetch_crash_events(hours_back=24)
        patterns = risk_analyzer.analyze_crash_patterns(crash_events)
        
        return {
            "patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "severity": pattern.severity,
                    "affected_components": pattern.affected_components,
                    "user_impact": pattern.user_impact
                }
                for pattern in patterns
            ],
            "count": len(patterns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting crash patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard Endpoints

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Get dashboard overview data."""
    try:
        # Get system statistics
        vector_stats = vector_store.get_collection_stats()
        
        # Get recent test results (simulated)
        recent_results = {
            "total_tests": 150,
            "passed": 120,
            "failed": 25,
            "skipped": 5,
            "success_rate": 0.8,
            "avg_execution_time": 45.2
        }
        
        # Get feedback summary
        feedback_analysis = feedback_loop.analyze_feedback_patterns()
        
        return {
            "system_stats": {
                "total_documents": vector_stats.get('total_documents', 0),
                "vector_store_status": "healthy" if vector_stats.get('total_documents', 0) > 0 else "warning"
            },
            "test_results": recent_results,
            "feedback_summary": {
                "total_feedback": feedback_analysis.get('total_feedback', 0),
                "average_rating": feedback_analysis.get('average_ratings', {}).get('overall', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/failure-analysis")
async def get_failure_analysis():
    """Get failure clustering and root cause analysis."""
    try:
        # Simulate failure data for demonstration
        simulated_failures = [
            {
                "test_name": "test_login_functionality",
                "error_message": "Element not found: login button",
                "stack_trace": "NoSuchElementException at line 45",
                "duration": 30,
                "timestamp": datetime.now().isoformat()
            },
            {
                "test_name": "test_user_registration",
                "error_message": "Timeout waiting for element",
                "stack_trace": "TimeoutException at line 67",
                "duration": 60,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Cluster failures
        clustering_result = failure_clusterer.cluster_failures(simulated_failures)
        
        # Analyze root causes
        if clustering_result.clusters:
            root_cause_analyses = root_cause_analyzer.analyze_root_causes(clustering_result.clusters)
        else:
            root_cause_analyses = []
        
        return {
            "clustering": {
                "total_clusters": len(clustering_result.clusters),
                "silhouette_score": clustering_result.silhouette_score,
                "method_used": clustering_result.method_used
            },
            "root_causes": [
                {
                    "cluster_id": analysis.cluster_id,
                    "primary_cause": analysis.primary_cause,
                    "confidence_score": analysis.confidence_score,
                    "recommendations": analysis.recommendations[:3]  # Top 3 recommendations
                }
                for analysis in root_cause_analyses
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting failure analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dashboard/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit human feedback for learning."""
    try:
        human_feedback = HumanFeedback(
            feedback_id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_result_id=feedback.test_result_id,
            feedback_type=feedback.feedback_type,
            rating=feedback.rating,
            comments=feedback.comments,
            corrections=feedback.corrections,
            timestamp=datetime.now(),
            reviewer=feedback.reviewer
        )
        
        success = feedback_loop.record_human_feedback(human_feedback)
        
        if success:
            return {
                "message": "Feedback recorded successfully",
                "feedback_id": human_feedback.feedback_id,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/feedback-analysis")
async def get_feedback_analysis():
    """Get feedback analysis and improvement recommendations."""
    try:
        analysis = feedback_loop.analyze_feedback_patterns()
        
        return {
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting feedback analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background Tasks

async def _ingest_jira_data_task():
    """Background task for Jira data ingestion."""
    try:
        # Fetch specifications, user stories, and bug tickets
        specs = jira_ingestor.fetch_specifications(days_back=30)
        stories = jira_ingestor.fetch_user_stories(days_back=30)
        bugs = jira_ingestor.fetch_bug_tickets(days_back=30)
        
        # Prepare documents for indexing
        documents = []
        
        for spec in specs:
            doc = jira_ingestor.get_ticket_content_for_indexing(spec)
            documents.append(doc)
        
        for story in stories:
            doc = jira_ingestor.get_ticket_content_for_indexing(story)
            documents.append(doc)
        
        for bug in bugs:
            doc = jira_ingestor.get_ticket_content_for_indexing(bug)
            documents.append(doc)
        
        # Index in vector store
        vector_store.add_documents(documents)
        
        logger.info(f"Jira data ingestion completed: {len(documents)} documents indexed")
        
    except Exception as e:
        logger.error(f"Error in Jira data ingestion task: {e}")


async def _ingest_analytics_data_task():
    """Background task for analytics data ingestion."""
    try:
        # Fetch crash events and usage events
        crash_events = analytics_ingestor.fetch_crash_events(hours_back=24)
        usage_events = analytics_ingestor.fetch_usage_events(hours_back=24)
        
        # Prepare documents for indexing
        documents = []
        
        for crash in crash_events:
            doc = analytics_ingestor.get_crash_content_for_indexing(crash)
            documents.append(doc)
        
        for usage in usage_events:
            doc = analytics_ingestor.get_usage_content_for_indexing(usage)
            documents.append(doc)
        
        # Index in vector store
        vector_store.add_documents(documents)
        
        logger.info(f"Analytics data ingestion completed: {len(documents)} documents indexed")
        
    except Exception as e:
        logger.error(f"Error in analytics data ingestion task: {e}")


async def _execute_tests_task(scenario_ids: List[str], execution_mode: str):
    """Background task for test execution."""
    try:
        logger.info(f"Starting test execution for {len(scenario_ids)} scenarios in {execution_mode} mode")
        
        # This would fetch scenarios and execute them
        # For now, just log the task
        
        logger.info("Test execution task completed")
        
    except Exception as e:
        logger.error(f"Error in test execution task: {e}")


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
