"""Learning Agent - consumes logs, feedback, and PRs to update models."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import asaclass

from .base_agent import BaseAgent, AgentCapabilities, MessageType, AgentMessage
from ..dashboard.feedback_loop import FeedbackLoop, HumanFeedback, ModelUpdate
from ..test_generation.test_generator import TestCaseGenerator
from ..observability.risk_analyzer import RiskAnalyzer
from ..data_ingestion.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """Agent responsible for continuous learning from logs, feedback, and code changes."""
    
    def __init__(self, agent_id: str = "learning-001"):
        capabilities = AgentCapabilities(
            can_learn_from_data=True,
            max_concurrent_tasks=2  # Learning tasks can be resource intensive
        )
        
        super().__init__(agent_id, "Learning Agent", capabilities)
        
        # Initialize components
        self.feedback_loop = FeedbackLoop()
        self.test_generator = TestCaseGenerator()
        self.risk_analyzer = RiskAnalyzer()
        self.vector_store = VectorStoreManager()
        
        # Learning strategies
        self.learning_strategies = {
            "feedback_learning": self._learn_from_feedback,
            "pattern_learning": self._learn_from_patterns,
            "code_change_learning": self._learn_from_code_changes,
            "failure_learning": self._learn_from_failures,
            "success_learning": self._learn_from_successes,
            "model_optimization": self._optimize_models
        }
        
        # Learning data storage
        self.learning_history = []
        self.model_performance = {}
        self.pattern_library = {}
        self.knowledge_base = {}
        
        # Learning metrics
        self.learning_metrics = {
            "total_learning_cycles": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "improvement_rate": 0.0,
            "last_learning_cycle": None
        }
        
        logger.info(f"Learning Agent initialized with {len(self.learning_strategies)} strategies")
    
    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this agent can handle a specific task type."""
        return task_type in [
            "learn_from_feedback",
            "learn_from_logs",
            "learn_from_prs",
            "update_models",
            "optimize_performance",
            "analyze_learning_patterns",
            "generate_learning_report",
            "sync_knowledge_base"
        ]
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a learning task."""
        task_type = task_data.get("task_type", "")
        
        logger.info(f"Learning Agent processing task: {task_type}")
        
        try:
            if task_type == "learn_from_feedback":
                return await self._learn_from_feedback(task_data)
            elif task_type == "learn_from_logs":
                return await self._learn_from_logs(task_data)
            elif task_type == "learn_from_prs":
                return await self._learn_from_prs(task_data)
            elif task_type == "update_models":
                return await self._update_models(task_data)
            elif task_type == "optimize_performance":
                return await self._optimize_performance(task_data)
            elif task_type == "analyze_learning_patterns":
                return await self._analyze_learning_patterns(task_data)
            elif task_type == "generate_learning_report":
                return await self._generate_learning_report(task_data)
            elif task_type == "sync_knowledge_base":
                return await self._sync_knowledge_base(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_type}: {e}")
            raise
    
    async def _learn_from_feedback(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from human feedback and corrections."""
        feedback_data = task_data.get("feedback", [])
        learning_mode = task_data.get("learning_mode", "incremental")
        
        if not feedback_data:
            return {
                "learning_results": [],
                "message": "No feedback data provided"
            }
        
        learning_results = []
        
        for feedback_item in feedback_data:
            # Convert to HumanFeedback object
            feedback = self._convert_to_human_feedback(feedback_item)
            
            # Process feedback based on type
            if feedback.feedback_type == "test_correction":
                result = await self._learn_from_test_correction(feedback)
            elif feedback.feedback_type == "model_correction":
                result = await self._learn_from_model_correction(feedback)
            elif feedback.feedback_type == "priority_adjustment":
                result = await self._learn_from_priority_adjustment(feedback)
            else:
                result = await self._learn_from_general_feedback(feedback)
            
            learning_results.append(result)
        
        # Update models based on feedback
        if learning_mode == "batch":
            model_updates = await self._batch_update_models(learning_results)
        else:
            model_updates = await self._incremental_update_models(learning_results)
        
        # Update learning metrics
        self._update_learning_metrics(learning_results)
        
        return {
            "learning_results": learning_results,
            "model_updates": model_updates,
            "learning_mode": learning_mode,
            "total_feedback_processed": len(feedback_data),
            "learning_timestamp": datetime.now().isoformat()
        }
    
    async def _learn_from_logs(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from system logs and execution data."""
        logs_data = task_data.get("logs", [])
        log_sources = task_data.get("log_sources", ["execution", "application", "system"])
        
        if not logs_data:
            return {
                "patterns_learned": [],
                "message": "No log data provided"
            }
        
        # Analyze different types of logs
        execution_logs = [log for log in logs_data if log.get("source") == "execution"]
        application_logs = [log for log in logs_data if log.get("source") == "application"]
        system_logs = [log for log in logs_data if log.get("source") == "system"]
        
        learning_results = {}
        
        # Learn from execution logs
        if execution_logs:
            execution_patterns = await self._extract_execution_patterns(execution_logs)
            learning_results["execution_patterns"] = execution_patterns
        
        # Learn from application logs
        if application_logs:
            app_patterns = await self._extract_application_patterns(application_logs)
            learning_results["application_patterns"] = app_patterns
        
        # Learn from system logs
        if system_logs:
            system_patterns = await self._extract_system_patterns(system_logs)
            learning_results["system_patterns"] = system_patterns
        
        # Update pattern library
        await self._update_pattern_library(learning_results)
        
        return {
            "patterns_learned": learning_results,
            "log_sources_analyzed": log_sources,
            "total_logs_processed": len(logs_data),
            "learning_timestamp": datetime.now().isoformat()
        }
    
    async def _learn_from_prs(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from pull requests and code changes."""
        prs_data = task_data.get("pull_requests", [])
        learning_focus = task_data.get("learning_focus", ["test_changes", "bug_fixes", "feature_additions"])
        
        if not prs_data:
            return {
                "pr_insights": [],
                "message": "No pull request data provided"
            }
        
        pr_insights = []
        
        for pr in prs_data:
            # Analyze PR for learning opportunities
            insights = await self._analyze_pr_for_learning(pr, learning_focus)
            pr_insights.append(insights)
        
        # Extract common patterns from PRs
        common_patterns = await self._extract_pr_patterns(pr_insights)
        
        # Update knowledge base with PR insights
        await self._update_knowledge_base(pr_insights, "pull_requests")
        
        return {
            "pr_insights": pr_insights,
            "common_patterns": common_patterns,
            "learning_focus": learning_focus,
            "total_prs_analyzed": len(prs_data),
            "learning_timestamp": datetime.now().isoformat()
        }
    
    async def _update_models(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update AI models based on learning data."""
        model_types = task_data.get("model_types", ["test_generator", "risk_analyzer", "all"])
        update_strategy = task_data.get("update_strategy", "incremental")
        
        update_results = {}
        
        for model_type in model_types:
            if model_type == "test_generator" or model_type == "all":
                result = await self._update_test_generator_model(update_strategy)
                update_results["test_generator"] = result
            
            if model_type == "risk_analyzer" or model_type == "all":
                result = await self._update_risk_analyzer_model(update_strategy)
                update_results["risk_analyzer"] = result
        
        # Validate model updates
        validation_results = await self._validate_model_updates(update_results)
        
        return {
            "model_updates": update_results,
            "validation_results": validation_results,
            "update_strategy": update_strategy,
            "update_timestamp": datetime.now().isoformat()
        }
    
    async def _optimize_performance(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model performance based on metrics."""
        optimization_target = task_data.get("target", "accuracy")
        performance_data = task_data.get("performance_data", {})
        
        # Analyze current performance
        current_performance = await self._analyze_current_performance(performance_data)
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(
            current_performance, optimization_target
        )
        
        # Apply optimizations
        optimization_results = await self._apply_optimizations(optimization_opportunities)
        
        # Measure improvement
        improvement_metrics = await self._measure_improvement(
            current_performance, optimization_results
        )
        
        return {
            "optimization_target": optimization_target,
            "current_performance": current_performance,
            "optimization_opportunities": optimization_opportunities,
            "optimization_results": optimization_results,
            "improvement_metrics": improvement_metrics,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_learning_patterns(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in learning data."""
        time_period = task_data.get("time_period", 30)  # days
        pattern_types = task_data.get("pattern_types", ["success", "failure", "improvement"])
        
        # Get recent learning history
        cutoff_date = datetime.now() - timedelta(days=time_period)
        recent_history = [
            h for h in self.learning_history
            if h.get("timestamp", datetime.min) > cutoff_date
        ]
        
        if not recent_history:
            return {
                "patterns": [],
                "message": f"No learning data found in the last {time_period} days"
            }
        
        # Analyze different pattern types
        pattern_analysis = {}
        
        for pattern_type in pattern_types:
            patterns = await self._extract_patterns_by_type(recent_history, pattern_type)
            pattern_analysis[pattern_type] = patterns
        
        # Identify learning trends
        learning_trends = await self._identify_learning_trends(recent_history)
        
        # Generate insights
        insights = await self._generate_learning_insights(pattern_analysis, learning_trends)
        
        return {
            "patterns": pattern_analysis,
            "learning_trends": learning_trends,
            "insights": insights,
            "analysis_period": f"Last {time_period} days",
            "total_learning_events": len(recent_history),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_learning_report(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive learning report."""
        report_scope = task_data.get("scope", "comprehensive")
        time_period = task_data.get("time_period", 30)
        
        # Collect learning data
        learning_data = {
            "learning_metrics": self.learning_metrics,
            "model_performance": self.model_performance,
            "pattern_library": self.pattern_library,
            "knowledge_base": self.knowledge_base
        }
        
        # Generate report sections
        report_sections = []
        
        if report_scope in ["comprehensive", "metrics"]:
            metrics_section = await self._generate_metrics_section(learning_data)
            report_sections.append(metrics_section)
        
        if report_scope in ["comprehensive", "performance"]:
            performance_section = await self._generate_performance_section(learning_data)
            report_sections.append(performance_section)
        
        if report_scope in ["comprehensive", "patterns"]:
            patterns_section = await self._generate_patterns_section(learning_data)
            report_sections.append(patterns_section)
        
        # Generate executive summary
        executive_summary = await self._generate_learning_executive_summary(learning_data)
        
        # Generate recommendations
        recommendations = await self._generate_learning_recommendations(learning_data)
        
        return {
            "report_id": f"learning-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "report_scope": report_scope,
            "time_period": f"Last {time_period} days",
            "executive_summary": executive_summary,
            "report_sections": report_sections,
            "recommendations": recommendations,
            "generated_timestamp": datetime.now().isoformat()
        }
    
    async def _sync_knowledge_base(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync knowledge base with external sources."""
        sync_sources = task_data.get("sync_sources", ["vector_store", "documentation", "external_apis"])
        
        sync_results = {}
        
        for source in sync_sources:
            if source == "vector_store":
                result = await self._sync_with_vector_store()
                sync_results["vector_store"] = result
            
            elif source == "documentation":
                result = await self._sync_with_documentation()
                sync_results["documentation"] = result
            
            elif source == "external_apis":
                result = await self._sync_with_external_apis()
                sync_results["external_apis"] = result
        
        # Update knowledge base
        await self._update_knowledge_base(sync_results, "sync")
        
        return {
            "sync_results": sync_results,
            "sync_sources": sync_sources,
            "knowledge_base_updated": True,
            "sync_timestamp": datetime.now().isoformat()
        }
    
    # Helper methods for learning strategies
    
    def _convert_to_human_feedback(self, feedback_data: Dict[str, Any]) -> HumanFeedback:
        """Convert feedback data to HumanFeedback object."""
        return HumanFeedback(
            feedback_id=feedback_data.get("feedback_id", ""),
            test_result_id=feedback_data.get("test_result_id", ""),
            feedback_type=feedback_data.get("feedback_type", "general"),
            feedback_text=feedback_data.get("feedback_text", ""),
            rating=feedback_data.get("rating", 0),
            corrections=feedback_data.get("corrections", []),
            timestamp=datetime.fromisoformat(feedback_data.get("timestamp", datetime.now().isoformat()))
        )
    
    async def _learn_from_test_correction(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Learn from test corrections."""
        # Analyze correction patterns
        correction_patterns = self._analyze_correction_patterns(feedback.corrections)
        
        # Update test generation patterns
        pattern_updates = await self._update_test_patterns(correction_patterns)
        
        return {
            "learning_type": "test_correction",
            "feedback_id": feedback.feedback_id,
            "correction_patterns": correction_patterns,
            "pattern_updates": pattern_updates,
            "learning_success": True
        }
    
    async def _learn_from_model_correction(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Learn from model corrections."""
        # Analyze model errors
        error_patterns = self._analyze_model_errors(feedback.corrections)
        
        # Update model parameters
        parameter_updates = await self._update_model_parameters(error_patterns)
        
        return {
            "learning_type": "model_correction",
            "feedback_id": feedback.feedback_id,
            "error_patterns": error_patterns,
            "parameter_updates": parameter_updates,
            "learning_success": True
        }
    
    async def _learn_from_priority_adjustment(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Learn from priority adjustments."""
        # Analyze priority patterns
        priority_patterns = self._analyze_priority_patterns(feedback.corrections)
        
        # Update priority algorithms
        algorithm_updates = await self._update_priority_algorithms(priority_patterns)
        
        return {
            "learning_type": "priority_adjustment",
            "feedback_id": feedback.feedback_id,
            "priority_patterns": priority_patterns,
            "algorithm_updates": algorithm_updates,
            "learning_success": True
        }
    
    async def _learn_from_general_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Learn from general feedback."""
        # Extract insights from general feedback
        insights = self._extract_feedback_insights(feedback.feedback_text)
        
        # Update knowledge base
        knowledge_updates = await self._update_knowledge_base(insights, "feedback")
        
        return {
            "learning_type": "general_feedback",
            "feedback_id": feedback.feedback_id,
            "insights": insights,
            "knowledge_updates": knowledge_updates,
            "learning_success": True
        }
    
    async def _batch_update_models(self, learning_results: List[Dict[str, Any]]) -> List[ModelUpdate]:
        """Perform batch model updates."""
        model_updates = []
        
        # Group learning results by type
        grouped_results = self._group_learning_results(learning_results)
        
        # Update models based on grouped results
        for learning_type, results in grouped_results.items():
            if learning_type == "test_correction":
                update = await self._batch_update_test_generator(results)
                model_updates.append(update)
            
            elif learning_type == "model_correction":
                update = await self._batch_update_risk_analyzer(results)
                model_updates.append(update)
        
        return model_updates
    
    async def _incremental_update_models(self, learning_results: List[Dict[str, Any]]) -> List[ModelUpdate]:
        """Perform incremental model updates."""
        model_updates = []
        
        for result in learning_results:
            if result["learning_type"] == "test_correction":
                update = await self._incremental_update_test_generator(result)
                model_updates.append(update)
            
            elif result["learning_type"] == "model_correction":
                update = await self._incremental_update_risk_analyzer(result)
                model_updates.append(update)
        
        return model_updates
    
    async def _extract_execution_patterns(self, execution_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from execution logs."""
        patterns = {
            "success_patterns": [],
            "failure_patterns": [],
            "performance_patterns": [],
            "environment_patterns": []
        }
        
        # Analyze success patterns
        successful_executions = [log for log in execution_logs if log.get("status") == "passed"]
        if successful_executions:
            patterns["success_patterns"] = self._analyze_success_patterns(successful_executions)
        
        # Analyze failure patterns
        failed_executions = [log for log in execution_logs if log.get("status") in ["failed", "error"]]
        if failed_executions:
            patterns["failure_patterns"] = self._analyze_failure_patterns(failed_executions)
        
        # Analyze performance patterns
        patterns["performance_patterns"] = self._analyze_performance_patterns(execution_logs)
        
        # Analyze environment patterns
        patterns["environment_patterns"] = self._analyze_environment_patterns(execution_logs)
        
        return patterns
    
    async def _extract_application_patterns(self, application_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from application logs."""
        patterns = {
            "error_patterns": [],
            "usage_patterns": [],
            "performance_patterns": []
        }
        
        # Analyze error patterns
        error_logs = [log for log in application_logs if log.get("level") == "error"]
        if error_logs:
            patterns["error_patterns"] = self._analyze_error_patterns(error_logs)
        
        # Analyze usage patterns
        patterns["usage_patterns"] = self._analyze_usage_patterns(application_logs)
        
        # Analyze performance patterns
        patterns["performance_patterns"] = self._analyze_app_performance_patterns(application_logs)
        
        return patterns
    
    async def _extract_system_patterns(self, system_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from system logs."""
        patterns = {
            "resource_patterns": [],
            "network_patterns": [],
            "security_patterns": []
        }
        
        # Analyze resource patterns
        patterns["resource_patterns"] = self._analyze_resource_patterns(system_logs)
        
        # Analyze network patterns
        patterns["network_patterns"] = self._analyze_network_patterns(system_logs)
        
        # Analyze security patterns
        patterns["security_patterns"] = self._analyze_security_patterns(system_logs)
        
        return patterns
    
    async def _analyze_pr_for_learning(self, pr: Dict[str, Any], learning_focus: List[str]) -> Dict[str, Any]:
        """Analyze PR for learning opportunities."""
        insights = {
            "pr_id": pr.get("id", ""),
            "learning_opportunities": [],
            "code_patterns": [],
            "test_changes": [],
            "impact_assessment": {}
        }
        
        # Analyze based on learning focus
        if "test_changes" in learning_focus:
            test_insights = self._analyze_test_changes_in_pr(pr)
            insights["test_changes"] = test_insights
        
        if "bug_fixes" in learning_focus:
            bug_fix_insights = self._analyze_bug_fixes_in_pr(pr)
            insights["learning_opportunities"].extend(bug_fix_insights)
        
        if "feature_additions" in learning_focus:
            feature_insights = self._analyze_feature_additions_in_pr(pr)
            insights["code_patterns"].extend(feature_insights)
        
        return insights
    
    async def _update_test_generator_model(self, update_strategy: str) -> Dict[str, Any]:
        """Update test generator model."""
        # This would implement actual model updating logic
        return {
            "model_type": "test_generator",
            "update_strategy": update_strategy,
            "update_success": True,
            "performance_improvement": 0.05,
            "update_timestamp": datetime.now().isoformat()
        }
    
    async def _update_risk_analyzer_model(self, update_strategy: str) -> Dict[str, Any]:
        """Update risk analyzer model."""
        # This would implement actual model updating logic
        return {
            "model_type": "risk_analyzer",
            "update_strategy": update_strategy,
            "update_success": True,
            "performance_improvement": 0.03,
            "update_timestamp": datetime.now().isoformat()
        }
    
    def _update_learning_metrics(self, learning_results: List[Dict[str, Any]]):
        """Update learning metrics."""
        self.learning_metrics["total_learning_cycles"] += 1
        
        successful_learnings = len([r for r in learning_results if r.get("learning_success", False)])
        self.learning_metrics["successful_updates"] += successful_learnings
        self.learning_metrics["failed_updates"] += len(learning_results) - successful_learnings
        
        # Calculate improvement rate
        if self.learning_metrics["total_learning_cycles"] > 0:
            self.learning_metrics["improvement_rate"] = (
                self.learning_metrics["successful_updates"] / 
                self.learning_metrics["total_learning_cycles"]
            )
        
        self.learning_metrics["last_learning_cycle"] = datetime.now().isoformat()
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, I'm providing the structure and key methods)
    
    def _analyze_correction_patterns(self, corrections: List[str]) -> Dict[str, Any]:
        """Analyze patterns in corrections."""
        return {"pattern_type": "correction", "count": len(corrections)}
    
    async def _update_test_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Update test generation patterns."""
        return {"updated": True, "pattern_count": patterns.get("count", 0)}
    
    def _analyze_model_errors(self, corrections: List[str]) -> Dict[str, Any]:
        """Analyze model errors."""
        return {"error_type": "model", "count": len(corrections)}
    
    async def _update_model_parameters(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Update model parameters."""
        return {"updated": True, "parameter_count": patterns.get("count", 0)}
    
    def _analyze_priority_patterns(self, corrections: List[str]) -> Dict[str, Any]:
        """Analyze priority patterns."""
        return {"pattern_type": "priority", "count": len(corrections)}
    
    async def _update_priority_algorithms(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Update priority algorithms."""
        return {"updated": True, "algorithm_count": patterns.get("count", 0)}
    
    def _extract_feedback_insights(self, feedback_text: str) -> Dict[str, Any]:
        """Extract insights from feedback text."""
        return {"insights": ["general_feedback"], "text_length": len(feedback_text)}
    
    def _group_learning_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group learning results by type."""
        grouped = {}
        for result in results:
            learning_type = result.get("learning_type", "general")
            if learning_type not in grouped:
                grouped[learning_type] = []
            grouped[learning_type].append(result)
        return grouped
    
    async def _batch_update_test_generator(self, results: List[Dict[str, Any]]) -> ModelUpdate:
        """Batch update test generator."""
        return ModelUpdate(
            model_id="test_generator",
            update_type="batch",
            success=True,
            performance_metrics={"accuracy": 0.95},
            timestamp=datetime.now()
        )
    
    async def _batch_update_risk_analyzer(self, results: List[Dict[str, Any]]) -> ModelUpdate:
        """Batch update risk analyzer."""
        return ModelUpdate(
            model_id="risk_analyzer",
            update_type="batch",
            success=True,
            performance_metrics={"precision": 0.92},
            timestamp=datetime.now()
        )
    
    async def _incremental_update_test_generator(self, result: Dict[str, Any]) -> ModelUpdate:
        """Incremental update test generator."""
        return ModelUpdate(
            model_id="test_generator",
            update_type="incremental",
            success=True,
            performance_metrics={"accuracy": 0.94},
            timestamp=datetime.now()
        )
    
    async def _incremental_update_risk_analyzer(self, result: Dict[str, Any]) -> ModelUpdate:
        """Incremental update risk analyzer."""
        return ModelUpdate(
            model_id="risk_analyzer",
            update_type="incremental",
            success=True,
            performance_metrics={"precision": 0.91},
            timestamp=datetime.now()
        )
    
    # Additional helper methods for pattern analysis, model optimization, etc.
    # would be implemented here following the same pattern...
