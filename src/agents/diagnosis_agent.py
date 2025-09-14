"""Diagnosis Agent - clusters results, triages bugs, and suggests fix paths."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

from .base_agent import BaseAgent, AgentCapabilities, MessageType, AgentMessage
from ..dashboard.failure_clusterer import FailureClusterer, FailureCluster
from ..dashboard.root_cause_analyzer import RootCauseAnalyzer, RootCauseAnalysis
from ..observability.risk_analyzer import RiskAnalyzer
from ..test_execution.test_runner import TestResult

logger = logging.getLogger(__name__)


class DiagnosisAgent(BaseAgent):
    """Agent responsible for clustering test results, triaging bugs, and suggesting fixes."""
    
    def __init__(self, agent_id: str = "diagnosis-001"):
        capabilities = AgentCapabilities(
            can_diagnose_failures=True,
            supported_platforms=["all"],  # Can analyze failures from any platform
            max_concurrent_tasks=3
        )
        
        super().__init__(agent_id, "Diagnosis Agent", capabilities)
        
        # Initialize components
        self.failure_clusterer = FailureClusterer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        # Diagnosis strategies
        self.diagnosis_strategies = {
            "failure_clustering": self._cluster_failures,
            "root_cause_analysis": self._analyze_root_causes,
            "bug_triage": self._triage_bugs,
            "fix_suggestions": self._suggest_fixes,
            "trend_analysis": self._analyze_failure_trends,
            "impact_assessment": self._assess_failure_impact
        }
        
        # Historical data for trend analysis
        self.failure_history = []
        self.cluster_patterns = {}
        self.bug_triage_history = []
        
        logger.info(f"Diagnosis Agent initialized with {len(self.diagnosis_strategies)} strategies")
    
    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this agent can handle a specific task type."""
        return task_type in [
            "cluster_test_failures",
            "analyze_root_causes",
            "triage_bugs",
            "suggest_fixes",
            "analyze_failure_trends",
            "assess_failure_impact",
            "generate_diagnosis_report",
            "escalate_critical_issues"
        ]
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a diagnosis task."""
        task_type = task_data.get("task_type", "")
        
        logger.info(f"Diagnosis Agent processing task: {task_type}")
        
        try:
            if task_type == "cluster_test_failures":
                return await self._cluster_test_failures(task_data)
            elif task_type == "analyze_root_causes":
                return await self._analyze_root_causes(task_data)
            elif task_type == "triage_bugs":
                return await self._triage_bugs(task_data)
            elif task_type == "suggest_fixes":
                return await self._suggest_fixes(task_data)
            elif task_type == "analyze_failure_trends":
                return await self._analyze_failure_trends(task_data)
            elif task_type == "assess_failure_impact":
                return await self._assess_failure_impact(task_data)
            elif task_type == "generate_diagnosis_report":
                return await self._generate_diagnosis_report(task_data)
            elif task_type == "escalate_critical_issues":
                return await self._escalate_critical_issues(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error processing task {task_type}: {e}")
            raise
    
    async def _cluster_test_failures(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster test failures to identify patterns."""
        test_results = task_data.get("test_results", [])
        clustering_method = task_data.get("clustering_method", "auto")
        min_cluster_size = task_data.get("min_cluster_size", 2)
        
        # Convert test results to failure data format
        failure_data = self._convert_test_results_to_failure_data(test_results)
        
        if not failure_data:
            return {
                "clusters": [],
                "clustering_method": clustering_method,
                "total_failures": 0,
                "message": "No failures to cluster"
            }
        
        # Perform clustering
        clustering_result = self.failure_clusterer.cluster_failures(failure_data)
        
        # Filter clusters by minimum size
        filtered_clusters = [
            cluster for cluster in clustering_result.clusters 
            if cluster.size >= min_cluster_size
        ]
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(filtered_clusters)
        
        # Store in history for trend analysis
        self.failure_history.extend(failure_data)
        
        return {
            "clusters": [self._cluster_to_dict(cluster) for cluster in filtered_clusters],
            "clustering_method": clustering_result.method_used,
            "silhouette_score": clustering_result.silhouette_score,
            "total_failures": len(failure_data),
            "clustered_failures": sum(cluster.size for cluster in filtered_clusters),
            "cluster_analysis": cluster_analysis,
            "clustering_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_root_causes(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze root causes of test failures."""
        clusters_data = task_data.get("clusters", [])
        
        if not clusters_data:
            return {
                "root_causes": [],
                "message": "No clusters provided for root cause analysis"
            }
        
        # Convert cluster data back to FailureCluster objects
        clusters = self._dict_to_clusters(clusters_data)
        
        # Perform root cause analysis
        root_cause_analyses = self.root_cause_analyzer.analyze_root_causes(clusters)
        
        # Generate summary
        summary = self.root_cause_analyzer.generate_summary(root_cause_analyses)
        
        # Identify critical issues requiring escalation
        critical_issues = [
            analysis for analysis in root_cause_analyses 
            if analysis.confidence_score > 0.8 and "critical" in analysis.primary_cause.lower()
        ]
        
        return {
            "root_causes": [self._root_cause_to_dict(analysis) for analysis in root_cause_analyses],
            "summary": {
                "total_clusters": summary.total_clusters,
                "analyzed_clusters": summary.analyzed_clusters,
                "common_causes": summary.common_causes,
                "top_recommendations": summary.top_recommendations,
                "overall_confidence": summary.overall_confidence
            },
            "critical_issues": len(critical_issues),
            "escalation_needed": len(critical_issues) > 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _triage_bugs(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Triage bugs based on severity and impact."""
        bugs_data = task_data.get("bugs", [])
        triage_criteria = task_data.get("triage_criteria", {
            "severity_threshold": 0.7,
            "impact_threshold": 0.8,
            "frequency_threshold": 5
        })
        
        if not bugs_data:
            return {
                "triaged_bugs": [],
                "triage_summary": {},
                "message": "No bugs provided for triage"
            }
        
        triaged_bugs = []
        
        for bug in bugs_data:
            # Calculate triage score
            triage_score = self._calculate_triage_score(bug, triage_criteria)
            
            # Determine priority
            priority = self._determine_bug_priority(triage_score, bug)
            
            # Assign to team member if needed
            assignee = self._suggest_assignee(bug, priority)
            
            # Generate triage recommendations
            recommendations = self._generate_triage_recommendations(bug, triage_score)
            
            triaged_bug = {
                "bug_id": bug.get("id", ""),
                "original_bug": bug,
                "triage_score": triage_score,
                "priority": priority,
                "assignee": assignee,
                "recommendations": recommendations,
                "triage_timestamp": datetime.now().isoformat()
            }
            
            triaged_bugs.append(triaged_bug)
        
        # Sort by priority and triage score
        triaged_bugs.sort(key=lambda x: (x["priority"], x["triage_score"]), reverse=True)
        
        # Generate triage summary
        triage_summary = self._generate_triage_summary(triaged_bugs)
        
        # Store in history
        self.bug_triage_history.extend(triaged_bugs)
        
        return {
            "triaged_bugs": triaged_bugs,
            "triage_summary": triage_summary,
            "triage_criteria": triage_criteria,
            "triage_timestamp": datetime.now().isoformat()
        }
    
    async def _suggest_fixes(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest fixes for identified issues."""
        issues = task_data.get("issues", [])
        fix_strategy = task_data.get("fix_strategy", "comprehensive")
        
        if not issues:
            return {
                "suggested_fixes": [],
                "message": "No issues provided for fix suggestions"
            }
        
        suggested_fixes = []
        
        for issue in issues:
            # Analyze issue type and generate fix suggestions
            fix_suggestions = self._generate_fix_suggestions(issue, fix_strategy)
            
            # Estimate fix complexity and effort
            complexity_assessment = self._assess_fix_complexity(issue, fix_suggestions)
            
            # Suggest implementation approach
            implementation_approach = self._suggest_implementation_approach(issue, fix_suggestions)
            
            suggested_fix = {
                "issue_id": issue.get("id", ""),
                "issue_description": issue.get("description", ""),
                "fix_suggestions": fix_suggestions,
                "complexity_assessment": complexity_assessment,
                "implementation_approach": implementation_approach,
                "estimated_effort": self._estimate_fix_effort(complexity_assessment),
                "suggested_timeline": self._suggest_fix_timeline(complexity_assessment),
                "fix_timestamp": datetime.now().isoformat()
            }
            
            suggested_fixes.append(suggested_fix)
        
        return {
            "suggested_fixes": suggested_fixes,
            "fix_strategy": fix_strategy,
            "total_issues": len(issues),
            "high_complexity_fixes": len([f for f in suggested_fixes if f["complexity_assessment"]["level"] == "high"]),
            "fix_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_failure_trends(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure trends over time."""
        time_period = task_data.get("time_period", 7)  # days
        trend_type = task_data.get("trend_type", "all")
        
        # Filter failure history by time period
        cutoff_date = datetime.now() - timedelta(days=time_period)
        recent_failures = [
            f for f in self.failure_history 
            if datetime.fromisoformat(f.get("timestamp", "")) > cutoff_date
        ]
        
        if not recent_failures:
            return {
                "trends": [],
                "message": f"No failures found in the last {time_period} days"
            }
        
        # Analyze trends
        trends = self._calculate_failure_trends(recent_failures, time_period)
        
        # Identify concerning patterns
        concerning_patterns = self._identify_concerning_patterns(trends)
        
        # Generate trend recommendations
        trend_recommendations = self._generate_trend_recommendations(trends, concerning_patterns)
        
        return {
            "trends": trends,
            "concerning_patterns": concerning_patterns,
            "trend_recommendations": trend_recommendations,
            "analysis_period": f"Last {time_period} days",
            "total_failures_analyzed": len(recent_failures),
            "trend_analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _assess_failure_impact(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of test failures."""
        failures = task_data.get("failures", [])
        impact_criteria = task_data.get("impact_criteria", {
            "user_facing": True,
            "business_critical": True,
            "data_loss": True,
            "security_impact": True
        })
        
        if not failures:
            return {
                "impact_assessment": [],
                "message": "No failures provided for impact assessment"
            }
        
        impact_assessments = []
        
        for failure in failures:
            # Assess various impact dimensions
            impact_score = self._calculate_impact_score(failure, impact_criteria)
            
            # Determine impact level
            impact_level = self._determine_impact_level(impact_score)
            
            # Identify affected users/components
            affected_scope = self._assess_affected_scope(failure)
            
            # Suggest mitigation strategies
            mitigation_strategies = self._suggest_mitigation_strategies(failure, impact_score)
            
            impact_assessment = {
                "failure_id": failure.get("id", ""),
                "impact_score": impact_score,
                "impact_level": impact_level,
                "affected_scope": affected_scope,
                "mitigation_strategies": mitigation_strategies,
                "requires_immediate_attention": impact_score > 0.8,
                "assessment_timestamp": datetime.now().isoformat()
            }
            
            impact_assessments.append(impact_assessment)
        
        # Sort by impact score
        impact_assessments.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return {
            "impact_assessments": impact_assessments,
            "high_impact_failures": len([a for a in impact_assessments if a["impact_score"] > 0.7]),
            "critical_failures": len([a for a in impact_assessments if a["requires_immediate_attention"]]),
            "impact_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_diagnosis_report(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive diagnosis report."""
        report_scope = task_data.get("scope", "comprehensive")
        time_period = task_data.get("time_period", 30)  # days
        
        report_sections = []
        
        if report_scope in ["comprehensive", "failures"]:
            # Failure clustering analysis
            recent_failures = self._get_recent_failures(time_period)
            if recent_failures:
                clustering_result = await self._cluster_test_failures({
                    "test_results": recent_failures,
                    "clustering_method": "auto"
                })
                report_sections.append({
                    "section": "failure_clustering",
                    "data": clustering_result
                })
        
        if report_scope in ["comprehensive", "trends"]:
            # Trend analysis
            trend_result = await self._analyze_failure_trends({
                "time_period": time_period,
                "trend_type": "all"
            })
            report_sections.append({
                "section": "failure_trends",
                "data": trend_result
            })
        
        if report_scope in ["comprehensive", "root_causes"]:
            # Root cause analysis
            if "clusters" in locals():
                root_cause_result = await self._analyze_root_causes({
                    "clusters": clustering_result.get("clusters", [])
                })
                report_sections.append({
                    "section": "root_cause_analysis",
                    "data": root_cause_result
                })
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(report_sections)
        
        # Generate recommendations
        recommendations = self._generate_report_recommendations(report_sections)
        
        return {
            "report_id": f"diagnosis-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "report_scope": report_scope,
            "time_period": f"Last {time_period} days",
            "executive_summary": executive_summary,
            "report_sections": report_sections,
            "recommendations": recommendations,
            "generated_timestamp": datetime.now().isoformat()
        }
    
    async def _escalate_critical_issues(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate critical issues that require immediate attention."""
        issues = task_data.get("issues", [])
        escalation_criteria = task_data.get("escalation_criteria", {
            "severity_threshold": 0.9,
            "impact_threshold": 0.8,
            "urgency_threshold": 0.7
        })
        
        critical_issues = []
        
        for issue in issues:
            escalation_score = self._calculate_escalation_score(issue, escalation_criteria)
            
            if escalation_score > 0.7:  # Threshold for escalation
                escalation_details = {
                    "issue_id": issue.get("id", ""),
                    "escalation_score": escalation_score,
                    "escalation_reason": self._get_escalation_reason(issue, escalation_criteria),
                    "recommended_action": self._get_recommended_action(issue),
                    "priority_level": self._get_priority_level(escalation_score),
                    "escalation_timestamp": datetime.now().isoformat()
                }
                
                critical_issues.append(escalation_details)
        
        # Sort by escalation score
        critical_issues.sort(key=lambda x: x["escalation_score"], reverse=True)
        
        # Send escalation notifications
        escalation_notifications = await self._send_escalation_notifications(critical_issues)
        
        return {
            "escalated_issues": critical_issues,
            "escalation_notifications": escalation_notifications,
            "total_escalated": len(critical_issues),
            "escalation_criteria": escalation_criteria,
            "escalation_timestamp": datetime.now().isoformat()
        }
    
    # Helper methods
    
    def _convert_test_results_to_failure_data(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert test results to failure data format for clustering."""
        failure_data = []
        
        for result in test_results:
            if result.get("status") in ["failed", "error"]:
                failure_data.append({
                    "test_name": result.get("scenario_id", ""),
                    "error_message": result.get("error_message", ""),
                    "stack_trace": result.get("stack_trace", ""),
                    "duration": result.get("duration", 0),
                    "timestamp": result.get("timestamp", datetime.now().isoformat()),
                    "environment": result.get("environment", ""),
                    "browser": result.get("browser", ""),
                    "platform": result.get("platform", "")
                })
        
        return failure_data
    
    def _analyze_clusters(self, clusters: List[FailureCluster]) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        if not clusters:
            return {}
        
        total_failures = sum(cluster.size for cluster in clusters)
        largest_cluster = max(clusters, key=lambda c: c.size)
        
        return {
            "total_clusters": len(clusters),
            "total_failures": total_failures,
            "largest_cluster_size": largest_cluster.size,
            "average_cluster_size": total_failures / len(clusters),
            "cluster_size_distribution": [c.size for c in clusters],
            "high_confidence_clusters": len([c for c in clusters if c.confidence_score > 0.7])
        }
    
    def _cluster_to_dict(self, cluster: FailureCluster) -> Dict[str, Any]:
        """Convert FailureCluster to dictionary."""
        return {
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "confidence_score": cluster.confidence_score,
            "common_patterns": cluster.common_patterns,
            "representative_failure": cluster.representative_failure,
            "failures_count": len(cluster.failures)
        }
    
    def _dict_to_clusters(self, clusters_data: List[Dict[str, Any]]) -> List[FailureCluster]:
        """Convert cluster dictionaries back to FailureCluster objects."""
        clusters = []
        
        for cluster_data in clusters_data:
            # This is a simplified conversion - in practice, you'd need more data
            cluster = FailureCluster(
                cluster_id=cluster_data.get("cluster_id", ""),
                size=cluster_data.get("size", 0),
                centroid=None,  # Would need actual centroid data
                failures=[],  # Would need actual failure data
                common_patterns=cluster_data.get("common_patterns", []),
                representative_failure=cluster_data.get("representative_failure", {}),
                confidence_score=cluster_data.get("confidence_score", 0.0)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _root_cause_to_dict(self, analysis: RootCauseAnalysis) -> Dict[str, Any]:
        """Convert RootCauseAnalysis to dictionary."""
        return {
            "cluster_id": analysis.cluster_id,
            "primary_cause": analysis.primary_cause,
            "contributing_factors": analysis.contributing_factors,
            "confidence_score": analysis.confidence_score,
            "recommendations": analysis.recommendations,
            "similar_incidents": analysis.similar_incidents,
            "prevention_strategies": analysis.prevention_strategies
        }
    
    def _calculate_triage_score(self, bug: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate triage score for a bug."""
        score = 0.0
        
        # Severity factor
        severity = bug.get("severity", "medium").lower()
        severity_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
        score += severity_scores.get(severity, 0.5) * 0.4
        
        # Impact factor
        impact = bug.get("impact", 0.5)
        score += impact * 0.3
        
        # Frequency factor
        frequency = bug.get("frequency", 1)
        frequency_score = min(frequency / 10.0, 1.0)  # Normalize to 0-1
        score += frequency_score * 0.2
        
        # User facing factor
        if bug.get("user_facing", False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _determine_bug_priority(self, triage_score: float, bug: Dict[str, Any]) -> str:
        """Determine bug priority based on triage score."""
        if triage_score > 0.8:
            return "critical"
        elif triage_score > 0.6:
            return "high"
        elif triage_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _suggest_assignee(self, bug: Dict[str, Any], priority: str) -> Optional[str]:
        """Suggest assignee for bug based on component and priority."""
        component = bug.get("component", "")
        
        # Simple assignment logic - in practice, this would be more sophisticated
        if component in ["authentication", "security"]:
            return "security-team"
        elif component in ["ui", "frontend"]:
            return "frontend-team"
        elif component in ["api", "backend"]:
            return "backend-team"
        elif priority == "critical":
            return "senior-developer"
        else:
            return None  # Auto-assign later
    
    def _generate_triage_recommendations(self, bug: Dict[str, Any], triage_score: float) -> List[str]:
        """Generate triage recommendations for a bug."""
        recommendations = []
        
        if triage_score > 0.8:
            recommendations.append("Immediate attention required")
            recommendations.append("Consider hotfix deployment")
        
        if bug.get("user_facing", False):
            recommendations.append("User communication needed")
        
        if bug.get("frequency", 1) > 5:
            recommendations.append("Investigate root cause")
        
        if bug.get("component") in ["authentication", "payment"]:
            recommendations.append("Security review recommended")
        
        return recommendations
    
    def _generate_triage_summary(self, triaged_bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of bug triage results."""
        total_bugs = len(triaged_bugs)
        priority_distribution = {}
        
        for bug in triaged_bugs:
            priority = bug["priority"]
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        return {
            "total_bugs": total_bugs,
            "priority_distribution": priority_distribution,
            "average_triage_score": sum(b["triage_score"] for b in triaged_bugs) / total_bugs if total_bugs > 0 else 0,
            "critical_bugs": priority_distribution.get("critical", 0),
            "high_priority_bugs": priority_distribution.get("high", 0)
        }
    
    def _generate_fix_suggestions(self, issue: Dict[str, Any], strategy: str) -> List[str]:
        """Generate fix suggestions for an issue."""
        suggestions = []
        
        issue_type = issue.get("type", "")
        description = issue.get("description", "").lower()
        
        # Generic suggestions based on issue type
        if "timeout" in description:
            suggestions.extend([
                "Increase timeout values",
                "Add explicit waits",
                "Optimize test execution speed"
            ])
        
        if "element not found" in description:
            suggestions.extend([
                "Update element locators",
                "Add dynamic wait strategies",
                "Verify element visibility"
            ])
        
        if "network" in description:
            suggestions.extend([
                "Check network connectivity",
                "Add retry mechanisms",
                "Implement network mocking"
            ])
        
        if issue_type == "flaky_test":
            suggestions.extend([
                "Add more robust waits",
                "Improve test isolation",
                "Reduce test dependencies"
            ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _assess_fix_complexity(self, issue: Dict[str, Any], suggestions: List[str]) -> Dict[str, Any]:
        """Assess the complexity of implementing fixes."""
        complexity_factors = {
            "simple": 0,
            "medium": 0,
            "high": 0
        }
        
        for suggestion in suggestions:
            if any(keyword in suggestion.lower() for keyword in ["increase", "add", "check"]):
                complexity_factors["simple"] += 1
            elif any(keyword in suggestion.lower() for keyword in ["implement", "optimize", "update"]):
                complexity_factors["medium"] += 1
            elif any(keyword in suggestion.lower() for keyword in ["refactor", "restructure", "redesign"]):
                complexity_factors["high"] += 1
        
        level = max(complexity_factors, key=complexity_factors.get)
        
        return {
            "level": level,
            "factors": complexity_factors,
            "estimated_effort_hours": {"simple": 2, "medium": 8, "high": 24}[level]
        }
    
    def _suggest_implementation_approach(self, issue: Dict[str, Any], suggestions: List[str]) -> str:
        """Suggest implementation approach for fixes."""
        if len(suggestions) == 1:
            return "Direct fix"
        elif len(suggestions) <= 3:
            return "Incremental approach"
        else:
            return "Comprehensive refactoring"
    
    def _estimate_fix_effort(self, complexity_assessment: Dict[str, Any]) -> str:
        """Estimate fix effort based on complexity."""
        level = complexity_assessment["level"]
        hours = complexity_assessment["estimated_effort_hours"]
        
        if hours <= 4:
            return "Low effort"
        elif hours <= 16:
            return "Medium effort"
        else:
            return "High effort"
    
    def _suggest_fix_timeline(self, complexity_assessment: Dict[str, Any]) -> str:
        """Suggest timeline for implementing fixes."""
        level = complexity_assessment["level"]
        
        if level == "simple":
            return "1-2 days"
        elif level == "medium":
            return "1 week"
        else:
            return "2-4 weeks"
    
    def _get_recent_failures(self, time_period: int) -> List[Dict[str, Any]]:
        """Get recent failures from history."""
        cutoff_date = datetime.now() - timedelta(days=time_period)
        return [
            f for f in self.failure_history
            if datetime.fromisoformat(f.get("timestamp", "")) > cutoff_date
        ]
    
    def _calculate_failure_trends(self, failures: List[Dict[str, Any]], time_period: int) -> List[Dict[str, Any]]:
        """Calculate failure trends over time."""
        # Group failures by day
        daily_failures = {}
        
        for failure in failures:
            failure_date = datetime.fromisoformat(failure.get("timestamp", "")).date()
            daily_failures[failure_date] = daily_failures.get(failure_date, 0) + 1
        
        # Calculate trend
        dates = sorted(daily_failures.keys())
        if len(dates) >= 2:
            recent_avg = sum(daily_failures[d] for d in dates[-3:]) / min(3, len(dates))
            earlier_avg = sum(daily_failures[d] for d in dates[:-3]) / max(1, len(dates) - 3)
            trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing"
        else:
            trend_direction = "stable"
        
        return [{
            "date": str(date),
            "failure_count": daily_failures[date]
        } for date in dates] + [{
            "trend_direction": trend_direction,
            "total_period_failures": len(failures)
        }]
    
    def _identify_concerning_patterns(self, trends: List[Dict[str, Any]]) -> List[str]:
        """Identify concerning patterns in failure trends."""
        patterns = []
        
        if trends and trends[-1].get("trend_direction") == "increasing":
            patterns.append("Increasing failure trend detected")
        
        # Check for high failure counts
        high_failure_days = [t for t in trends if isinstance(t.get("failure_count"), int) and t["failure_count"] > 10]
        if len(high_failure_days) > 2:
            patterns.append("Multiple high-failure days detected")
        
        return patterns
    
    def _generate_trend_recommendations(self, trends: List[Dict[str, Any]], patterns: List[str]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if "Increasing failure trend detected" in patterns:
            recommendations.extend([
                "Investigate root causes of increasing failures",
                "Consider additional monitoring",
                "Review recent code changes"
            ])
        
        if "Multiple high-failure days detected" in patterns:
            recommendations.extend([
                "Schedule maintenance window",
                "Review test stability",
                "Consider test environment improvements"
            ])
        
        return recommendations
    
    def _calculate_impact_score(self, failure: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate impact score for a failure."""
        score = 0.0
        
        # User facing impact
        if failure.get("user_facing", False) and criteria.get("user_facing", True):
            score += 0.3
        
        # Business critical impact
        if failure.get("business_critical", False) and criteria.get("business_critical", True):
            score += 0.3
        
        # Data loss impact
        if failure.get("data_loss", False) and criteria.get("data_loss", True):
            score += 0.2
        
        # Security impact
        if failure.get("security_impact", False) and criteria.get("security_impact", True):
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_impact_level(self, impact_score: float) -> str:
        """Determine impact level based on score."""
        if impact_score > 0.8:
            return "critical"
        elif impact_score > 0.6:
            return "high"
        elif impact_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _assess_affected_scope(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the scope of impact."""
        return {
            "affected_components": failure.get("components", []),
            "affected_users": failure.get("affected_users", "unknown"),
            "affected_environments": failure.get("environments", []),
            "geographic_impact": failure.get("geographic_impact", "unknown")
        }
    
    def _suggest_mitigation_strategies(self, failure: Dict[str, Any], impact_score: float) -> List[str]:
        """Suggest mitigation strategies for a failure."""
        strategies = []
        
        if impact_score > 0.8:
            strategies.extend([
                "Implement immediate rollback",
                "Activate incident response team",
                "Communicate with stakeholders"
            ])
        
        if failure.get("user_facing", False):
            strategies.append("Deploy user communication")
        
        if failure.get("data_loss", False):
            strategies.append("Initiate data recovery procedures")
        
        return strategies
    
    def _generate_executive_summary(self, report_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary for diagnosis report."""
        total_failures = 0
        total_clusters = 0
        
        for section in report_sections:
            if section["section"] == "failure_clustering":
                data = section["data"]
                total_failures += data.get("total_failures", 0)
                total_clusters += data.get("clusters", [])
        
        return {
            "total_failures": total_failures,
            "total_clusters": total_clusters,
            "key_findings": [
                f"Identified {total_clusters} failure clusters",
                f"Analyzed {total_failures} total failures",
                "Root cause analysis completed"
            ],
            "overall_health": "needs_attention" if total_failures > 10 else "good"
        }
    
    def _generate_report_recommendations(self, report_sections: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on report sections."""
        recommendations = []
        
        for section in report_sections:
            if section["section"] == "failure_clustering":
                data = section["data"]
                if data.get("total_failures", 0) > 20:
                    recommendations.append("High failure rate - investigate test stability")
            
            elif section["section"] == "failure_trends":
                data = section["data"]
                if data.get("concerning_patterns"):
                    recommendations.append("Address concerning failure patterns")
        
        return recommendations
    
    def _calculate_escalation_score(self, issue: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate escalation score for an issue."""
        score = 0.0
        
        # Severity factor
        severity = issue.get("severity", 0.5)
        if severity > criteria.get("severity_threshold", 0.7):
            score += 0.4
        
        # Impact factor
        impact = issue.get("impact", 0.5)
        if impact > criteria.get("impact_threshold", 0.8):
            score += 0.3
        
        # Urgency factor
        urgency = issue.get("urgency", 0.5)
        if urgency > criteria.get("urgency_threshold", 0.7):
            score += 0.3
        
        return min(score, 1.0)
    
    def _get_escalation_reason(self, issue: Dict[str, Any], criteria: Dict[str, Any]) -> str:
        """Get reason for escalation."""
        reasons = []
        
        if issue.get("severity", 0) > criteria.get("severity_threshold", 0.7):
            reasons.append("High severity")
        
        if issue.get("impact", 0) > criteria.get("impact_threshold", 0.8):
            reasons.append("High impact")
        
        if issue.get("urgency", 0) > criteria.get("urgency_threshold", 0.7):
            reasons.append("High urgency")
        
        return "; ".join(reasons) if reasons else "Manual escalation"
    
    def _get_recommended_action(self, issue: Dict[str, Any]) -> str:
        """Get recommended action for escalated issue."""
        if issue.get("severity", 0) > 0.9:
            return "Immediate intervention required"
        elif issue.get("impact", 0) > 0.8:
            return "Schedule emergency meeting"
        else:
            return "Review and prioritize"
    
    def _get_priority_level(self, escalation_score: float) -> str:
        """Get priority level based on escalation score."""
        if escalation_score > 0.9:
            return "P0 - Critical"
        elif escalation_score > 0.8:
            return "P1 - High"
        elif escalation_score > 0.7:
            return "P2 - Medium"
        else:
            return "P3 - Low"
    
    async def _send_escalation_notifications(self, critical_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send escalation notifications for critical issues."""
        notifications = []
        
        for issue in critical_issues:
            notification = {
                "issue_id": issue["issue_id"],
                "recipients": ["team-lead", "senior-developer"],
                "notification_type": "escalation",
                "priority": issue["priority_level"],
                "sent_timestamp": datetime.now().isoformat(),
                "status": "sent"
            }
            notifications.append(notification)
        
        return notifications
