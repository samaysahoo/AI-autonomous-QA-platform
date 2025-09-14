"""LangGraph-based Diagnosis Agent."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import (
    TestAutomationState, FailureCluster, LearningFeedback, Task, TaskStatus, 
    AgentStatus, create_task_from_data
)
from ..dashboard.failure_clusterer import FailureClusterer
from ..dashboard.root_cause_analyzer import RootCauseAnalyzer

logger = logging.getLogger(__name__)


class DiagnosisAgent:
    """LangGraph-based Diagnosis Agent for failure analysis and bug triage."""
    
    def __init__(self, agent_id: str = "diagnosis"):
        self.agent_id = agent_id
        self.name = "Diagnosis Agent"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize components
        self.failure_clusterer = FailureClusterer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        # Setup prompts
        self._setup_prompts()
        
        logger.info(f"Diagnosis Agent {agent_id} initialized")
    
    def _setup_prompts(self):
        """Setup LangChain prompts for the agent."""
        
        # Failure clustering prompt
        self.clustering_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Failure Analysis specialist. Analyze test failures and identify patterns for clustering.
            
            Your capabilities:
            - Analyze failure patterns and commonalities
            - Group similar failures into clusters
            - Identify representative failures
            - Assess clustering confidence
            - Suggest root cause hypotheses
            
            Focus on identifying actionable patterns that can guide debugging and prevention.
            """),
            HumanMessage(content="""
            Analyze the following test failures and create clusters:
            
            Test Failures: {test_failures}
            Clustering Method: {clustering_method}
            Minimum Cluster Size: {min_cluster_size}
            
            Provide clustering analysis in JSON format:
            {{
                "clusters": [
                    {{
                        "cluster_id": "unique_id",
                        "size": number,
                        "confidence_score": 0.0-1.0,
                        "common_patterns": ["pattern1", "pattern2"],
                        "representative_failure": {{
                            "test_name": "test_name",
                            "error_message": "error_message",
                            "stack_trace": "stack_trace"
                        }},
                        "suggested_fixes": ["fix1", "fix2"],
                        "root_cause_hypothesis": "hypothesis"
                    }}
                ],
                "clustering_quality": {{
                    "silhouette_score": 0.0-1.0,
                    "method_used": "clustering_method",
                    "total_failures": number,
                    "clustered_failures": number
                }},
                "insights": [
                    "insight1",
                    "insight2"
                ]
            }}
            """)
        ])
        
        # Root cause analysis prompt
        self.root_cause_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Root Cause Analysis specialist. Analyze failure clusters to identify underlying causes.
            
            Your expertise:
            - Deep analysis of failure patterns
            - Identification of contributing factors
            - Confidence assessment of root causes
            - Prevention strategy recommendations
            - Similar incident identification
            
            Provide actionable insights for debugging and prevention.
            """),
            HumanMessage(content="""
            Analyze root causes for the following failure clusters:
            
            Failure Clusters: {failure_clusters}
            Historical Context: {historical_context}
            System Context: {system_context}
            
            Provide root cause analysis in JSON format:
            {{
                "root_causes": [
                    {{
                        "cluster_id": "cluster_id",
                        "primary_cause": "main_root_cause",
                        "contributing_factors": ["factor1", "factor2"],
                        "confidence_score": 0.0-1.0,
                        "evidence": ["evidence1", "evidence2"],
                        "similar_incidents": ["incident1", "incident2"],
                        "prevention_strategies": ["strategy1", "strategy2"],
                        "immediate_actions": ["action1", "action2"]
                    }}
                ],
                "summary": {{
                    "total_clusters": number,
                    "analyzed_clusters": number,
                    "high_confidence_causes": number,
                    "common_causes": ["cause1", "cause2"],
                    "top_recommendations": ["rec1", "rec2"],
                    "overall_confidence": 0.0-1.0
                }},
                "escalation_needed": true|false,
                "escalation_reason": "reason_if_needed"
            }}
            """)
        ])
        
        # Bug triage prompt
        self.bug_triage_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Bug Triage specialist. Analyze and prioritize bugs for efficient resolution.
            
            Your responsibilities:
            - Assess bug severity and impact
            - Prioritize bugs based on multiple factors
            - Suggest appropriate assignees
            - Provide resolution recommendations
            - Identify patterns in bug reports
            
            Focus on maximizing team efficiency and customer impact.
            """),
            HumanMessage(content="""
            Triage the following bugs:
            
            Bugs: {bugs}
            Triage Criteria: {triage_criteria}
            Team Context: {team_context}
            
            Provide bug triage in JSON format:
            {{
                "triaged_bugs": [
                    {{
                        "bug_id": "bug_id",
                        "priority": "critical|high|medium|low",
                        "severity": "critical|high|medium|low",
                        "impact_score": 0.0-1.0,
                        "triage_score": 0.0-1.0,
                        "recommended_assignee": "team_member",
                        "estimated_effort": "hours",
                        "recommendations": ["rec1", "rec2"],
                        "related_bugs": ["bug1", "bug2"],
                        "escalation_needed": true|false
                    }}
                ],
                "triage_summary": {{
                    "total_bugs": number,
                    "priority_distribution": {{"critical": 0, "high": 0, "medium": 0, "low": 0}},
                    "average_triage_score": 0.0,
                    "escalation_count": number,
                    "team_workload": {{"team_member": "hours"}}
                }},
                "insights": ["insight1", "insight2"]
            }}
            """)
        ])
        
        # Fix suggestion prompt
        self.fix_suggestion_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Fix Suggestion specialist. Analyze issues and provide actionable fix recommendations.
            
            Your expertise:
            - Code analysis and debugging
            - Fix strategy recommendations
            - Implementation approach guidance
            - Complexity assessment
            - Timeline estimation
            
            Provide practical, implementable solutions.
            """),
            HumanMessage(content="""
            Suggest fixes for the following issues:
            
            Issues: {issues}
            Context: {context}
            Available Resources: {available_resources}
            
            Provide fix suggestions in JSON format:
            {{
                "suggested_fixes": [
                    {{
                        "issue_id": "issue_id",
                        "fix_type": "immediate|short_term|long_term",
                        "complexity": "low|medium|high",
                        "estimated_effort": "hours",
                        "fix_description": "detailed_fix_description",
                        "implementation_steps": ["step1", "step2"],
                        "testing_requirements": ["test1", "test2"],
                        "risk_assessment": "low|medium|high",
                        "success_probability": 0.0-1.0,
                        "alternative_approaches": ["alt1", "alt2"]
                    }}
                ],
                "fix_summary": {{
                    "total_issues": number,
                    "immediate_fixes": number,
                    "complex_fixes": number,
                    "estimated_total_effort": "hours",
                    "recommended_timeline": "timeline"
                }},
                "prioritization": [
                    {{"issue_id": "id", "priority": 1, "reason": "reason"}}
                ]
            }}
            """)
        ])
    
    async def cluster_test_failures(self, state: TestAutomationState) -> TestAutomationState:
        """Cluster test failures to identify patterns."""
        logger.info("Diagnosis Agent: Clustering test failures")
        
        try:
            test_results = state["test_results"]
            if not test_results:
                state["messages"].append(AIMessage(content="No test results to cluster"))
                return state
            
            # Convert test results to failure data format
            failure_data = self._convert_test_results_to_failure_data(test_results)
            
            # Use LLM for clustering analysis
            clustering_chain = self.clustering_prompt | self.llm | JsonOutputParser()
            
            result = await clustering_chain.ainvoke({
                "test_failures": failure_data,
                "clustering_method": "auto",
                "min_cluster_size": 2
            })
            
            # Convert to FailureCluster objects
            clusters = []
            for cluster_data in result.get("clusters", []):
                cluster = FailureCluster(
                    cluster_id=cluster_data["cluster_id"],
                    size=cluster_data["size"],
                    common_patterns=cluster_data["common_patterns"],
                    representative_failure=cluster_data["representative_failure"],
                    confidence_score=cluster_data["confidence_score"],
                    suggested_fixes=cluster_data.get("suggested_fixes", [])
                )
                clusters.append(cluster)
                state["failure_clusters"].append(cluster)
            
            # Add clustering task to completed
            clustering_task = create_task_from_data("cluster_failures", {
                "total_failures": len(failure_data),
                "clusters_created": len(clusters),
                "clustering_quality": result.get("clustering_quality", {}),
                "insights": result.get("insights", [])
            })
            clustering_task.status = TaskStatus.COMPLETED
            clustering_task.completed_at = datetime.now()
            state["completed_tasks"].append(clustering_task)
            
            state["messages"].append(AIMessage(content=f"Created {len(clusters)} failure clusters from {len(failure_data)} failures"))
            
        except Exception as e:
            logger.error(f"Error in failure clustering: {e}")
            state["errors"].append(f"Failure clustering error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Failure clustering failed: {str(e)}"))
        
        return state
    
    async def analyze_root_causes(self, state: TestAutomationState) -> TestAutomationState:
        """Analyze root causes of test failures."""
        logger.info("Diagnosis Agent: Analyzing root causes")
        
        try:
            clusters = state["failure_clusters"]
            if not clusters:
                state["messages"].append(AIMessage(content="No failure clusters to analyze"))
                return state
            
            # Prepare cluster data for analysis
            cluster_data = [self._cluster_to_dict(cluster) for cluster in clusters]
            
            # Use LLM for root cause analysis
            root_cause_chain = self.root_cause_prompt | self.llm | JsonOutputParser()
            
            result = await root_cause_chain.ainvoke({
                "failure_clusters": cluster_data,
                "historical_context": state.get("learning_insights", {}),
                "system_context": state.get("config", {})
            })
            
            # Update learning insights with root cause analysis
            state["learning_insights"]["root_cause_analysis"] = result
            
            # Check for escalation needs
            if result.get("escalation_needed", False):
                state["escalation_needed"] = True
                state["escalation_level"] = "high"
                state["messages"].append(AIMessage(content=f"Escalation needed: {result.get('escalation_reason', 'Critical root causes identified')}"))
            
            # Add root cause analysis task to completed
            analysis_task = create_task_from_data("analyze_root_causes", {
                "clusters_analyzed": len(clusters),
                "summary": result.get("summary", {}),
                "escalation_needed": result.get("escalation_needed", False)
            })
            analysis_task.status = TaskStatus.COMPLETED
            analysis_task.completed_at = datetime.now()
            state["completed_tasks"].append(analysis_task)
            
            state["messages"].append(AIMessage(content=f"Root cause analysis completed for {len(clusters)} clusters"))
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            state["errors"].append(f"Root cause analysis error: {str(e)}")
        
        return state
    
    async def triage_bugs(self, state: TestAutomationState) -> TestAutomationState:
        """Triage bugs based on severity and impact."""
        logger.info("Diagnosis Agent: Triaging bugs")
        
        try:
            # Extract bugs from test results and failure clusters
            bugs = self._extract_bugs_from_failures(state)
            
            if not bugs:
                state["messages"].append(AIMessage(content="No bugs to triage"))
                return state
            
            # Use LLM for bug triage
            triage_chain = self.bug_triage_prompt | self.llm | JsonOutputParser()
            
            result = await triage_chain.ainvoke({
                "bugs": bugs,
                "triage_criteria": {
                    "severity_threshold": 0.7,
                    "impact_threshold": 0.8,
                    "frequency_threshold": 5
                },
                "team_context": {
                    "available_team_members": ["developer1", "developer2", "senior_dev"],
                    "current_workload": {"developer1": 20, "developer2": 15, "senior_dev": 10}
                }
            })
            
            # Update learning insights with bug triage results
            state["learning_insights"]["bug_triage"] = result
            
            # Add triage task to completed
            triage_task = create_task_from_data("triage_bugs", {
                "bugs_triaged": len(bugs),
                "triage_summary": result.get("triage_summary", {}),
                "escalation_count": result.get("triage_summary", {}).get("escalation_count", 0)
            })
            triage_task.status = TaskStatus.COMPLETED
            triage_task.completed_at = datetime.now()
            state["completed_tasks"].append(triage_task)
            
            state["messages"].append(AIMessage(content=f"Triaged {len(bugs)} bugs"))
            
        except Exception as e:
            logger.error(f"Error in bug triage: {e}")
            state["errors"].append(f"Bug triage error: {str(e)}")
        
        return state
    
    async def suggest_fixes(self, state: TestAutomationState) -> TestAutomationState:
        """Suggest fixes for identified issues."""
        logger.info("Diagnosis Agent: Suggesting fixes")
        
        try:
            # Extract issues from clusters and triage results
            issues = self._extract_issues_from_analysis(state)
            
            if not issues:
                state["messages"].append(AIMessage(content="No issues to suggest fixes for"))
                return state
            
            # Use LLM for fix suggestions
            fix_chain = self.fix_suggestion_prompt | self.llm | JsonOutputParser()
            
            result = await fix_chain.ainvoke({
                "issues": issues,
                "context": {
                    "code_changes": state.get("code_changes", {}),
                    "system_config": state.get("config", {})
                },
                "available_resources": {
                    "team_size": 3,
                    "available_hours": 40,
                    "priority_level": "high"
                }
            })
            
            # Update learning insights with fix suggestions
            state["learning_insights"]["fix_suggestions"] = result
            
            # Add fix suggestion task to completed
            fix_task = create_task_from_data("suggest_fixes", {
                "issues_analyzed": len(issues),
                "fix_summary": result.get("fix_summary", {}),
                "prioritization": result.get("prioritization", [])
            })
            fix_task.status = TaskStatus.COMPLETED
            fix_task.completed_at = datetime.now()
            state["completed_tasks"].append(fix_task)
            
            state["messages"].append(AIMessage(content=f"Suggested fixes for {len(issues)} issues"))
            
        except Exception as e:
            logger.error(f"Error in fix suggestions: {e}")
            state["errors"].append(f"Fix suggestion error: {str(e)}")
        
        return state
    
    async def analyze_failure_trends(self, state: TestAutomationState) -> TestAutomationState:
        """Analyze failure trends over time."""
        logger.info("Diagnosis Agent: Analyzing failure trends")
        
        try:
            # Analyze trends from test results
            trends = self._analyze_trends_from_results(state["test_results"])
            
            # Identify concerning patterns
            concerning_patterns = self._identify_concerning_patterns(trends)
            
            # Generate trend recommendations
            trend_recommendations = self._generate_trend_recommendations(trends, concerning_patterns)
            
            # Update learning insights
            state["learning_insights"]["failure_trends"] = {
                "trends": trends,
                "concerning_patterns": concerning_patterns,
                "recommendations": trend_recommendations
            }
            
            state["messages"].append(AIMessage(content=f"Analyzed failure trends, identified {len(concerning_patterns)} concerning patterns"))
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            state["errors"].append(f"Trend analysis error: {str(e)}")
        
        return state
    
    def _convert_test_results_to_failure_data(self, test_results: List) -> List[Dict[str, Any]]:
        """Convert test results to failure data format for clustering."""
        failure_data = []
        
        for result in test_results:
            if result.status in ["failed", "error"]:
                failure_data.append({
                    "test_name": result.scenario_id,
                    "error_message": result.error_message or "Unknown error",
                    "stack_trace": "",  # Would be populated from actual data
                    "duration": result.execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "environment": "test",
                    "platform": "unknown"
                })
        
        return failure_data
    
    def _cluster_to_dict(self, cluster: FailureCluster) -> Dict[str, Any]:
        """Convert FailureCluster to dictionary."""
        return {
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "common_patterns": cluster.common_patterns,
            "representative_failure": cluster.representative_failure,
            "confidence_score": cluster.confidence_score,
            "suggested_fixes": cluster.suggested_fixes
        }
    
    def _extract_bugs_from_failures(self, state: TestAutomationState) -> List[Dict[str, Any]]:
        """Extract bugs from test results and failure clusters."""
        bugs = []
        
        # Extract bugs from test results
        for result in state["test_results"]:
            if result.status == "failed":
                bugs.append({
                    "id": f"bug-{result.result_id}",
                    "description": result.error_message or "Test failure",
                    "severity": "medium",
                    "component": "unknown",
                    "frequency": 1,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Extract bugs from failure clusters
        for cluster in state["failure_clusters"]:
            if cluster.confidence_score > 0.7:  # High confidence clusters
                bugs.append({
                    "id": f"cluster-bug-{cluster.cluster_id}",
                    "description": f"Pattern failure: {' '.join(cluster.common_patterns)}",
                    "severity": "high" if cluster.size > 5 else "medium",
                    "component": "multiple",
                    "frequency": cluster.size,
                    "timestamp": datetime.now().isoformat()
                })
        
        return bugs
    
    def _extract_issues_from_analysis(self, state: TestAutomationState) -> List[Dict[str, Any]]:
        """Extract issues from analysis results."""
        issues = []
        
        # Extract from root cause analysis
        root_cause_analysis = state.get("learning_insights", {}).get("root_cause_analysis", {})
        for root_cause in root_cause_analysis.get("root_causes", []):
            issues.append({
                "id": f"root-cause-{root_cause['cluster_id']}",
                "type": "root_cause",
                "description": root_cause["primary_cause"],
                "complexity": "high" if root_cause["confidence_score"] > 0.8 else "medium",
                "impact": "high"
            })
        
        # Extract from bug triage
        bug_triage = state.get("learning_insights", {}).get("bug_triage", {})
        for bug in bug_triage.get("triaged_bugs", []):
            if bug["priority"] in ["critical", "high"]:
                issues.append({
                    "id": bug["bug_id"],
                    "type": "bug",
                    "description": f"High priority bug: {bug['priority']}",
                    "complexity": "medium",
                    "impact": bug["impact_score"]
                })
        
        return issues
    
    def _analyze_trends_from_results(self, test_results: List) -> List[Dict[str, Any]]:
        """Analyze trends from test results."""
        # Simple trend analysis - in practice, this would be more sophisticated
        trends = []
        
        # Group by day (simplified)
        daily_failures = {}
        for result in test_results:
            if result.status == "failed":
                date = datetime.now().date()  # Would use actual date from result
                daily_failures[date] = daily_failures.get(date, 0) + 1
        
        for date, count in daily_failures.items():
            trends.append({
                "date": str(date),
                "failure_count": count,
                "trend": "increasing" if count > 5 else "stable"
            })
        
        return trends
    
    def _identify_concerning_patterns(self, trends: List[Dict[str, Any]]) -> List[str]:
        """Identify concerning patterns in trends."""
        patterns = []
        
        # Check for increasing trends
        high_failure_days = [t for t in trends if t.get("failure_count", 0) > 5]
        if len(high_failure_days) > 2:
            patterns.append("Multiple high-failure days detected")
        
        # Check for trend direction
        increasing_trends = [t for t in trends if t.get("trend") == "increasing"]
        if len(increasing_trends) > 1:
            patterns.append("Increasing failure trend detected")
        
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
    
    async def should_continue(self, state: TestAutomationState) -> str:
        """Determine if the agent should continue or move to next step."""
        # Check if there are pending diagnosis tasks
        pending_tasks = [task for task in state["active_tasks"] if task.task_type.startswith(("cluster_", "analyze_", "triage_", "suggest_"))]
        
        if pending_tasks:
            return "continue"
        elif state["failure_clusters"] or state["learning_insights"].get("root_cause_analysis"):
            return "next"
        else:
            return "error"
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": "active",
            "capabilities": {
                "cluster_test_failures": True,
                "analyze_root_causes": True,
                "triage_bugs": True,
                "suggest_fixes": True,
                "analyze_failure_trends": True
            },
            "metrics": {
                "tasks_completed": 0,
                "success_rate": 1.0,
                "last_activity": datetime.now().isoformat()
            }
        }
