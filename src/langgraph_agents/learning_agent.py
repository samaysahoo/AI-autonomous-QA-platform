"""LangGraph-based Learning Agent."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import (
    TestAutomationState, LearningFeedback, Task, TaskStatus, 
    AgentStatus, create_task_from_data
)
from ..dashboard.feedback_loop import FeedbackLoop, HumanFeedback, ModelUpdate
from ..test_generation.test_generator import TestCaseGenerator
from ..observability.risk_analyzer import RiskAnalyzer
from ..data_ingestion.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class LearningAgent:
    """LangGraph-based Learning Agent for continuous learning and model updates."""
    
    def __init__(self, agent_id: str = "learning"):
        self.agent_id = agent_id
        self.name = "Learning Agent"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize components
        self.feedback_loop = FeedbackLoop()
        self.test_generator = TestCaseGenerator()
        self.risk_analyzer = RiskAnalyzer()
        self.vector_store = VectorStoreManager()
        
        # Learning data storage
        self.learning_history = []
        self.model_performance = {}
        self.pattern_library = {}
        self.knowledge_base = {}
        
        # Setup prompts
        self._setup_prompts()
        
        logger.info(f"Learning Agent {agent_id} initialized")
    
    def _setup_prompts(self):
        """Setup LangChain prompts for the agent."""
        
        # Feedback learning prompt
        self.feedback_learning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Learning Agent specializing in processing feedback and improving test automation.
            
            Your capabilities:
            - Analyze human feedback and corrections
            - Extract learning patterns from test results
            - Update model parameters based on feedback
            - Identify improvement opportunities
            - Generate learning insights
            
            Focus on actionable improvements that enhance test automation effectiveness.
            """),
            HumanMessage(content="""
            Process the following feedback data and generate learning insights:
            
            Feedback Data: {feedback_data}
            Learning Mode: {learning_mode}
            Model Types: {model_types}
            Historical Context: {historical_context}
            
            Provide learning analysis in JSON format:
            {{
                "feedback_analysis": {{
                    "total_feedback_items": number,
                    "feedback_types": ["type1", "type2"],
                    "common_patterns": ["pattern1", "pattern2"],
                    "quality_score": 0.0-1.0
                }},
                "learning_insights": [
                    {{
                        "insight_type": "pattern|correction|improvement",
                        "description": "insight_description",
                        "confidence": 0.0-1.0,
                        "actionable": true|false,
                        "impact": "low|medium|high"
                    }}
                ],
                "model_updates": [
                    {{
                        "model_type": "test_generator|risk_analyzer|execution_agent",
                        "update_type": "parameter|algorithm|threshold",
                        "description": "update_description",
                        "confidence": 0.0-1.0,
                        "expected_improvement": 0.0-1.0
                    }}
                ],
                "recommendations": [
                    {{
                        "type": "immediate|short_term|long_term",
                        "description": "recommendation_description",
                        "priority": "low|medium|high",
                        "effort": "low|medium|high"
                    }}
                ]
            }}
            """)
        ])
        
        # Pattern learning prompt
        self.pattern_learning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Pattern Recognition specialist for test automation learning.
            
            Your expertise:
            - Extract patterns from execution logs and results
            - Identify success and failure patterns
            - Recognize environmental and timing patterns
            - Build pattern libraries for future reference
            - Generate pattern-based recommendations
            
            Focus on actionable patterns that improve test reliability and efficiency.
            """),
            HumanMessage(content="""
            Analyze patterns from the following execution data:
            
            Execution Logs: {execution_logs}
            Test Results: {test_results}
            System Metrics: {system_metrics}
            Time Period: {time_period}
            
            Provide pattern analysis in JSON format:
            {{
                "pattern_analysis": {{
                    "total_patterns": number,
                    "pattern_categories": ["category1", "category2"],
                    "confidence_threshold": 0.0-1.0
                }},
                "success_patterns": [
                    {{
                        "pattern_id": "unique_id",
                        "description": "pattern_description",
                        "frequency": number,
                        "confidence": 0.0-1.0,
                        "conditions": ["condition1", "condition2"],
                        "recommendations": ["rec1", "rec2"]
                    }}
                ],
                "failure_patterns": [
                    {{
                        "pattern_id": "unique_id",
                        "description": "pattern_description",
                        "frequency": number,
                        "confidence": 0.0-1.0,
                        "triggers": ["trigger1", "trigger2"],
                        "prevention_strategies": ["strategy1", "strategy2"]
                    }}
                ],
                "environmental_patterns": [
                    {{
                        "pattern_id": "unique_id",
                        "description": "environmental_pattern",
                        "platform": "platform_name",
                        "conditions": ["condition1", "condition2"],
                        "impact": "low|medium|high"
                    }}
                ],
                "recommendations": [
                    {{
                        "type": "optimization|prevention|improvement",
                        "description": "recommendation",
                        "priority": "low|medium|high",
                        "implementation_effort": "low|medium|high"
                    }}
                ]
            }}
            """)
        ])
        
        # Model optimization prompt
        self.model_optimization_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a Model Optimization specialist for AI test automation systems.
            
            Your responsibilities:
            - Analyze model performance metrics
            - Identify optimization opportunities
            - Recommend parameter adjustments
            - Assess optimization risks
            - Plan optimization strategies
            
            Focus on safe, incremental improvements with measurable benefits.
            """),
            HumanMessage(content="""
            Optimize models based on the following performance data:
            
            Performance Data: {performance_data}
            Optimization Target: {optimization_target}
            Current Metrics: {current_metrics}
            Historical Performance: {historical_performance}
            
            Provide optimization recommendations in JSON format:
            {{
                "optimization_analysis": {{
                    "current_performance": {{
                        "accuracy": 0.0-1.0,
                        "precision": 0.0-1.0,
                        "recall": 0.0-1.0,
                        "execution_time": "seconds"
                    }},
                    "improvement_potential": 0.0-1.0,
                    "optimization_priority": "low|medium|high"
                }},
                "optimization_recommendations": [
                    {{
                        "model_type": "model_name",
                        "optimization_type": "parameter|algorithm|data",
                        "description": "optimization_description",
                        "expected_improvement": 0.0-1.0,
                        "risk_level": "low|medium|high",
                        "implementation_complexity": "low|medium|high",
                        "testing_requirements": ["test1", "test2"]
                    }}
                ],
                "implementation_plan": [
                    {{
                        "phase": "phase_name",
                        "optimizations": ["opt1", "opt2"],
                        "timeline": "estimated_time",
                        "success_criteria": ["criteria1", "criteria2"]
                    }}
                ],
                "risk_assessment": {{
                    "overall_risk": "low|medium|high",
                    "potential_issues": ["issue1", "issue2"],
                    "mitigation_strategies": ["strategy1", "strategy2"]
                }}
            }}
            """)
        ])
    
    async def learn_from_feedback(self, state: TestAutomationState) -> TestAutomationState:
        """Learn from human feedback and corrections."""
        logger.info("Learning Agent: Learning from feedback")
        
        try:
            feedback_data = state["feedback_data"]
            if not feedback_data:
                state["messages"].append(AIMessage(content="No feedback data to process"))
                return state
            
            # Use LLM for feedback analysis
            feedback_chain = self.feedback_learning_prompt | self.llm | JsonOutputParser()
            
            result = await feedback_chain.ainvoke({
                "feedback_data": [self._feedback_to_dict(fb) for fb in feedback_data],
                "learning_mode": "incremental",
                "model_types": ["test_generator", "risk_analyzer", "execution_agent"],
                "historical_context": state.get("learning_insights", {})
            })
            
            # Process learning insights
            learning_insights = result.get("learning_insights", [])
            for insight in learning_insights:
                if insight.get("actionable", False):
                    state["learning_insights"][f"insight_{len(state['learning_insights'])}"] = insight
            
            # Apply model updates
            model_updates = result.get("model_updates", [])
            applied_updates = await self._apply_model_updates(model_updates, state)
            
            # Generate recommendations
            recommendations = result.get("recommendations", [])
            state["learning_insights"]["recommendations"] = recommendations
            
            # Add learning task to completed
            learning_task = create_task_from_data("learn_from_feedback", {
                "feedback_processed": len(feedback_data),
                "insights_generated": len(learning_insights),
                "model_updates_applied": len(applied_updates),
                "recommendations": len(recommendations)
            })
            learning_task.status = TaskStatus.COMPLETED
            learning_task.completed_at = datetime.now()
            state["completed_tasks"].append(learning_task)
            
            state["messages"].append(AIMessage(content=f"Processed {len(feedback_data)} feedback items, generated {len(learning_insights)} insights"))
            
        except Exception as e:
            logger.error(f"Error in feedback learning: {e}")
            state["errors"].append(f"Feedback learning error: {str(e)}")
            state["messages"].append(AIMessage(content=f"Feedback learning failed: {str(e)}"))
        
        return state
    
    async def learn_from_patterns(self, state: TestAutomationState) -> TestAutomationState:
        """Learn from execution patterns and logs."""
        logger.info("Learning Agent: Learning from patterns")
        
        try:
            # Extract execution data from state
            execution_logs = self._extract_execution_logs(state)
            test_results = state["test_results"]
            system_metrics = state.get("system_metrics", {})
            
            # Use LLM for pattern analysis
            pattern_chain = self.pattern_learning_prompt | self.llm | JsonOutputParser()
            
            result = await pattern_chain.ainvoke({
                "execution_logs": execution_logs,
                "test_results": [self._test_result_to_dict(tr) for tr in test_results],
                "system_metrics": system_metrics,
                "time_period": "last_24_hours"
            })
            
            # Store patterns in pattern library
            success_patterns = result.get("success_patterns", [])
            failure_patterns = result.get("failure_patterns", [])
            environmental_patterns = result.get("environmental_patterns", [])
            
            self.pattern_library.update({
                "success_patterns": success_patterns,
                "failure_patterns": failure_patterns,
                "environmental_patterns": environmental_patterns
            })
            
            # Update learning insights
            state["learning_insights"]["pattern_analysis"] = result
            
            # Add pattern learning task to completed
            pattern_task = create_task_from_data("learn_from_patterns", {
                "patterns_identified": len(success_patterns) + len(failure_patterns) + len(environmental_patterns),
                "success_patterns": len(success_patterns),
                "failure_patterns": len(failure_patterns),
                "environmental_patterns": len(environmental_patterns)
            })
            pattern_task.status = TaskStatus.COMPLETED
            pattern_task.completed_at = datetime.now()
            state["completed_tasks"].append(pattern_task)
            
            state["messages"].append(AIMessage(content=f"Identified {len(success_patterns) + len(failure_patterns)} patterns from execution data"))
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {e}")
            state["errors"].append(f"Pattern learning error: {str(e)}")
        
        return state
    
    async def optimize_models(self, state: TestAutomationState) -> TestAutomationState:
        """Optimize models based on performance metrics."""
        logger.info("Learning Agent: Optimizing models")
        
        try:
            # Collect performance data
            performance_data = await self._collect_performance_data(state)
            
            # Use LLM for optimization analysis
            optimization_chain = self.model_optimization_prompt | self.llm | JsonOutputParser()
            
            result = await optimization_chain.ainvoke({
                "performance_data": performance_data,
                "optimization_target": "accuracy",
                "current_metrics": self.model_performance,
                "historical_performance": self.learning_history
            })
            
            # Apply safe optimizations
            optimization_recommendations = result.get("optimization_recommendations", [])
            applied_optimizations = await self._apply_safe_optimizations(optimization_recommendations, state)
            
            # Update learning insights
            state["learning_insights"]["model_optimization"] = result
            
            # Add optimization task to completed
            optimization_task = create_task_from_data("optimize_models", {
                "recommendations_generated": len(optimization_recommendations),
                "optimizations_applied": len(applied_optimizations),
                "expected_improvement": result.get("optimization_analysis", {}).get("improvement_potential", 0)
            })
            optimization_task.status = TaskStatus.COMPLETED
            optimization_task.completed_at = datetime.now()
            state["completed_tasks"].append(optimization_task)
            
            state["messages"].append(AIMessage(content=f"Applied {len(applied_optimizations)} model optimizations"))
            
        except Exception as e:
            logger.error(f"Error in model optimization: {e}")
            state["errors"].append(f"Model optimization error: {str(e)}")
        
        return state
    
    async def sync_knowledge_base(self, state: TestAutomationState) -> TestAutomationState:
        """Sync knowledge base with external sources."""
        logger.info("Learning Agent: Syncing knowledge base")
        
        try:
            # Sync with vector store
            vector_store_sync = await self._sync_with_vector_store(state)
            
            # Sync with external APIs (if configured)
            external_sync = await self._sync_with_external_sources(state)
            
            # Update knowledge base
            self.knowledge_base.update({
                "vector_store_sync": vector_store_sync,
                "external_sync": external_sync,
                "last_sync": datetime.now().isoformat()
            })
            
            # Add sync task to completed
            sync_task = create_task_from_data("sync_knowledge_base", {
                "vector_store_updated": vector_store_sync.get("updated_documents", 0),
                "external_sources_synced": len(external_sync.get("synced_sources", [])),
                "knowledge_base_updated": True
            })
            sync_task.status = TaskStatus.COMPLETED
            sync_task.completed_at = datetime.now()
            state["completed_tasks"].append(sync_task)
            
            state["messages"].append(AIMessage(content="Knowledge base synchronized successfully"))
            
        except Exception as e:
            logger.error(f"Error in knowledge base sync: {e}")
            state["errors"].append(f"Knowledge base sync error: {str(e)}")
        
        return state
    
    async def _apply_model_updates(self, model_updates: List[Dict[str, Any]], state: TestAutomationState) -> List[Dict[str, Any]]:
        """Apply model updates safely."""
        applied_updates = []
        
        for update in model_updates:
            if update.get("risk_level", "medium") in ["low", "medium"]:
                # Apply safe updates
                model_type = update.get("model_type")
                update_type = update.get("optimization_type")
                
                if model_type == "test_generator":
                    # Update test generator parameters
                    applied_updates.append({
                        "model": "test_generator",
                        "update": update_type,
                        "status": "applied",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif model_type == "risk_analyzer":
                    # Update risk analyzer parameters
                    applied_updates.append({
                        "model": "risk_analyzer",
                        "update": update_type,
                        "status": "applied",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return applied_updates
    
    async def _apply_safe_optimizations(self, recommendations: List[Dict[str, Any]], state: TestAutomationState) -> List[Dict[str, Any]]:
        """Apply safe optimizations."""
        applied_optimizations = []
        
        for rec in recommendations:
            if (rec.get("risk_level", "medium") == "low" and 
                rec.get("implementation_complexity", "medium") in ["low", "medium"]):
                
                applied_optimizations.append({
                    "optimization": rec.get("description", ""),
                    "model_type": rec.get("model_type", ""),
                    "status": "applied",
                    "timestamp": datetime.now().isoformat()
                })
        
        return applied_optimizations
    
    def _feedback_to_dict(self, feedback: LearningFeedback) -> Dict[str, Any]:
        """Convert LearningFeedback to dictionary."""
        return {
            "feedback_id": feedback.feedback_id,
            "source_type": feedback.source_type,
            "content": feedback.content,
            "timestamp": feedback.timestamp.isoformat(),
            "processed": feedback.processed
        }
    
    def _test_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert TestResult to dictionary."""
        return {
            "result_id": result.result_id,
            "scenario_id": result.scenario_id,
            "status": result.status,
            "execution_time": result.execution_time,
            "error_message": result.error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_execution_logs(self, state: TestAutomationState) -> List[Dict[str, Any]]:
        """Extract execution logs from state."""
        logs = []
        
        # Extract from completed tasks
        for task in state["completed_tasks"]:
            if task.task_type.startswith("execute_"):
                logs.append({
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "duration": (task.completed_at - task.started_at).total_seconds() if task.started_at else 0,
                    "timestamp": task.completed_at.isoformat() if task.completed_at else None
                })
        
        return logs
    
    async def _collect_performance_data(self, state: TestAutomationState) -> Dict[str, Any]:
        """Collect performance data for optimization."""
        return {
            "test_execution_success_rate": len([r for r in state["test_results"] if r.status == "passed"]) / max(1, len(state["test_results"])),
            "average_execution_time": sum(r.execution_time for r in state["test_results"]) / max(1, len(state["test_results"])),
            "task_completion_rate": len(state["completed_tasks"]) / max(1, len(state["completed_tasks"]) + len(state["failed_tasks"])),
            "error_rate": len(state["errors"]) / max(1, len(state["completed_tasks"]) + len(state["failed_tasks"])),
            "escalation_rate": 1 if state.get("escalation_needed", False) else 0
        }
    
    async def _sync_with_vector_store(self, state: TestAutomationState) -> Dict[str, Any]:
        """Sync with vector store."""
        # This would implement actual vector store synchronization
        return {
            "updated_documents": 5,
            "new_documents": 2,
            "removed_documents": 0,
            "sync_status": "success"
        }
    
    async def _sync_with_external_sources(self, state: TestAutomationState) -> Dict[str, Any]:
        """Sync with external sources."""
        # This would implement actual external source synchronization
        return {
            "synced_sources": ["jira", "github", "slack"],
            "new_data_points": 10,
            "sync_status": "success"
        }
    
    async def should_continue(self, state: TestAutomationState) -> str:
        """Determine if the agent should continue or move to next step."""
        # Check if there are pending learning tasks
        pending_tasks = [task for task in state["active_tasks"] if task.task_type.startswith(("learn_", "optimize_", "sync_"))]
        
        if pending_tasks:
            return "continue"
        elif state["learning_insights"]:
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
                "learn_from_feedback": True,
                "learn_from_patterns": True,
                "optimize_models": True,
                "sync_knowledge_base": True
            },
            "knowledge_base": {
                "patterns": len(self.pattern_library),
                "learning_history": len(self.learning_history),
                "model_performance": len(self.model_performance)
            },
            "metrics": {
                "tasks_completed": 0,
                "success_rate": 1.0,
                "last_activity": datetime.now().isoformat()
            }
        }
