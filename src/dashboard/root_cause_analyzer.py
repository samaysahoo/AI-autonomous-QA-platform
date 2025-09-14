"""LLM-based root cause analysis for test failures."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI

from config.settings import get_settings
from .failure_clusterer import FailureCluster

logger = logging.getLogger(__name__)


@dataclass
class RootCauseAnalysis:
    """Result of root cause analysis."""
    cluster_id: int
    primary_cause: str
    contributing_factors: List[str]
    confidence_score: float
    recommendations: List[str]
    similar_incidents: List[str]
    prevention_strategies: List[str]


@dataclass
class RootCauseSummary:
    """Summary of root cause analysis across multiple clusters."""
    total_clusters: int
    analyzed_clusters: int
    common_causes: List[str]
    top_recommendations: List[str]
    overall_confidence: float


class RootCauseAnalyzer:
    """Analyzes root causes of test failures using LLM."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
    
    def analyze_root_causes(self, clusters: List[FailureCluster]) -> List[RootCauseAnalysis]:
        """Analyze root causes for multiple failure clusters."""
        
        analyses = []
        
        for cluster in clusters:
            try:
                analysis = self._analyze_single_cluster(cluster)
                analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing cluster {cluster.cluster_id}: {e}")
                # Create fallback analysis
                fallback_analysis = RootCauseAnalysis(
                    cluster_id=cluster.cluster_id,
                    primary_cause="Unable to analyze - error occurred",
                    contributing_factors=["Analysis failed"],
                    confidence_score=0.0,
                    recommendations=["Manual investigation required"],
                    similar_incidents=[],
                    prevention_strategies=[]
                )
                analyses.append(fallback_analysis)
        
        logger.info(f"Completed root cause analysis for {len(analyses)} clusters")
        return analyses
    
    def _analyze_single_cluster(self, cluster: FailureCluster) -> RootCauseAnalysis:
        """Analyze root cause for a single failure cluster."""
        
        # Prepare failure data for analysis
        failure_data = self._prepare_failure_data(cluster)
        
        # Generate LLM prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(failure_data)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse LLM response
            analysis_result = self._parse_llm_response(response.choices[0].message.content)
            
            # Create RootCauseAnalysis object
            analysis = RootCauseAnalysis(
                cluster_id=cluster.cluster_id,
                primary_cause=analysis_result.get("primary_cause", "Unknown"),
                contributing_factors=analysis_result.get("contributing_factors", []),
                confidence_score=analysis_result.get("confidence_score", 0.5),
                recommendations=analysis_result.get("recommendations", []),
                similar_incidents=analysis_result.get("similar_incidents", []),
                prevention_strategies=analysis_result.get("prevention_strategies", [])
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return fallback analysis
            return RootCauseAnalysis(
                cluster_id=cluster.cluster_id,
                primary_cause="Analysis failed due to API error",
                contributing_factors=["LLM API error"],
                confidence_score=0.0,
                recommendations=["Manual investigation required"],
                similar_incidents=[],
                prevention_strategies=[]
            )
    
    def _prepare_failure_data(self, cluster: FailureCluster) -> Dict[str, Any]:
        """Prepare failure data for LLM analysis."""
        
        # Extract common patterns
        error_messages = [f.get('error_message', '') for f in cluster.failures]
        stack_traces = [f.get('stack_trace', '') for f in cluster.failures]
        test_names = [f.get('test_name', '') for f in cluster.failures]
        
        # Get representative failure
        representative = cluster.representative_failure
        
        return {
            "cluster_size": cluster.size,
            "confidence_score": cluster.confidence_score,
            "common_patterns": cluster.common_patterns,
            "representative_failure": {
                "test_name": representative.get('test_name', ''),
                "error_message": representative.get('error_message', ''),
                "stack_trace": representative.get('stack_trace', ''),
                "duration": representative.get('duration', 0),
                "timestamp": representative.get('timestamp', ''),
                "environment": representative.get('environment', ''),
                "browser": representative.get('browser', ''),
                "os": representative.get('os', '')
            },
            "error_summary": {
                "unique_errors": len(set(error_messages)),
                "common_errors": self._get_common_errors(error_messages),
                "test_types": list(set(test_names)),
                "failure_timeline": self._analyze_timeline(cluster.failures)
            }
        }
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM analysis."""
        
        return """You are an expert QA engineer and software debugging specialist. Your task is to analyze test failure clusters and identify root causes.

Analyze the provided test failure data and provide:
1. Primary root cause
2. Contributing factors
3. Confidence score (0.0 to 1.0)
4. Specific recommendations
5. Similar incidents (if any)
6. Prevention strategies

Focus on:
- Technical root causes (code issues, infrastructure, dependencies)
- Environmental factors (browser, OS, network)
- Test design issues
- Data or configuration problems
- Timing or concurrency issues

Be specific and actionable in your recommendations. Consider both immediate fixes and long-term prevention strategies."""
    
    def _create_user_prompt(self, failure_data: Dict[str, Any]) -> str:
        """Create user prompt with failure data."""
        
        prompt = f"""
Please analyze the following test failure cluster:

CLUSTER INFORMATION:
- Cluster Size: {failure_data['cluster_size']} failures
- Confidence Score: {failure_data['confidence_score']:.2f}
- Common Patterns: {', '.join(failure_data['common_patterns'][:5])}

REPRESENTATIVE FAILURE:
- Test Name: {failure_data['representative_failure']['test_name']}
- Error Message: {failure_data['representative_failure']['error_message']}
- Stack Trace: {failure_data['representative_failure']['stack_trace'][:500]}...
- Duration: {failure_data['representative_failure']['duration']} seconds
- Environment: {failure_data['representative_failure']['environment']}
- Browser: {failure_data['representative_failure']['browser']}
- OS: {failure_data['representative_failure']['os']}

ERROR SUMMARY:
- Unique Errors: {failure_data['error_summary']['unique_errors']}
- Common Errors: {', '.join(failure_data['error_summary']['common_errors'][:3])}
- Test Types: {', '.join(failure_data['error_summary']['test_types'][:5])}

Please provide your analysis in the following JSON format:
{{
    "primary_cause": "Main root cause description",
    "contributing_factors": ["Factor 1", "Factor 2", "Factor 3"],
    "confidence_score": 0.85,
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "similar_incidents": ["Similar incident 1", "Similar incident 2"],
    "prevention_strategies": ["Strategy 1", "Strategy 2"]
}}
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails."""
        
        return {
            "primary_cause": "Unable to parse analysis",
            "contributing_factors": ["Parsing error"],
            "confidence_score": 0.0,
            "recommendations": ["Manual investigation required"],
            "similar_incidents": [],
            "prevention_strategies": []
        }
    
    def _get_common_errors(self, error_messages: List[str]) -> List[str]:
        """Extract common error patterns."""
        
        # Simple frequency analysis
        from collections import Counter
        
        # Clean and tokenize error messages
        words = []
        for message in error_messages:
            words.extend(message.lower().split())
        
        # Remove common words and get most frequent
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        counter = Counter(words)
        return [word for word, count in counter.most_common(10)]
    
    def _analyze_timeline(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure timeline."""
        
        try:
            timestamps = []
            for failure in failures:
                if 'timestamp' in failure:
                    timestamps.append(failure['timestamp'])
            
            if not timestamps:
                return {"pattern": "unknown", "trend": "unknown"}
            
            # Simple timeline analysis
            if len(timestamps) > 1:
                return {"pattern": "multiple", "trend": "increasing"}
            else:
                return {"pattern": "single", "trend": "stable"}
                
        except Exception as e:
            logger.error(f"Error analyzing timeline: {e}")
            return {"pattern": "unknown", "trend": "unknown"}
    
    def generate_summary(self, analyses: List[RootCauseAnalysis]) -> RootCauseSummary:
        """Generate summary of root cause analyses."""
        
        try:
            if not analyses:
                return RootCauseSummary(
                    total_clusters=0,
                    analyzed_clusters=0,
                    common_causes=[],
                    top_recommendations=[],
                    overall_confidence=0.0
                )
            
            # Extract common causes
            all_causes = [analysis.primary_cause for analysis in analyses]
            cause_counter = {}
            for cause in all_causes:
                cause_counter[cause] = cause_counter.get(cause, 0) + 1
            
            common_causes = [cause for cause, count in sorted(cause_counter.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            # Extract top recommendations
            all_recommendations = []
            for analysis in analyses:
                all_recommendations.extend(analysis.recommendations)
            
            from collections import Counter
            rec_counter = Counter(all_recommendations)
            top_recommendations = [rec for rec, count in rec_counter.most_common(10)]
            
            # Calculate overall confidence
            confidences = [analysis.confidence_score for analysis in analyses]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return RootCauseSummary(
                total_clusters=len(analyses),
                analyzed_clusters=len([a for a in analyses if a.confidence_score > 0.5]),
                common_causes=common_causes,
                top_recommendations=top_recommendations,
                overall_confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return RootCauseSummary(
                total_clusters=0,
                analyzed_clusters=0,
                common_causes=[],
                top_recommendations=[],
                overall_confidence=0.0
            )
    
    def export_analysis_report(self, analyses: List[RootCauseAnalysis], 
                             summary: RootCauseSummary, 
                             output_path: str) -> bool:
        """Export analysis report to file."""
        
        try:
            report = {
                "summary": {
                    "total_clusters": summary.total_clusters,
                    "analyzed_clusters": summary.analyzed_clusters,
                    "common_causes": summary.common_causes,
                    "top_recommendations": summary.top_recommendations,
                    "overall_confidence": summary.overall_confidence
                },
                "detailed_analyses": [
                    {
                        "cluster_id": analysis.cluster_id,
                        "primary_cause": analysis.primary_cause,
                        "contributing_factors": analysis.contributing_factors,
                        "confidence_score": analysis.confidence_score,
                        "recommendations": analysis.recommendations,
                        "similar_incidents": analysis.similar_incidents,
                        "prevention_strategies": analysis.prevention_strategies
                    }
                    for analysis in analyses
                ]
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Analysis report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            return False
