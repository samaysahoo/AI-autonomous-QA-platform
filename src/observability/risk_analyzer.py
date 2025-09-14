"""Risk analysis for code changes and crash patterns."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from config.settings import get_settings
from src.data_ingestion.analytics_ingestor import CrashEvent

logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    """Represents a risk score for a code change or area."""
    component: str
    risk_level: float  # 0.0 to 1.0
    factors: Dict[str, float]
    confidence: float
    recommendations: List[str]
    affected_areas: List[str]


@dataclass
class CrashPattern:
    """Represents a pattern in crash data."""
    pattern_id: str
    description: str
    frequency: int
    severity: float
    affected_components: List[str]
    common_stack_traces: List[str]
    user_impact: float


class RiskAnalyzer:
    """Analyzes risk levels for code changes based on crash and usage data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.crash_patterns = []
        self.component_risk_history = {}
    
    def analyze_code_change_risk(self, 
                                diff_content: str,
                                changed_files: List[str],
                                commit_metadata: Dict[str, Any]) -> RiskScore:
        """Analyze risk level of a code change based on historical data."""
        
        try:
            # Extract components from changed files
            components = self._extract_components(changed_files)
            
            # Calculate risk factors
            factors = {}
            
            # Factor 1: Crash frequency in affected components
            crash_risk = self._calculate_crash_risk(components)
            factors['crash_frequency'] = crash_risk
            
            # Factor 2: Code complexity and change size
            complexity_risk = self._calculate_complexity_risk(diff_content)
            factors['complexity'] = complexity_risk
            
            # Factor 3: Historical failure rate for similar changes
            historical_risk = self._calculate_historical_risk(diff_content, components)
            factors['historical_failure'] = historical_risk
            
            # Factor 4: User impact assessment
            user_impact_risk = self._calculate_user_impact_risk(components)
            factors['user_impact'] = user_impact_risk
            
            # Factor 5: Test coverage for changed areas
            test_coverage_risk = self._calculate_test_coverage_risk(changed_files)
            factors['test_coverage'] = test_coverage_risk
            
            # Calculate overall risk score
            risk_level = self._calculate_overall_risk(factors)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(factors, components)
            
            # Calculate confidence based on available data
            confidence = self._calculate_confidence(components)
            
            return RiskScore(
                component=components[0] if components else "unknown",
                risk_level=risk_level,
                factors=factors,
                confidence=confidence,
                recommendations=recommendations,
                affected_areas=components
            )
            
        except Exception as e:
            logger.error(f"Error analyzing code change risk: {e}")
            return RiskScore(
                component="unknown",
                risk_level=0.5,
                factors={},
                confidence=0.0,
                recommendations=["Unable to analyze risk due to error"],
                affected_areas=[]
            )
    
    def analyze_crash_patterns(self, crash_events: List[CrashEvent]) -> List[CrashPattern]:
        """Analyze crash events to identify patterns."""
        
        try:
            # Group crashes by similarity
            crash_groups = self._group_similar_crashes(crash_events)
            
            patterns = []
            for group in crash_groups:
                pattern = self._create_crash_pattern(group)
                patterns.append(pattern)
            
            # Sort by severity and frequency
            patterns.sort(key=lambda p: p.severity * p.frequency, reverse=True)
            
            self.crash_patterns = patterns
            logger.info(f"Identified {len(patterns)} crash patterns")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing crash patterns: {e}")
            return []
    
    def _extract_components(self, changed_files: List[str]) -> List[str]:
        """Extract component names from changed file paths."""
        
        components = []
        
        for file_path in changed_files:
            # Extract component based on directory structure
            parts = file_path.split('/')
            
            if len(parts) >= 2:
                if parts[0] in ['src', 'lib', 'app']:
                    component = parts[1]
                else:
                    component = parts[0]
                
                if component not in components:
                    components.append(component)
        
        return components if components else ['unknown']
    
    def _calculate_crash_risk(self, components: List[str]) -> float:
        """Calculate crash risk based on historical crash data."""
        
        try:
            # This would integrate with actual crash data
            # For now, use simulated risk scores
            
            component_crash_rates = {
                'authentication': 0.8,
                'payment': 0.9,
                'database': 0.7,
                'ui': 0.6,
                'api': 0.5,
                'unknown': 0.3
            }
            
            max_risk = 0.0
            for component in components:
                risk = component_crash_rates.get(component, 0.3)
                max_risk = max(max_risk, risk)
            
            return max_risk
            
        except Exception as e:
            logger.error(f"Error calculating crash risk: {e}")
            return 0.5
    
    def _calculate_complexity_risk(self, diff_content: str) -> float:
        """Calculate risk based on code complexity and change size."""
        
        try:
            # Analyze diff content for complexity indicators
            lines = diff_content.split('\n')
            
            # Count additions and deletions
            additions = sum(1 for line in lines if line.startswith('+'))
            deletions = sum(1 for line in lines if line.startswith('-'))
            
            # Calculate change size
            change_size = additions + deletions
            
            # Look for complexity indicators
            complexity_indicators = 0
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'if', 'for', 'while', 'try', 'catch', 'switch', 'case'
                ]):
                    complexity_indicators += 1
            
            # Calculate complexity risk (0.0 to 1.0)
            size_risk = min(change_size / 100.0, 1.0)  # Normalize to 100 lines
            complexity_risk = min(complexity_indicators / 20.0, 1.0)  # Normalize to 20 indicators
            
            # Combine size and complexity
            overall_risk = (size_risk * 0.6 + complexity_risk * 0.4)
            
            return min(overall_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating complexity risk: {e}")
            return 0.5
    
    def _calculate_historical_risk(self, diff_content: str, components: List[str]) -> float:
        """Calculate risk based on historical failure rates for similar changes."""
        
        try:
            # This would integrate with historical test failure data
            # For now, use simulated risk scores
            
            historical_risk_by_component = {
                'authentication': 0.7,
                'payment': 0.8,
                'database': 0.6,
                'ui': 0.4,
                'api': 0.5,
                'unknown': 0.3
            }
            
            max_risk = 0.0
            for component in components:
                risk = historical_risk_by_component.get(component, 0.3)
                max_risk = max(max_risk, risk)
            
            return max_risk
            
        except Exception as e:
            logger.error(f"Error calculating historical risk: {e}")
            return 0.5
    
    def _calculate_user_impact_risk(self, components: List[str]) -> float:
        """Calculate risk based on potential user impact."""
        
        try:
            # Define user impact scores for different components
            user_impact_scores = {
                'authentication': 0.9,  # High impact - users can't log in
                'payment': 0.95,        # Very high impact - financial transactions
                'database': 0.8,        # High impact - data integrity
                'ui': 0.6,              # Medium impact - user experience
                'api': 0.7,             # Medium-high impact - integration issues
                'unknown': 0.4          # Low-medium impact
            }
            
            max_impact = 0.0
            for component in components:
                impact = user_impact_scores.get(component, 0.4)
                max_impact = max(max_impact, impact)
            
            return max_impact
            
        except Exception as e:
            logger.error(f"Error calculating user impact risk: {e}")
            return 0.5
    
    def _calculate_test_coverage_risk(self, changed_files: List[str]) -> float:
        """Calculate risk based on test coverage for changed areas."""
        
        try:
            # This would integrate with test coverage data
            # For now, simulate based on file types and patterns
            
            uncovered_files = 0
            total_files = len(changed_files)
            
            for file_path in changed_files:
                # Check if file has corresponding test file
                if not self._has_test_file(file_path):
                    uncovered_files += 1
            
            if total_files == 0:
                return 1.0  # Maximum risk if no files
            
            coverage_ratio = 1.0 - (uncovered_files / total_files)
            return 1.0 - coverage_ratio  # Higher risk for lower coverage
            
        except Exception as e:
            logger.error(f"Error calculating test coverage risk: {e}")
            return 0.5
    
    def _has_test_file(self, file_path: str) -> bool:
        """Check if a file has a corresponding test file."""
        
        # Simple heuristic - look for test files in common patterns
        test_patterns = [
            file_path.replace('.py', '_test.py'),
            file_path.replace('.py', 'Test.py'),
            file_path.replace('.js', '.test.js'),
            file_path.replace('.js', '.spec.js'),
            file_path.replace('.java', 'Test.java')
        ]
        
        # In a real implementation, you'd check if these files exist
        # For now, simulate based on component
        if 'test' in file_path.lower() or 'spec' in file_path.lower():
            return True
        
        # Simulate coverage based on component type
        if any(comp in file_path for comp in ['auth', 'payment', 'database']):
            return True  # Critical components typically have tests
        
        return False
    
    def _calculate_overall_risk(self, factors: Dict[str, float]) -> float:
        """Calculate overall risk score from individual factors."""
        
        # Weighted average of risk factors
        weights = {
            'crash_frequency': 0.25,
            'complexity': 0.20,
            'historical_failure': 0.20,
            'user_impact': 0.25,
            'test_coverage': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in factors:
                weighted_sum += factors[factor] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Default risk if no factors available
        
        return min(weighted_sum / total_weight, 1.0)
    
    def _generate_recommendations(self, factors: Dict[str, float], 
                                components: List[str]) -> List[str]:
        """Generate recommendations based on risk factors."""
        
        recommendations = []
        
        # High crash frequency recommendation
        if factors.get('crash_frequency', 0) > 0.7:
            recommendations.append(
                f"High crash frequency detected in {', '.join(components)}. "
                "Consider running comprehensive regression tests."
            )
        
        # High complexity recommendation
        if factors.get('complexity', 0) > 0.7:
            recommendations.append(
                "Complex code changes detected. "
                "Consider breaking down into smaller, more testable changes."
            )
        
        # Low test coverage recommendation
        if factors.get('test_coverage', 0) > 0.7:
            recommendations.append(
                "Low test coverage for changed areas. "
                "Add unit tests before deploying."
            )
        
        # High user impact recommendation
        if factors.get('user_impact', 0) > 0.8:
            recommendations.append(
                f"High user impact component changed: {', '.join(components)}. "
                "Run smoke tests and monitor closely after deployment."
            )
        
        # Historical failure recommendation
        if factors.get('historical_failure', 0) > 0.7:
            recommendations.append(
                "Similar changes have failed historically. "
                "Increase test coverage and add additional validation."
            )
        
        if not recommendations:
            recommendations.append("Risk level is acceptable. Standard testing should suffice.")
        
        return recommendations
    
    def _calculate_confidence(self, components: List[str]) -> float:
        """Calculate confidence in the risk assessment."""
        
        # Confidence based on available historical data
        # More data = higher confidence
        
        confidence_by_component = {
            'authentication': 0.9,
            'payment': 0.9,
            'database': 0.8,
            'ui': 0.7,
            'api': 0.8,
            'unknown': 0.3
        }
        
        if not components:
            return 0.3
        
        # Use the highest confidence among components
        max_confidence = 0.0
        for component in components:
            confidence = confidence_by_component.get(component, 0.3)
            max_confidence = max(max_confidence, confidence)
        
        return max_confidence
    
    def _group_similar_crashes(self, crash_events: List[CrashEvent]) -> List[List[CrashEvent]]:
        """Group similar crashes together."""
        
        try:
            if not crash_events:
                return []
            
            # Extract text features from crash events
            crash_texts = []
            for event in crash_events:
                text = f"{event.error_type} {event.error_message} {event.stack_trace}"
                crash_texts.append(text)
            
            # Use TF-IDF to vectorize crash descriptions
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(crash_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Group crashes based on similarity threshold
            threshold = 0.7
            groups = []
            used_indices = set()
            
            for i, event in enumerate(crash_events):
                if i in used_indices:
                    continue
                
                group = [event]
                used_indices.add(i)
                
                # Find similar crashes
                for j, other_event in enumerate(crash_events[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    if similarity_matrix[i][j] > threshold:
                        group.append(other_event)
                        used_indices.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping similar crashes: {e}")
            return [[event] for event in crash_events]
    
    def _create_crash_pattern(self, crash_group: List[CrashEvent]) -> CrashPattern:
        """Create a crash pattern from a group of similar crashes."""
        
        pattern_id = f"pattern_{len(self.crash_patterns) + 1}"
        
        # Calculate frequency
        frequency = len(crash_group)
        
        # Calculate severity (average of individual severities)
        severity_map = {
            'critical': 1.0,
            'error': 0.8,
            'warning': 0.6,
            'info': 0.4
        }
        
        severity = sum(severity_map.get(event.severity, 0.5) for event in crash_group) / frequency
        
        # Extract affected components
        components = set()
        for event in crash_group:
            if event.device_info:
                # Extract component from device info or stack trace
                components.add("ui")  # Simplified
        
        # Extract common stack traces
        stack_traces = [event.stack_trace for event in crash_group if event.stack_trace]
        
        # Calculate user impact (based on frequency and severity)
        user_impact = (frequency * severity) / 100.0  # Normalize
        
        return CrashPattern(
            pattern_id=pattern_id,
            description=f"Crash pattern affecting {frequency} events",
            frequency=frequency,
            severity=severity,
            affected_components=list(components),
            common_stack_traces=stack_traces[:5],  # Top 5
            user_impact=min(user_impact, 1.0)
        )
