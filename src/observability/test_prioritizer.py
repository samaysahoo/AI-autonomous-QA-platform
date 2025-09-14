"""Test prioritization based on observability data and risk analysis."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import get_settings
from src.test_generation.test_scenario import TestScenario, TestPriority
from .risk_analyzer import RiskScore, CrashPattern

logger = logging.getLogger(__name__)


@dataclass
class TestPriorityScore:
    """Represents a priority score for a test scenario."""
    scenario_id: str
    priority_score: float  # 0.0 to 1.0
    risk_factors: Dict[str, float]
    execution_time_estimate: int  # seconds
    expected_failure_rate: float
    business_impact: float
    recommendations: List[str]


@dataclass
class TestSuite:
    """Represents a prioritized test suite."""
    suite_id: str
    name: str
    scenarios: List[TestScenario]
    priority_scores: List[TestPriorityScore]
    total_estimated_time: int
    coverage_areas: List[str]


class TestPrioritizer:
    """Prioritizes test scenarios based on risk analysis and observability data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.risk_analyzer = None  # Will be injected
        self.crash_patterns = []
        self.component_risk_scores = {}
    
    def prioritize_tests(self, 
                        scenarios: List[TestScenario],
                        risk_scores: List[RiskScore],
                        crash_patterns: List[CrashPattern],
                        time_constraint: Optional[int] = None) -> TestSuite:
        """Prioritize test scenarios based on risk and time constraints."""
        
        try:
            # Calculate priority scores for each scenario
            priority_scores = []
            for scenario in scenarios:
                score = self._calculate_priority_score(
                    scenario, risk_scores, crash_patterns
                )
                priority_scores.append(score)
            
            # Sort scenarios by priority score
            sorted_scenarios = sorted(
                zip(scenarios, priority_scores),
                key=lambda x: x[1].priority_score,
                reverse=True
            )
            
            # Apply time constraint if provided
            if time_constraint:
                selected_scenarios, selected_scores = self._apply_time_constraint(
                    sorted_scenarios, time_constraint
                )
            else:
                selected_scenarios = [scenario for scenario, _ in sorted_scenarios]
                selected_scores = [score for _, score in sorted_scenarios]
            
            # Create test suite
            suite = TestSuite(
                suite_id=f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="AI-Prioritized Test Suite",
                scenarios=selected_scenarios,
                priority_scores=selected_scores,
                total_estimated_time=sum(score.execution_time_estimate for score in selected_scores),
                coverage_areas=self._extract_coverage_areas(selected_scenarios)
            )
            
            logger.info(f"Created prioritized test suite with {len(selected_scenarios)} scenarios")
            return suite
            
        except Exception as e:
            logger.error(f"Error prioritizing tests: {e}")
            # Return fallback suite with original scenarios
            return TestSuite(
                suite_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Fallback Test Suite",
                scenarios=scenarios,
                priority_scores=[],
                total_estimated_time=0,
                coverage_areas=[]
            )
    
    def create_smoke_test_suite(self, 
                              scenarios: List[TestScenario],
                              risk_scores: List[RiskScore]) -> TestSuite:
        """Create a focused smoke test suite for high-risk changes."""
        
        try:
            # Filter for high-risk scenarios
            high_risk_scenarios = []
            high_risk_scores = []
            
            for scenario, risk_score in zip(scenarios, risk_scores):
                if risk_score.risk_level > 0.7:  # High risk threshold
                    high_risk_scenarios.append(scenario)
                    high_risk_scores.append(risk_score)
            
            # Prioritize by business impact and failure probability
            smoke_priorities = []
            for scenario, risk_score in zip(high_risk_scenarios, high_risk_scores):
                priority_score = self._calculate_smoke_priority(scenario, risk_score)
                smoke_priorities.append(priority_score)
            
            # Sort by priority and take top scenarios
            sorted_smoke = sorted(
                zip(high_risk_scenarios, smoke_priorities),
                key=lambda x: x[1].priority_score,
                reverse=True
            )
            
            # Limit to 10 scenarios for smoke test
            selected_smoke = sorted_smoke[:10]
            
            suite = TestSuite(
                suite_id=f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Smoke Test Suite",
                scenarios=[scenario for scenario, _ in selected_smoke],
                priority_scores=[score for _, score in selected_smoke],
                total_estimated_time=sum(score.execution_time_estimate for _, score in selected_smoke),
                coverage_areas=self._extract_coverage_areas([scenario for scenario, _ in selected_smoke])
            )
            
            logger.info(f"Created smoke test suite with {len(selected_smoke)} scenarios")
            return suite
            
        except Exception as e:
            logger.error(f"Error creating smoke test suite: {e}")
            return TestSuite(
                suite_id="smoke_fallback",
                name="Fallback Smoke Test Suite",
                scenarios=scenarios[:5],  # Take first 5 as fallback
                priority_scores=[],
                total_estimated_time=0,
                coverage_areas=[]
            )
    
    def create_regression_test_suite(self, 
                                   scenarios: List[TestScenario],
                                   crash_patterns: List[CrashPattern]) -> TestSuite:
        """Create a regression test suite targeting known crash patterns."""
        
        try:
            # Find scenarios that address crash patterns
            regression_scenarios = []
            regression_priorities = []
            
            for scenario in scenarios:
                # Check if scenario addresses any crash patterns
                pattern_relevance = self._calculate_pattern_relevance(scenario, crash_patterns)
                
                if pattern_relevance > 0.5:  # Relevant to crash patterns
                    priority_score = TestPriorityScore(
                        scenario_id=scenario.scenario_id,
                        priority_score=pattern_relevance,
                        risk_factors={"crash_pattern_relevance": pattern_relevance},
                        execution_time_estimate=scenario.get_estimated_duration(),
                        expected_failure_rate=0.3,  # Assume 30% failure rate for regression tests
                        business_impact=0.8,  # High business impact for regression
                        recommendations=["Focus on crash pattern validation"]
                    )
                    
                    regression_scenarios.append(scenario)
                    regression_priorities.append(priority_score)
            
            # Sort by pattern relevance
            sorted_regression = sorted(
                zip(regression_scenarios, regression_priorities),
                key=lambda x: x[1].priority_score,
                reverse=True
            )
            
            suite = TestSuite(
                suite_id=f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Regression Test Suite",
                scenarios=[scenario for scenario, _ in sorted_regression],
                priority_scores=[score for _, score in sorted_regression],
                total_estimated_time=sum(score.execution_time_estimate for _, score in sorted_regression),
                coverage_areas=self._extract_coverage_areas([scenario for scenario, _ in sorted_regression])
            )
            
            logger.info(f"Created regression test suite with {len(sorted_regression)} scenarios")
            return suite
            
        except Exception as e:
            logger.error(f"Error creating regression test suite: {e}")
            return TestSuite(
                suite_id="regression_fallback",
                name="Fallback Regression Test Suite",
                scenarios=scenarios[:10],  # Take first 10 as fallback
                priority_scores=[],
                total_estimated_time=0,
                coverage_areas=[]
            )
    
    def _calculate_priority_score(self, 
                                scenario: TestScenario,
                                risk_scores: List[RiskScore],
                                crash_patterns: List[CrashPattern]) -> TestPriorityScore:
        """Calculate priority score for a test scenario."""
        
        risk_factors = {}
        
        # Factor 1: Risk level of affected components
        component_risk = self._get_component_risk(scenario, risk_scores)
        risk_factors['component_risk'] = component_risk
        
        # Factor 2: Crash pattern relevance
        pattern_relevance = self._calculate_pattern_relevance(scenario, crash_patterns)
        risk_factors['crash_pattern_relevance'] = pattern_relevance
        
        # Factor 3: Test priority level
        priority_multiplier = {
            TestPriority.CRITICAL: 1.0,
            TestPriority.HIGH: 0.8,
            TestPriority.MEDIUM: 0.6,
            TestPriority.LOW: 0.4
        }
        test_priority_factor = priority_multiplier.get(scenario.priority, 0.6)
        risk_factors['test_priority'] = test_priority_factor
        
        # Factor 4: Business impact based on test type
        business_impact = self._calculate_business_impact(scenario)
        risk_factors['business_impact'] = business_impact
        
        # Factor 5: Expected failure rate
        expected_failure_rate = self._calculate_expected_failure_rate(scenario)
        risk_factors['expected_failure_rate'] = expected_failure_rate
        
        # Calculate overall priority score
        priority_score = (
            component_risk * 0.3 +
            pattern_relevance * 0.25 +
            test_priority_factor * 0.2 +
            business_impact * 0.15 +
            expected_failure_rate * 0.1
        )
        
        # Generate recommendations
        recommendations = self._generate_priority_recommendations(risk_factors)
        
        return TestPriorityScore(
            scenario_id=scenario.scenario_id,
            priority_score=priority_score,
            risk_factors=risk_factors,
            execution_time_estimate=scenario.get_estimated_duration(),
            expected_failure_rate=expected_failure_rate,
            business_impact=business_impact,
            recommendations=recommendations
        )
    
    def _calculate_smoke_priority(self, scenario: TestScenario, 
                                risk_score: RiskScore) -> TestPriorityScore:
        """Calculate priority for smoke test scenarios."""
        
        # Smoke tests prioritize high-risk, high-impact scenarios
        priority_score = (
            risk_score.risk_level * 0.4 +
            self._calculate_business_impact(scenario) * 0.3 +
            self._calculate_user_impact(scenario) * 0.3
        )
        
        return TestPriorityScore(
            scenario_id=scenario.scenario_id,
            priority_score=priority_score,
            risk_factors={
                "risk_level": risk_score.risk_level,
                "business_impact": self._calculate_business_impact(scenario),
                "user_impact": self._calculate_user_impact(scenario)
            },
            execution_time_estimate=scenario.get_estimated_duration(),
            expected_failure_rate=0.2,  # Lower failure rate for smoke tests
            business_impact=self._calculate_business_impact(scenario),
            recommendations=["Critical for smoke testing"]
        )
    
    def _get_component_risk(self, scenario: TestScenario, 
                          risk_scores: List[RiskScore]) -> float:
        """Get risk level for components affected by the scenario."""
        
        # Extract components from scenario metadata or tags
        scenario_components = scenario.metadata.get('components', [])
        if not scenario_components:
            scenario_components = scenario.tags
        
        max_risk = 0.0
        for risk_score in risk_scores:
            if any(comp in risk_score.affected_areas for comp in scenario_components):
                max_risk = max(max_risk, risk_score.risk_level)
        
        return max_risk if max_risk > 0 else 0.3  # Default risk
    
    def _calculate_pattern_relevance(self, scenario: TestScenario, 
                                   crash_patterns: List[CrashPattern]) -> float:
        """Calculate how relevant a scenario is to known crash patterns."""
        
        if not crash_patterns:
            return 0.0
        
        # Simple text-based relevance calculation
        scenario_text = f"{scenario.title} {scenario.description} {' '.join(scenario.tags)}"
        
        max_relevance = 0.0
        for pattern in crash_patterns:
            # Calculate text similarity between scenario and pattern
            relevance = self._calculate_text_similarity(
                scenario_text, 
                f"{pattern.description} {' '.join(pattern.affected_components)}"
            )
            max_relevance = max(max_relevance, relevance)
        
        return max_relevance
    
    def _calculate_business_impact(self, scenario: TestScenario) -> float:
        """Calculate business impact of a test scenario."""
        
        # Business impact based on test type and priority
        type_impact = {
            'e2e': 0.9,
            'ui': 0.8,
            'integration': 0.7,
            'api': 0.6,
            'unit': 0.4
        }
        
        priority_impact = {
            TestPriority.CRITICAL: 1.0,
            TestPriority.HIGH: 0.8,
            TestPriority.MEDIUM: 0.6,
            TestPriority.LOW: 0.4
        }
        
        base_impact = type_impact.get(scenario.test_type.value, 0.5)
        priority_multiplier = priority_impact.get(scenario.priority, 0.6)
        
        return base_impact * priority_multiplier
    
    def _calculate_user_impact(self, scenario: TestScenario) -> float:
        """Calculate user impact of a test scenario."""
        
        # User impact based on test type
        user_impact_map = {
            'ui': 0.9,
            'e2e': 0.8,
            'integration': 0.6,
            'api': 0.5,
            'unit': 0.2
        }
        
        return user_impact_map.get(scenario.test_type.value, 0.5)
    
    def _calculate_expected_failure_rate(self, scenario: TestScenario) -> float:
        """Calculate expected failure rate for a scenario."""
        
        # Base failure rate by test type
        base_failure_rates = {
            'ui': 0.3,
            'e2e': 0.25,
            'integration': 0.2,
            'api': 0.15,
            'unit': 0.1
        }
        
        base_rate = base_failure_rates.get(scenario.test_type.value, 0.2)
        
        # Adjust based on priority (higher priority = more likely to fail)
        priority_adjustments = {
            TestPriority.CRITICAL: 1.2,
            TestPriority.HIGH: 1.1,
            TestPriority.MEDIUM: 1.0,
            TestPriority.LOW: 0.9
        }
        
        adjustment = priority_adjustments.get(scenario.priority, 1.0)
        
        return min(base_rate * adjustment, 1.0)
    
    def _apply_time_constraint(self, 
                             sorted_scenarios: List[Tuple[TestScenario, TestPriorityScore]],
                             time_constraint: int) -> Tuple[List[TestScenario], List[TestPriorityScore]]:
        """Apply time constraint to select scenarios."""
        
        selected_scenarios = []
        selected_scores = []
        total_time = 0
        
        for scenario, score in sorted_scenarios:
            if total_time + score.execution_time_estimate <= time_constraint:
                selected_scenarios.append(scenario)
                selected_scores.append(score)
                total_time += score.execution_time_estimate
            else:
                break
        
        return selected_scenarios, selected_scores
    
    def _extract_coverage_areas(self, scenarios: List[TestScenario]) -> List[str]:
        """Extract coverage areas from scenarios."""
        
        areas = set()
        for scenario in scenarios:
            areas.update(scenario.tags)
            areas.update(scenario.metadata.get('components', []))
        
        return list(areas)
    
    def _generate_priority_recommendations(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate recommendations based on risk factors."""
        
        recommendations = []
        
        if risk_factors.get('component_risk', 0) > 0.7:
            recommendations.append("High component risk - prioritize this test")
        
        if risk_factors.get('crash_pattern_relevance', 0) > 0.6:
            recommendations.append("Addresses known crash patterns")
        
        if risk_factors.get('business_impact', 0) > 0.8:
            recommendations.append("High business impact - critical for validation")
        
        if risk_factors.get('expected_failure_rate', 0) > 0.4:
            recommendations.append("High expected failure rate - good candidate for testing")
        
        return recommendations
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
