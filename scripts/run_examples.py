#!/usr/bin/env python3
"""Run example workflows to demonstrate the AI Test Automation Platform."""

import sys
import os
import asyncio
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.test_generation import TestCaseGenerator
from src.test_generation.test_scenario import TestType, TestFramework
from src.observability import RiskAnalyzer, TestPrioritizer
from src.dashboard import FailureClusterer, RootCauseAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_test_generation():
    """Demonstrate test generation capabilities."""
    
    logger.info("=== Test Generation Example ===")
    
    try:
        test_generator = TestCaseGenerator()
        
        # Generate test scenarios for user login
        scenarios = test_generator.generate_test_scenarios(
            query="User login with email and password, including error handling for invalid credentials",
            test_type=TestType.E2E,
            framework=TestFramework.APPIUM,
            max_scenarios=3
        )
        
        logger.info(f"Generated {len(scenarios)} test scenarios")
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\nScenario {i}: {scenario.title}")
            logger.info(f"  Description: {scenario.description}")
            logger.info(f"  Priority: {scenario.priority.value}")
            logger.info(f"  Steps: {scenario.get_step_count()}")
            logger.info(f"  Estimated Duration: {scenario.get_estimated_duration()}s")
            
            if scenario.steps:
                logger.info("  Test Steps:")
                for j, step in enumerate(scenario.steps, 1):
                    logger.info(f"    {j}. {step.description}")
                    logger.info(f"       Action: {step.action}")
                    logger.info(f"       Expected: {step.expected_result}")
        
        # Generate test code for the first scenario
        if scenarios:
            from src.test_generation import CodeGenerator
            code_generator = CodeGenerator()
            test_code = code_generator.generate_test_code(scenarios[0])
            
            logger.info(f"\nGenerated Test Code:")
            logger.info(f"  File: {test_code['filename']}")
            logger.info(f"  Framework: {test_code['framework']}")
            logger.info(f"  Language: {test_code['language']}")
            
            # Show a snippet of the generated code
            code_lines = test_code['code'].split('\n')
            logger.info("  Code Preview:")
            for line in code_lines[:10]:  # Show first 10 lines
                logger.info(f"    {line}")
            if len(code_lines) > 10:
                logger.info("    ...")
        
        logger.info("\nTest generation example completed successfully!")
        
    except Exception as e:
        logger.error(f"Test generation example failed: {e}")


async def example_risk_analysis():
    """Demonstrate risk analysis capabilities."""
    
    logger.info("\n=== Risk Analysis Example ===")
    
    try:
        risk_analyzer = RiskAnalyzer()
        
        # Example code diff for authentication changes
        diff_content = """
        + def authenticate_user(username, password):
        +     # New authentication logic
        +     if not username or not password:
        +         raise ValueError("Username and password required")
        +     
        +     user = get_user_by_username(username)
        +     if user and verify_password(password, user.password_hash):
        +         return create_session(user.id)
        +     return None
        
        + def verify_password(password, password_hash):
        +     return bcrypt.checkpw(password.encode('utf-8'), password_hash)
        
        - def old_authenticate(username, password):
        -     # Old insecure authentication
        -     return username == "admin" and password == "password"
        """
        
        changed_files = [
            "src/auth/authentication.py",
            "src/auth/password_utils.py",
            "src/models/user.py"
        ]
        
        commit_metadata = {
            "hash": "a1b2c3d4e5f6",
            "message": "Implement secure authentication with bcrypt",
            "author": "security-team@company.com",
            "branch": "feature/secure-auth"
        }
        
        # Analyze risk
        risk_score = risk_analyzer.analyze_code_change_risk(
            diff_content=diff_content,
            changed_files=changed_files,
            commit_metadata=commit_metadata
        )
        
        logger.info(f"Risk Analysis Results:")
        logger.info(f"  Overall Risk Level: {risk_score.risk_level:.2f}/1.0")
        logger.info(f"  Confidence Score: {risk_score.confidence:.2f}/1.0")
        logger.info(f"  Affected Components: {', '.join(risk_score.affected_areas)}")
        
        logger.info(f"\nRisk Factors:")
        for factor, value in risk_score.factors.items():
            logger.info(f"  {factor}: {value:.2f}")
        
        logger.info(f"\nRecommendations:")
        for i, rec in enumerate(risk_score.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # Analyze crash patterns (simulated)
        logger.info(f"\nCrash Pattern Analysis:")
        simulated_crashes = [
            {
                "error_type": "AuthenticationError",
                "error_message": "Invalid credentials provided",
                "stack_trace": "AuthenticationError at auth.py:45",
                "severity": "Medium",
                "frequency": 25
            },
            {
                "error_type": "DatabaseConnectionError", 
                "error_message": "Connection pool exhausted",
                "stack_trace": "DatabaseConnectionError at db.py:123",
                "severity": "High",
                "frequency": 8
            }
        ]
        
        patterns = risk_analyzer.analyze_crash_patterns(simulated_crashes)
        logger.info(f"  Identified {len(patterns)} crash patterns")
        
        for pattern in patterns:
            logger.info(f"    Pattern {pattern.pattern_id}: {pattern.description}")
            logger.info(f"      Frequency: {pattern.frequency}, Severity: {pattern.severity:.2f}")
        
        logger.info("\nRisk analysis example completed successfully!")
        
    except Exception as e:
        logger.error(f"Risk analysis example failed: {e}")


async def example_test_prioritization():
    """Demonstrate test prioritization capabilities."""
    
    logger.info("\n=== Test Prioritization Example ===")
    
    try:
        from src.observability import TestPrioritizer
        from src.test_generation.test_scenario import TestScenario, TestType, TestFramework, TestPriority
        
        test_prioritizer = TestPrioritizer()
        
        # Create sample test scenarios
        scenarios = [
            TestScenario(
                scenario_id="test_1",
                title="User Login Flow",
                description="Test complete user login process",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                priority=TestPriority.HIGH,
                tags=["authentication", "login"],
                metadata={"components": ["authentication"]}
            ),
            TestScenario(
                scenario_id="test_2",
                title="Payment Processing",
                description="Test payment transaction flow",
                test_type=TestType.E2E,
                framework=TestFramework.APPIUM,
                priority=TestPriority.CRITICAL,
                tags=["payment", "transaction"],
                metadata={"components": ["payment"]}
            ),
            TestScenario(
                scenario_id="test_3",
                title="UI Navigation",
                description="Test app navigation and menu interactions",
                test_type=TestType.UI,
                framework=TestFramework.APPIUM,
                priority=TestPriority.MEDIUM,
                tags=["ui", "navigation"],
                metadata={"components": ["ui"]}
            )
        ]
        
        # Create sample risk scores
        from src.observability.risk_analyzer import RiskScore
        
        risk_scores = [
            RiskScore(
                component="authentication",
                risk_level=0.8,
                factors={"crash_frequency": 0.8, "user_impact": 0.9},
                confidence=0.9,
                recommendations=["High risk authentication changes"],
                affected_areas=["authentication"]
            ),
            RiskScore(
                component="payment",
                risk_level=0.9,
                factors={"crash_frequency": 0.9, "user_impact": 0.95},
                confidence=0.95,
                recommendations=["Critical payment system changes"],
                affected_areas=["payment"]
            ),
            RiskScore(
                component="ui",
                risk_level=0.4,
                factors={"crash_frequency": 0.3, "user_impact": 0.6},
                confidence=0.7,
                recommendations=["Low risk UI changes"],
                affected_areas=["ui"]
            )
        ]
        
        # Prioritize tests
        prioritized_suite = test_prioritizer.prioritize_tests(
            scenarios=scenarios,
            risk_scores=risk_scores,
            crash_patterns=[],
            time_constraint=1800  # 30 minutes
        )
        
        logger.info(f"Prioritized Test Suite:")
        logger.info(f"  Suite ID: {prioritized_suite.suite_id}")
        logger.info(f"  Total Scenarios: {len(prioritized_suite.scenarios)}")
        logger.info(f"  Estimated Time: {prioritized_suite.total_estimated_time}s")
        logger.info(f"  Coverage Areas: {', '.join(prioritized_suite.coverage_areas)}")
        
        logger.info(f"\nPrioritized Scenarios:")
        for i, (scenario, score) in enumerate(zip(prioritized_suite.scenarios, prioritized_suite.priority_scores), 1):
            logger.info(f"  {i}. {scenario.title}")
            logger.info(f"     Priority Score: {score.priority_score:.2f}")
            logger.info(f"     Business Impact: {score.business_impact:.2f}")
            logger.info(f"     Expected Duration: {score.execution_time_estimate}s")
        
        # Create smoke test suite
        smoke_suite = test_prioritizer.create_smoke_test_suite(scenarios, risk_scores)
        
        logger.info(f"\nSmoke Test Suite:")
        logger.info(f"  Scenarios: {len(smoke_suite.scenarios)}")
        logger.info(f"  Estimated Time: {smoke_suite.total_estimated_time}s")
        
        logger.info("\nTest prioritization example completed successfully!")
        
    except Exception as e:
        logger.error(f"Test prioritization example failed: {e}")


async def example_failure_analysis():
    """Demonstrate failure clustering and root cause analysis."""
    
    logger.info("\n=== Failure Analysis Example ===")
    
    try:
        # Create sample failure data
        sample_failures = [
            {
                "test_name": "test_user_login",
                "error_message": "Element not found: login button",
                "stack_trace": "NoSuchElementException at LoginPage.java:45",
                "duration": 30,
                "timestamp": "2024-01-15T10:30:00Z",
                "environment": "staging",
                "browser": "chrome"
            },
            {
                "test_name": "test_user_registration", 
                "error_message": "Element not found: register button",
                "stack_trace": "NoSuchElementException at RegistrationPage.java:67",
                "duration": 45,
                "timestamp": "2024-01-15T10:35:00Z",
                "environment": "staging",
                "browser": "chrome"
            },
            {
                "test_name": "test_payment_flow",
                "error_message": "Timeout waiting for payment form",
                "stack_trace": "TimeoutException at PaymentPage.java:123",
                "duration": 60,
                "timestamp": "2024-01-15T10:40:00Z",
                "environment": "staging",
                "browser": "firefox"
            },
            {
                "test_name": "test_checkout_process",
                "error_message": "Timeout waiting for checkout form", 
                "stack_trace": "TimeoutException at CheckoutPage.java:89",
                "duration": 55,
                "timestamp": "2024-01-15T10:45:00Z",
                "environment": "staging",
                "browser": "firefox"
            }
        ]
        
        # Cluster failures
        logger.info("Clustering test failures...")
        failure_clusterer = FailureClusterer()
        clustering_result = failure_clusterer.cluster_failures(sample_failures)
        
        logger.info(f"Clustering Results:")
        logger.info(f"  Method Used: {clustering_result.method_used}")
        logger.info(f"  Number of Clusters: {len(clustering_result.clusters)}")
        logger.info(f"  Silhouette Score: {clustering_result.silhouette_score:.3f}")
        
        for cluster in clustering_result.clusters:
            logger.info(f"\n  Cluster {cluster.cluster_id}:")
            logger.info(f"    Size: {cluster.size} failures")
            logger.info(f"    Confidence: {cluster.confidence_score:.2f}")
            logger.info(f"    Common Patterns: {', '.join(cluster.common_patterns[:3])}")
        
        # Analyze root causes
        if clustering_result.clusters:
            logger.info(f"\nAnalyzing root causes...")
            root_cause_analyzer = RootCauseAnalyzer()
            root_cause_analyses = root_cause_analyzer.analyze_root_causes(clustering_result.clusters)
            
            logger.info(f"Root Cause Analysis Results:")
            for analysis in root_cause_analyses:
                logger.info(f"\n  Cluster {analysis.cluster_id}:")
                logger.info(f"    Primary Cause: {analysis.primary_cause}")
                logger.info(f"    Confidence: {analysis.confidence_score:.2f}")
                logger.info(f"    Contributing Factors: {', '.join(analysis.contributing_factors[:3])}")
                logger.info(f"    Top Recommendations:")
                for i, rec in enumerate(analysis.recommendations[:2], 1):
                    logger.info(f"      {i}. {rec}")
            
            # Generate summary
            summary = root_cause_analyzer.generate_summary(root_cause_analyses)
            
            logger.info(f"\nAnalysis Summary:")
            logger.info(f"  Total Clusters: {summary.total_clusters}")
            logger.info(f"  Analyzed Clusters: {summary.analyzed_clusters}")
            logger.info(f"  Overall Confidence: {summary.overall_confidence:.2f}")
            logger.info(f"  Common Causes: {', '.join(summary.common_causes[:3])}")
        
        logger.info("\nFailure analysis example completed successfully!")
        
    except Exception as e:
        logger.error(f"Failure analysis example failed: {e}")


async def main():
    """Run all examples."""
    
    logger.info("Starting AI Test Automation Platform Examples")
    logger.info("=" * 60)
    
    try:
        # Run all examples
        await example_test_generation()
        await example_risk_analysis()
        await example_test_prioritization()
        await example_failure_analysis()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("\nTo run the dashboard, use: python main.py dashboard")
        logger.info("To initialize the vector store, use: python main.py init")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
