"""Feedback loop for continuous improvement of the AI testing system."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config.settings import get_settings
from src.test_generation.test_scenario import TestScenario
from src.test_execution.test_runner import TestResult

logger = logging.getLogger(__name__)


@dataclass
class HumanFeedback:
    """Data store for human feedback on test results aka typical structure ofcomments and corrections."""
    feedback_id: str
    test_result_id: str
    feedback_type: str  # correction, improvement, validation
    rating: int  # 1-5 scale
    comments: str
    corrections: Dict[str, Any]
    timestamp: datetime
    reviewer: str


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress."""
    model_version: str
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    feedback_count: int
    improvement_rate: float


class FeedbackLoop:
    """Manages feedback loop for continuous improvement."""
    
    def __init__(self):
        self.settings = get_settings()
        self.feedback_history = []
        self.learning_model = None
        self.metrics_history = []
        self.improvement_tracker = {}
    
    def record_human_feedback(self, feedback: HumanFeedback) -> bool:
        """Record human feedback for learning."""
        
        try:
            self.feedback_history.append(feedback)
            
            # Update improvement tracker
            self._update_improvement_tracker(feedback)
            
            # Trigger learning if enough feedback accumulated
            if len(self.feedback_history) % 10 == 0:  # Every 10 feedbacks
                self._trigger_learning()
            
            logger.info(f"Recorded feedback {feedback.feedback_id} for test {feedback.test_result_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording human feedback: {e}")
            return False
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in human feedback."""
        
        try:
            if not self.feedback_history:
                return {"error": "No feedback data available"}
            
            # Analyze feedback by type
            feedback_by_type = {}
            for feedback in self.feedback_history:
                feedback_type = feedback.feedback_type
                if feedback_type not in feedback_by_type:
                    feedback_by_type[feedback_type] = []
                feedback_by_type[feedback_type].append(feedback)
            
            # Calculate average ratings
            avg_ratings = {}
            for feedback_type, feedbacks in feedback_by_type.items():
                ratings = [f.rating for f in feedbacks]
                avg_ratings[feedback_type] = sum(ratings) / len(ratings)
            
            # Analyze improvement trends
            improvement_trend = self._calculate_improvement_trend()
            
            # Identify common correction patterns
            common_corrections = self._identify_common_corrections()
            
            return {
                "total_feedback": len(self.feedback_history),
                "feedback_by_type": {k: len(v) for k, v in feedback_by_type.items()},
                "average_ratings": avg_ratings,
                "improvement_trend": improvement_trend,
                "common_corrections": common_corrections,
                "recommendations": self._generate_improvement_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {"error": str(e)}
    
    def train_improvement_model(self) -> LearningMetrics:
        """Train ML model for improvement based on feedback."""
        
        try:
            if len(self.feedback_history) < 20:
                return LearningMetrics(
                    model_version="insufficient_data",
                    accuracy_score=0.0,
                    precision_score=0.0,
                    recall_score=0.0,
                    f1_score=0.0,
                    feedback_count=len(self.feedback_history),
                    improvement_rate=0.0
                )
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if X is None or y is None:
                return LearningMetrics(
                    model_version="data_preparation_failed",
                    accuracy_score=0.0,
                    precision_score=0.0,
                    recall_score=0.0,
                    f1_score=0.0,
                    feedback_count=len(self.feedback_history),
                    improvement_rate=0.0
                )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            # TODO: Use a better model, like a neural network or a more sophisticated ML model
            self.learning_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.learning_model.fit(X_train, y_train)
            
            # Evaluate model
            # TODO: Research more sophisticated metrics for evaluation
            y_pred = self.learning_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate improvement rate
            improvement_rate = self._calculate_improvement_rate()
            
            metrics = LearningMetrics(
                model_version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy_score=r2,  # Using R² as accuracy proxy
                precision_score=0.0,  # Would need classification metrics
                recall_score=0.0,     # Would need classification metrics
                f1_score=0.0,         # Would need classification metrics
                feedback_count=len(self.feedback_history),
                improvement_rate=improvement_rate
            )
            
            self.metrics_history.append(metrics)
            
            logger.info(f"Trained improvement model with R² score: {r2:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training improvement model: {e}")
            return LearningMetrics(
                model_version="training_failed",
                accuracy_score=0.0,
                precision_score=0.0,
                recall_score=0.0,
                f1_score=0.0,
                feedback_count=len(self.feedback_history),
                improvement_rate=0.0
            )
    
    def predict_improvement_potential(self, test_scenario: TestScenario) -> float:
        """Predict improvement potential for a test scenario."""
        
        try:
            if self.learning_model is None:
                return 0.5  # Default if no model trained
            
            # Extract features from test scenario
            features = self._extract_scenario_features(test_scenario)
            
            # Make prediction
            improvement_potential = self.learning_model.predict([features])[0]
            
            # Normalize to 0-1 range
            improvement_potential = max(0.0, min(1.0, improvement_potential))
            
            return improvement_potential
            
        except Exception as e:
            logger.error(f"Error predicting improvement potential: {e}")
            return 0.5
    
    def suggest_test_improvements(self, test_scenario: TestScenario) -> List[str]:
        """Suggest improvements for a test scenario based on feedback patterns."""
        
        try:
            improvements = []
            
            # Analyze feedback patterns for similar scenarios
            similar_feedback = self._find_similar_feedback(test_scenario)
            
            # Extract common improvement suggestions
            for feedback in similar_feedback:
                if feedback.comments:
                    improvements.extend(self._extract_improvement_suggestions(feedback.comments))
            
            # Add generic improvements based on scenario characteristics
            improvements.extend(self._generate_generic_improvements(test_scenario))
            
            # Remove duplicates and limit
            unique_improvements = list(set(improvements))[:10]
            
            return unique_improvements
            
        except Exception as e:
            logger.error(f"Error suggesting test improvements: {e}")
            return ["Unable to generate suggestions due to error"]
    
    def _update_improvement_tracker(self, feedback: HumanFeedback):
        """Update improvement tracking metrics."""
        
        # Track ratings over time
        if 'rating_trend' not in self.improvement_tracker:
            self.improvement_tracker['rating_trend'] = []
        
        self.improvement_tracker['rating_trend'].append({
            'timestamp': feedback.timestamp,
            'rating': feedback.rating,
            'type': feedback.feedback_type
        })
        
        # Track correction patterns
        if 'corrections' not in self.improvement_tracker:
            self.improvement_tracker['corrections'] = {}
        
        for correction_type, correction_value in feedback.corrections.items():
            if correction_type not in self.improvement_tracker['corrections']:
                self.improvement_tracker['corrections'][correction_type] = []
            
            self.improvement_tracker['corrections'][correction_type].append({
                'timestamp': feedback.timestamp,
                'value': correction_value
            })
    
    def _trigger_learning(self):
        """Trigger learning process when enough feedback is available."""
        
        try:
            logger.info(f"Triggering learning with {len(self.feedback_history)} feedback records")
            
            # Train improvement model
            metrics = self.train_improvement_model()
            
            # Update system based on learning
            self._update_system_based_on_learning()
            
            logger.info(f"Learning completed with accuracy: {metrics.accuracy_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error in learning trigger: {e}")
    
    def _calculate_improvement_trend(self) -> Dict[str, float]:
        """Calculate improvement trend over time."""
        
        try:
            if 'rating_trend' not in self.improvement_tracker:
                return {"trend": "no_data", "slope": 0.0}
            
            ratings_data = self.improvement_tracker['rating_trend']
            
            if len(ratings_data) < 2:
                return {"trend": "insufficient_data", "slope": 0.0}
            
            # Calculate trend slope
            ratings = [r['rating'] for r in ratings_data]
            x = list(range(len(ratings)))
            
            # Simple linear regression
            n = len(ratings)
            sum_x = sum(x)
            sum_y = sum(ratings)
            sum_xy = sum(x[i] * ratings[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            return {"trend": trend, "slope": slope}
            
        except Exception as e:
            logger.error(f"Error calculating improvement trend: {e}")
            return {"trend": "error", "slope": 0.0}
    
    def _identify_common_corrections(self) -> List[str]:
        """Identify common correction patterns from feedback."""
        
        try:
            if 'corrections' not in self.improvement_tracker:
                return []
            
            correction_counts = {}
            for correction_type, corrections in self.improvement_tracker['corrections'].items():
                correction_counts[correction_type] = len(corrections)
            
            # Sort by frequency
            sorted_corrections = sorted(
                correction_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [correction_type for correction_type, count in sorted_corrections[:5]]
            
        except Exception as e:
            logger.error(f"Error identifying common corrections: {e}")
            return []
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on feedback analysis."""
        
        recommendations = []
        
        try:
            # Analyze rating trends
            trend = self._calculate_improvement_trend()
            if trend['trend'] == 'declining':
                recommendations.append("Overall quality is declining. Review recent changes.")
            
            # Analyze common corrections
            common_corrections = self._identify_common_corrections()
            for correction in common_corrections[:3]:
                recommendations.append(f"Address frequent {correction} issues.")
            
            # Analyze feedback volume
            recent_feedback = [
                f for f in self.feedback_history
                if f.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if len(recent_feedback) > 50:
                recommendations.append("High feedback volume - consider automated improvements.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating improvement recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ML model."""
        
        try:
            if len(self.feedback_history) < 10:
                return None, None
            
            X = []
            y = []
            
            for feedback in self.feedback_history:
                # Extract features
                features = [
                    feedback.rating,
                    1 if feedback.feedback_type == 'correction' else 0,
                    1 if feedback.feedback_type == 'improvement' else 0,
                    len(feedback.comments),
                    len(feedback.corrections)
                ]
                
                X.append(features)
                
                # Target: improvement potential (based on rating and corrections)
                target = feedback.rating / 5.0  # Normalize rating
                if feedback.feedback_type == 'correction':
                    target *= 0.8  # Corrections indicate higher improvement potential
                
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate overall improvement rate."""
        
        try:
            if len(self.feedback_history) < 2:
                return 0.0
            
            # Calculate average rating improvement over time
            recent_feedback = sorted(self.feedback_history, key=lambda x: x.timestamp)
            
            # Split into two halves
            mid_point = len(recent_feedback) // 2
            first_half = recent_feedback[:mid_point]
            second_half = recent_feedback[mid_point:]
            
            if not first_half or not second_half:
                return 0.0
            
            avg_first = sum(f.rating for f in first_half) / len(first_half)
            avg_second = sum(f.rating for f in second_half) / len(second_half)
            
            improvement_rate = (avg_second - avg_first) / avg_first if avg_first > 0 else 0.0
            
            return max(-1.0, min(1.0, improvement_rate))  # Clamp to [-1, 1]
            
        except Exception as e:
            logger.error(f"Error calculating improvement rate: {e}")
            return 0.0
    
    def _extract_scenario_features(self, scenario: TestScenario) -> List[float]:
        """Extract features from test scenario for prediction."""
        
        return [
            len(scenario.steps),
            len(scenario.prerequisites),
            len(scenario.tags),
            scenario.get_estimated_duration(),
            1 if scenario.priority.value == 'critical' else 0,
            1 if scenario.priority.value == 'high' else 0,
            1 if scenario.test_type.value == 'e2e' else 0,
            1 if scenario.test_type.value == 'ui' else 0
        ]
    
    def _find_similar_feedback(self, scenario: TestScenario) -> List[HumanFeedback]:
        """Find similar feedback based on scenario characteristics."""
        
        # Simple similarity based on tags and test type
        similar_feedback = []
        
        for feedback in self.feedback_history:
            # This would be more sophisticated in practice
            # For now, just return recent feedback
            if len(similar_feedback) < 10:
                similar_feedback.append(feedback)
        
        return similar_feedback
    
    def _extract_improvement_suggestions(self, comments: str) -> List[str]:
        """Extract improvement suggestions from comments."""
        
        # Simple keyword-based extraction
        suggestions = []
        
        if 'timeout' in comments.lower():
            suggestions.append("Consider adding explicit waits")
        
        if 'flaky' in comments.lower():
            suggestions.append("Improve test stability")
        
        if 'slow' in comments.lower():
            suggestions.append("Optimize test performance")
        
        if 'unclear' in comments.lower():
            suggestions.append("Improve test documentation")
        
        return suggestions
    
    def _generate_generic_improvements(self, scenario: TestScenario) -> List[str]:
        """Generate generic improvement suggestions."""
        
        improvements = []
        
        if scenario.get_estimated_duration() > 300:
            improvements.append("Consider breaking down long test")
        
        if len(scenario.steps) > 20:
            improvements.append("Reduce number of test steps")
        
        if not scenario.prerequisites:
            improvements.append("Add test prerequisites")
        
        if len(scenario.tags) < 3:
            improvements.append("Add more descriptive tags")
        
        return improvements
    
    def _update_system_based_on_learning(self):
        """Update system parameters based on learning results."""
        
        try:
            # This would update various system parameters
            # For now, just log that learning occurred
            logger.info("System updated based on learning results")
            
        except Exception as e:
            logger.error(f"Error updating system based on learning: {e}")
    
    def export_feedback_data(self, output_path: str) -> bool:
        """Export feedback data for external analysis."""
        
        try:
            export_data = {
                "feedback_history": [
                    {
                        "feedback_id": f.feedback_id,
                        "test_result_id": f.test_result_id,
                        "feedback_type": f.feedback_type,
                        "rating": f.rating,
                        "comments": f.comments,
                        "corrections": f.corrections,
                        "timestamp": f.timestamp.isoformat(),
                        "reviewer": f.reviewer
                    }
                    for f in self.feedback_history
                ],
                "improvement_tracker": self.improvement_tracker,
                "metrics_history": [
                    {
                        "model_version": m.model_version,
                        "accuracy_score": m.accuracy_score,
                        "improvement_rate": m.improvement_rate,
                        "feedback_count": m.feedback_count
                    }
                    for m in self.metrics_history
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Feedback data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feedback data: {e}")
            return False
