"""Test scenario data models and validation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    UI = "ui"
    API = "api"
    PERFORMANCE = "performance"
    SECURITY = "security"


class TestFramework(Enum):
    """Supported test frameworks."""
    APPIUM = "appium"
    ESPRESSO = "espresso"
    XCUITEST = "xcuitest"
    SELENIUM = "selenium"
    PYTEST = "pytest"
    JEST = "jest"


class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestStep:
    """Represents a single test step."""
    step_id: str
    description: str
    action: str
    expected_result: str
    locator: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    wait_time: Optional[int] = None


@dataclass
class TestScenario:
    """Represents a complete test scenario."""
    scenario_id: str
    title: str
    description: str
    test_type: TestType
    framework: TestFramework
    priority: TestPriority
    steps: List[TestStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    expected_duration: Optional[int] = None  # in seconds
    source_documents: List[str] = field(default_factory=list)  # IDs of source documents
    generated_by: str = "ai_test_generator"
    confidence_score: Optional[float] = None  # AI confidence in this test
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: TestStep):
        """Add a test step to the scenario."""
        self.steps.append(step)
    
    def get_step_count(self) -> int:
        """Get the number of steps in the scenario."""
        return len(self.steps)
    
    def get_estimated_duration(self) -> int:
        """Estimate test duration based on steps."""
        if self.expected_duration:
            return self.expected_duration
        
        # Rough estimation: 10 seconds per step + 5 seconds base
        return len(self.steps) * 10 + 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "test_type": self.test_type.value,
            "framework": self.framework.value,
            "priority": self.priority.value,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "action": step.action,
                    "expected_result": step.expected_result,
                    "locator": step.locator,
                    "input_data": step.input_data,
                    "wait_time": step.wait_time
                }
                for step in self.steps
            ],
            "prerequisites": self.prerequisites,
            "tags": self.tags,
            "expected_duration": self.get_estimated_duration(),
            "source_documents": self.source_documents,
            "generated_by": self.generated_by,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestScenario":
        """Create scenario from dictionary."""
        steps = [
            TestStep(
                step_id=step["step_id"],
                description=step["description"],
                action=step["action"],
                expected_result=step["expected_result"],
                locator=step.get("locator"),
                input_data=step.get("input_data"),
                wait_time=step.get("wait_time")
            )
            for step in data.get("steps", [])
        ]
        
        return cls(
            scenario_id=data["scenario_id"],
            title=data["title"],
            description=data["description"],
            test_type=TestType(data["test_type"]),
            framework=TestFramework(data["framework"]),
            priority=TestPriority(data["priority"]),
            steps=steps,
            prerequisites=data.get("prerequisites", []),
            tags=data.get("tags", []),
            expected_duration=data.get("expected_duration"),
            source_documents=data.get("source_documents", []),
            generated_by=data.get("generated_by", "ai_test_generator"),
            confidence_score=data.get("confidence_score"),
            metadata=data.get("metadata", {})
        )
