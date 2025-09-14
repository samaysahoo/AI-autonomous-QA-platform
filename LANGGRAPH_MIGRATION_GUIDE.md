# Migration Guide: Custom Multi-Agent to LangGraph Implementation

## Overview

This guide helps you migrate from the custom multi-agent implementation to the industry-standard LangGraph-based system. LangGraph provides better state management, workflow orchestration, and debugging capabilities.

## Why Migrate to LangGraph?

### Advantages of LangGraph

1. **Industry Standard**: LangGraph is the de facto standard for multi-agent systems in the LangChain ecosystem
2. **Better State Management**: Built-in state persistence with checkpointing and resumability
3. **Robust Workflow Orchestration**: Graph-based workflow execution with conditional routing
4. **Enhanced Debugging**: Better observability and error handling
5. **Message History**: Built-in message management for agent communication
6. **Persistence**: Automatic checkpointing for workflow recovery
7. **Community Support**: Large community and extensive documentation

### Comparison

| Feature | Custom Implementation | LangGraph Implementation |
|---------|----------------------|--------------------------|
| Framework | Custom coordinator | LangGraph StateGraph |
| State Management | Manual state tracking | Built-in TypedDict state |
| Workflow Execution | Custom orchestration | Graph-based execution |
| Message Handling | Custom message queue | Built-in message history |
| Error Recovery | Manual error handling | Built-in error handling |
| Persistence | Manual checkpointing | Automatic checkpointing |
| Debugging | Limited observability | Rich debugging tools |
| Community | Custom solution | Industry standard |

## Migration Steps

### 1. Update Dependencies

Update your `requirements.txt`:

```diff
# Core AI/ML dependencies
openai==1.3.0
-langchain==0.0.350
-langchain-openai==0.0.2
+langchain==0.1.0
+langchain-openai==0.0.5
+langgraph==0.0.20
+langgraph-checkpoint==0.0.3
```

### 2. Import Changes

**Before (Custom Implementation):**
```python
from src.agents.agent_coordinator import AgentCoordinator
from src.agents.agent_registry import AgentRegistry
from src.agents.test_planner_agent import TestPlannerAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.learning_agent import LearningAgent
```

**After (LangGraph Implementation):**
```python
from src.langgraph_agents.workflow_graph import TestAutomationWorkflow
from src.langgraph_agents.state import TestAutomationState, create_initial_state
from src.langgraph_agents.test_planner_agent import TestPlannerAgent
from src.langgraph_agents.execution_agent import ExecutionAgent
from src.langgraph_agents.diagnosis_agent import DiagnosisAgent
from src.langgraph_agents.learning_agent import LearningAgent
```

### 3. Initialization Changes

**Before (Custom Implementation):**
```python
# Initialize coordinator and registry
coordinator = AgentCoordinator()
registry = AgentRegistry()

# Register agents
test_planner = TestPlannerAgent("test-planner-001")
registry.register_agent(test_planner)
coordinator.agents["test-planner"] = test_planner

# Start coordination
await coordinator.start_coordination()
```

**After (LangGraph Implementation):**
```python
# Initialize workflow
workflow = TestAutomationWorkflow()

# Agents are automatically initialized and managed by the workflow
# No manual registration needed
```

### 4. Workflow Execution Changes

**Before (Custom Implementation):**
```python
# Start workflow manually
execution_id = await coordinator.start_workflow("e2e-test-workflow", input_data)

# Monitor workflow status
workflow_status = coordinator.active_workflows.get(execution_id)
```

**After (LangGraph Implementation):**
```python
# Execute workflow with automatic state management
result = await workflow.execute_workflow(
    workflow_id="e2e-test-workflow",
    input_data=input_data,
    config=config
)

# Result contains complete execution summary
print(f"Status: {result['status']}")
print(f"Summary: {result['summary']}")
```

### 5. Agent Communication Changes

**Before (Custom Implementation):**
```python
# Manual message routing
message = AgentMessage(
    sender_id="coordinator",
    receiver_id="test-planner-001",
    message_type=MessageType.TASK_REQUEST,
    payload={"task_data": data}
)

response = await coordinator._route_message(message)
```

**After (LangGraph Implementation):**
```python
# Automatic message handling through state
state["messages"].append(HumanMessage(content="Start test planning"))
state["code_changes"] = input_data

# Messages are automatically managed by LangGraph
```

### 6. State Management Changes

**Before (Custom Implementation):**
```python
# Manual state tracking
class WorkflowExecution:
    execution_id: str
    current_step: int
    results: Dict[str, Any]
    errors: List[str]
    # ... manual state management
```

**After (LangGraph Implementation):**
```python
# Automatic state management with TypedDict
class TestAutomationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    test_scenarios: List[TestScenario]
    test_results: List[TestResult]
    failure_clusters: List[FailureCluster]
    learning_insights: Dict[str, Any]
    # ... automatic state management
```

### 7. Error Handling Changes

**Before (Custom Implementation):**
```python
# Manual error handling and escalation
try:
    result = await agent.process_task(task_data)
except Exception as e:
    await coordinator._handle_escalation(EscalationLevel.HIGH, agent_id, str(e))
```

**After (LangGraph Implementation):**
```python
# Automatic error handling through graph nodes
workflow.add_node("error_handling", self._error_handling_node)
workflow.add_node("escalation", self._escalation_node)

# Automatic routing to error handling nodes
workflow.add_conditional_edges(
    "test_planner",
    self._route_after_planning,
    {"error": "error_handling", "escalate": "escalation"}
)
```

### 8. API Changes

**Before (Custom Implementation):**
```python
# Custom API endpoints
@app.post("/agents/{agent_id}/tasks")
async def execute_agent_task(agent_id: str, task_request: TaskRequest):
    # Manual task execution and response handling
```

**After (LangGraph Implementation):**
```python
# LangGraph-based API endpoints
@app.post("/workflows/execute")
async def execute_workflow(workflow_request: WorkflowRequest):
    # Automatic workflow execution with state management
    result = await workflow.execute_workflow(
        workflow_request.workflow_id,
        workflow_request.input_data
    )
    return WorkflowResponse(**result)
```

## Code Examples

### Complete Migration Example

**Before (Custom Implementation):**
```python
from src.agents.multi_agent_api import app

# Start custom API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**After (LangGraph Implementation):**
```python
from src.langgraph_agents.api import app

# Start LangGraph API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Workflow Execution Example

**Before (Custom Implementation):**
```python
# Manual workflow execution
async def run_e2e_test():
    coordinator = AgentCoordinator()
    await coordinator.start_coordination()
    
    execution_id = await coordinator.start_workflow(
        "e2e-test-workflow",
        {
            "change_type": "feature_implementation",
            "diff_content": "+def new_feature(): ..."
        }
    )
    
    # Manual monitoring
    while execution_id in coordinator.active_workflows:
        execution = coordinator.active_workflows[execution_id]
        if execution.status in ["completed", "failed"]:
            break
        await asyncio.sleep(10)
    
    return coordinator.active_workflows[execution_id]
```

**After (LangGraph Implementation):**
```python
# Automatic workflow execution
async def run_e2e_test():
    workflow = TestAutomationWorkflow()
    
    result = await workflow.execute_workflow(
        "e2e-test-workflow",
        {
            "change_type": "feature_implementation", 
            "diff_content": "+def new_feature(): ..."
        }
    )
    
    return result  # Complete result with summary, errors, etc.
```

### Agent Implementation Example

**Before (Custom Implementation):**
```python
class TestPlannerAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Test Planner Agent", capabilities)
        self.message_handlers = {}
        self.active_tasks = {}
    
    async def receive_message(self, message: AgentMessage):
        # Manual message processing
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
    
    async def process_task(self, task_data: Dict[str, Any]):
        # Manual task processing
        pass
```

**After (LangGraph Implementation):**
```python
class TestPlannerAgent:
    def __init__(self, agent_id: str = "test-planner"):
        self.agent_id = agent_id
        self.name = "Test Planner Agent"
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        # No manual message handling needed
    
    async def plan_tests_for_change(self, state: TestAutomationState) -> TestAutomationState:
        # Automatic state management
        state["messages"].append(SystemMessage(content="Starting test planning"))
        
        # Use LLM for test planning
        result = await self.planning_chain.ainvoke({
            "change_type": state["code_changes"]["change_type"],
            "diff_content": state["code_changes"]["diff_content"]
        })
        
        # Update state automatically
        for scenario_data in result["scenarios"]:
            scenario = TestScenario(**scenario_data)
            state["test_scenarios"].append(scenario)
        
        return state
```

## Migration Checklist

### Phase 1: Setup
- [ ] Update dependencies in `requirements.txt`
- [ ] Install new LangGraph dependencies
- [ ] Backup existing custom implementation
- [ ] Set up new LangGraph project structure

### Phase 2: Core Migration
- [ ] Migrate state management to TypedDict
- [ ] Convert agents to LangGraph format
- [ ] Implement workflow graph with nodes and edges
- [ ] Add conditional routing logic
- [ ] Implement error handling and escalation nodes

### Phase 3: API Migration
- [ ] Update API endpoints to use LangGraph workflow
- [ ] Migrate request/response models
- [ ] Update documentation and examples
- [ ] Test API compatibility

### Phase 4: Testing and Validation
- [ ] Run side-by-side comparison tests
- [ ] Validate workflow execution results
- [ ] Test error handling and recovery
- [ ] Performance testing and optimization
- [ ] User acceptance testing

### Phase 5: Deployment
- [ ] Deploy LangGraph implementation
- [ ] Monitor system performance
- [ ] Gradually migrate traffic
- [ ] Deprecate custom implementation
- [ ] Update documentation and training

## Testing the Migration

### 1. Side-by-Side Testing

Run both implementations with the same input data:

```python
# Test custom implementation
custom_result = await custom_workflow.execute(input_data)

# Test LangGraph implementation  
langgraph_result = await langgraph_workflow.execute_workflow(
    "e2e-test-workflow", input_data
)

# Compare results
assert custom_result["scenarios"] == langgraph_result["results"]["test_scenarios"]
assert custom_result["success_rate"] == langgraph_result["summary"]["success_rate"]
```

### 2. Performance Testing

```python
import time

# Benchmark custom implementation
start_time = time.time()
custom_result = await custom_workflow.execute(input_data)
custom_duration = time.time() - start_time

# Benchmark LangGraph implementation
start_time = time.time()
langgraph_result = await langgraph_workflow.execute_workflow(
    "e2e-test-workflow", input_data
)
langgraph_duration = time.time() - start_time

print(f"Custom: {custom_duration}s, LangGraph: {langgraph_duration}s")
```

### 3. Error Handling Testing

```python
# Test error scenarios
error_input = {"invalid": "data"}

try:
    custom_result = await custom_workflow.execute(error_input)
except Exception as e:
    print(f"Custom error: {e}")

try:
    langgraph_result = await langgraph_workflow.execute_workflow(
        "e2e-test-workflow", error_input
    )
except Exception as e:
    print(f"LangGraph error: {e}")
```

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**: Switch back to custom implementation
2. **Gradual Migration**: Migrate one workflow at a time
3. **Feature Flags**: Use feature flags to toggle between implementations
4. **Monitoring**: Monitor both systems during transition period

## Benefits After Migration

1. **Better Debugging**: Rich observability with LangGraph's debugging tools
2. **Automatic Persistence**: Built-in checkpointing for workflow recovery
3. **Community Support**: Access to LangGraph community and resources
4. **Future-Proof**: Industry-standard framework with active development
5. **Enhanced Features**: Access to new LangGraph features as they're released

## Support and Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangGraph Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Community Support**: LangChain Discord and GitHub discussions
- **Migration Issues**: Create issues in the project repository

## Conclusion

The migration to LangGraph provides significant benefits in terms of maintainability, debugging, and community support. While the migration requires some effort, the long-term benefits make it worthwhile for production systems.

The LangGraph implementation provides:
- Industry-standard multi-agent orchestration
- Better state management and persistence
- Enhanced error handling and recovery
- Rich debugging and observability tools
- Active community support and development

Follow this guide step-by-step to ensure a smooth migration from the custom implementation to LangGraph.
