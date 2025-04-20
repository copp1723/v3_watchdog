# AgentOps Integration Plan for Watchdog AI

## Overview
Integrate AgentOps monitoring into the Watchdog AI platform to track LLM interactions, performance metrics, and errors.

## Implementation Steps

### 1. Add Dependencies
- Add agentops to requirements.txt
- Add AgentOps configuration to environment variables

### 2. Create AgentOps Configuration Module
- Create src/utils/agentops_config.py
- Implement initialization and handler management
- Add logging and error handling

### 3. Update LLM Engine
- Add AgentOps session tracking
- Integrate monitoring for LLM calls
- Preserve existing error handling
- Add custom tags for query types

### 4. Update Intent Manager
- Add AgentOps tracking for intent detection
- Monitor intent matching performance
- Track unsupported queries

### 5. Add Unit Tests
- Test AgentOps initialization
- Test session tracking
- Test error handling
- Test custom tag propagation

### 6. Update Documentation
- Add AgentOps setup instructions
- Document monitoring capabilities
- Add troubleshooting guide

## Technical Details

### AgentOps Configuration
```python
# src/utils/agentops_config.py
import os
import logging
import agentops
from agentops import AgentOpsLangchainCallbackHandler

def init_agentops():
    api_key = os.getenv("AGENTOPS_API_KEY")
    if api_key:
        agentops.init(api_key, tags=["watchdog-ai"])
        return True
    return False

def get_handler(session_id=None, query_type=None):
    tags = {"session_id": session_id} if session_id else {}
    if query_type:
        tags["query_type"] = query_type
    return AgentOpsLangchainCallbackHandler(tags=tags)
```

### LLM Engine Integration
```python
@agentops.session
def generate_insight(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    # Existing implementation with AgentOps monitoring
```

### Intent Manager Integration
```python
def detect_intent(self, query: str) -> str:
    with agentops.track(tags={"operation": "intent_detection"}):
        # Existing implementation
```

### Unit Tests
```python
def test_agentops_integration():
    # Test initialization
    # Test session tracking
    # Test error handling
```

## Success Criteria
1. AgentOps successfully tracks all LLM interactions
2. Custom tags provide query context
3. Error handling remains robust
4. No performance degradation
5. Tests pass with high coverage

## Rollback Plan
1. Remove AgentOps imports
2. Restore original LLM engine
3. Remove environment variables
4. Update documentation