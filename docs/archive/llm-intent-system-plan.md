# LLM-Driven Intent System Implementation Plan

## Overview
Transform the current hardcoded intent system into a flexible LLM-driven approach that can handle a wider range of queries while maintaining robustness and performance.

## Core Components

### 1. Enhanced Prompt System
- Create structured prompts that guide the LLM to output standardized JSON
- Include examples and strict formatting requirements
- Support for column mapping and data type inference

### 2. Intent Processing Pipeline
1. Query Understanding
   - LLM analyzes query and available columns
   - Outputs structured JSON with intent classification
   - Handles column mapping and metric identification

2. Data Processing
   - Generic processors for common operations (grouping, aggregation)
   - Robust error handling and validation
   - Support for multiple metrics and dimensions

3. Response Generation
   - Consistent JSON structure for all responses
   - Support for visualizations and recommendations
   - Error handling and fallback responses

## Implementation Steps

### Phase 1: Core Infrastructure

1. Create new prompt template:
```
src/insights/prompts/intent_detection.tpl:
You are an analytics assistant for a car dealership. Given a user query and these columns:
{available_columns}

Return ONLY a JSON object with these fields:
{
  "intent": "groupby" | "metric" | "comparison" | "trend" | "fallback",
  "metrics": [{"name": string, "aggregation": "sum"|"avg"|"count"|"max"|"min"}],
  "dimensions": [{"name": string, "type": "category"|"time"|"numeric"}],
  "filters": [{"column": string, "operator": "="|">"|"<", "value": any}],
  "sort": {"column": string, "direction": "asc"|"desc"},
  "limit": number
}

Example Query: "Which lead source produced the most sales?"
Example Response:
{
  "intent": "groupby",
  "metrics": [{"name": "count", "aggregation": "count"}],
  "dimensions": [{"name": "lead_source", "type": "category"}],
  "filters": [],
  "sort": {"column": "count", "direction": "desc"},
  "limit": 5
}
```

2. Create Pydantic models for validation:
```python
# src/insights/models.py
from pydantic import BaseModel
from typing import List, Optional, Literal

class Metric(BaseModel):
    name: str
    aggregation: Literal["sum", "avg", "count", "max", "min"]

class Dimension(BaseModel):
    name: str
    type: Literal["category", "time", "numeric"]

class Filter(BaseModel):
    column: str
    operator: Literal["=", ">", "<"]
    value: Any

class Sort(BaseModel):
    column: str
    direction: Literal["asc", "desc"]

class IntentSchema(BaseModel):
    intent: Literal["groupby", "metric", "comparison", "trend", "fallback"]
    metrics: List[Metric]
    dimensions: List[Dimension]
    filters: List[Filter]
    sort: Optional[Sort]
    limit: Optional[int]
```

### Phase 2: Intent Processing

1. Update ConversationManager:
```python
# src/insight_conversation.py
from .models import IntentSchema
import pandas as pd

class ConversationManager:
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        # Generate prompt with available columns
        prompt = self._generate_prompt(query, df.columns.tolist())
        
        # Get LLM response
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse and validate response
        try:
            intent_data = IntentSchema.parse_raw(response.choices[0].message.content)
            return self._process_intent(intent_data, df)
        except ValidationError as e:
            return self._generate_error_response(str(e))

    def _process_intent(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        if intent.intent == "groupby":
            return self._handle_groupby(intent, df)
        elif intent.intent == "metric":
            return self._handle_metric(intent, df)
        # ... handle other intents
        
        return self._generate_fallback_response()

    def _handle_groupby(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Apply filters
            for filter in intent.filters:
                df = df[df[filter.column].apply(lambda x: eval(f"x {filter.operator} {filter.value}"))]
            
            # Group and aggregate
            aggs = {metric.name: metric.aggregation for metric in intent.metrics}
            dimensions = [dim.name for dim in intent.dimensions]
            
            result = df.groupby(dimensions).agg(aggs).reset_index()
            
            # Sort if specified
            if intent.sort:
                result = result.sort_values(
                    intent.sort.column,
                    ascending=intent.sort.direction == "asc"
                )
            
            # Apply limit
            if intent.limit:
                result = result.head(intent.limit)
            
            # Format response
            return {
                "summary": self._generate_summary(intent, result),
                "metrics": self._format_metrics(result),
                "breakdown": result.to_dict("records"),
                "chart_data": self._format_chart_data(result),
                "recommendations": self._generate_recommendations(intent, result),
                "confidence": "high"
            }
            
        except Exception as e:
            return self._generate_error_response(str(e))
```

### Phase 3: Response Generation

1. Create response formatter:
```python
# src/insights/formatters.py
class ResponseFormatter:
    def format_groupby_response(self, result: pd.DataFrame, intent: IntentSchema) -> Dict[str, Any]:
        # Get primary metric
        metric = intent.metrics[0]
        dimension = intent.dimensions[0]
        
        # Get top result
        top_row = result.iloc[0]
        
        return {
            "summary": f"{top_row[dimension.name]} leads with {top_row[metric.name]} {metric.name}s",
            "metrics": {
                "top_performer": top_row[dimension.name],
                "value": self._format_value(top_row[metric.name], metric),
                "context": self._generate_context(result, metric)
            },
            "breakdown": result.to_dict("records"),
            "chart_data": self._format_chart_data(result, dimension, metric),
            "recommendations": self._generate_recommendations(result, intent),
            "confidence": "high"
        }

    def _format_value(self, value: Any, metric: Metric) -> str:
        if "gross" in metric.name.lower():
            return f"${value:,.2f}"
        return f"{value:,}"

    def _generate_context(self, result: pd.DataFrame, metric: Metric) -> str:
        total = result[metric.name].sum()
        return f"Out of {self._format_value(total, metric)} total {metric.name}s"
```

### Phase 4: Testing

1. Create comprehensive tests:
```python
# tests/test_intent_processing.py
def test_lead_source_query():
    manager = ConversationManager()
    df = pd.DataFrame({
        "lead_source": ["CarGurus", "AutoTrader"],
        "gross_profit": [1000, 2000]
    })
    
    result = manager.process_query(
        "Which lead source produced the most sales?",
        df
    )
    
    assert result["summary"] == "AutoTrader leads with 2000 gross_profit"
    assert result["metrics"]["top_performer"] == "AutoTrader"
    assert result["confidence"] == "high"

def test_invalid_column():
    manager = ConversationManager()
    df = pd.DataFrame({"a": [1]})
    
    result = manager.process_query(
        "Show me sales by invalid_column",
        df
    )
    
    assert "error" in result
    assert "column not found" in result["error"].lower()
```

## Migration Strategy

1. Phase out existing intent classes gradually:
   - Keep them as fallback while testing new system
   - Monitor error rates and performance
   - Migrate one intent type at a time

2. Update UI components to handle new response format:
   - Add visualization support
   - Improve error displays
   - Add feedback collection

3. Add monitoring and logging:
   - Track LLM response quality
   - Monitor performance metrics
   - Collect user feedback

## Success Metrics

1. Query Success Rate:
   - % of queries successfully processed
   - % of queries requiring fallback
   - Error rate by type

2. Performance:
   - Response time (p50, p95, p99)
   - Memory usage
   - LLM token usage

3. User Satisfaction:
   - Feedback scores
   - Error report rate
   - Query refinement rate

## Rollout Plan

1. Development (Week 1):
   - Implement core infrastructure
   - Create basic intent processing
   - Add initial tests

2. Testing (Week 2):
   - Add comprehensive tests
   - Test with real queries
   - Measure performance

3. Gradual Rollout (Week 3):
   - Roll out to 10% of queries
   - Monitor errors
   - Collect feedback

4. Full Deployment (Week 4):
   - Scale to 100%
   - Remove old intent system
   - Document new system