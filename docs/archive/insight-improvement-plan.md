# Insight Generation Improvement Plan

## 1. Enhanced Intent Recognition

### New Intent Classes
```python
class CountMetricIntent:
    """Intent for counting/aggregating specific data points."""
    def matches(self, prompt: str) -> bool:
        return any(term in prompt.lower() for term in [
            "how many", "count of", "number of", "total number"
        ])
    
    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        # Direct calculation logic for counts
        # Return structured result with exact numbers

class NegativeProfitIntent:
    """Intent for analyzing negative profit transactions."""
    def matches(self, prompt: str) -> bool:
        return "negative" in prompt.lower() and any(
            term in prompt.lower() for term in ["profit", "gross", "margin"]
        )
    
    def analyze(self, df: pd.DataFrame, prompt: str) -> InsightResult:
        # Calculate negative profit metrics
        # Return detailed analysis with counts and patterns
```

### Improved Intent Matching
- Add fuzzy matching for terms
- Support compound queries
- Better handling of business terminology

## 2. Direct Data Analysis

### Core Analysis Functions
- Add specific calculation functions for common metrics
- Support for time-based aggregations
- Enhanced pattern detection

### Data Validation
- Validate numeric columns before analysis
- Handle missing/invalid data gracefully
- Support multiple column name formats

## 3. Response Enhancement

### Structured Output
- Include exact counts/numbers
- Add relevant context
- Provide trend information

### Visualization Support
- Add charts for numeric insights
- Show distributions
- Highlight anomalies

## 4. Testing & Validation

### Test Cases
- Add specific test cases for numeric queries
- Validate accuracy of calculations
- Test edge cases and error handling

### Quality Checks
- Ensure responses include specific numbers
- Validate business logic
- Check visualization accuracy

## Implementation Steps

1. Add new intent classes for specific metrics
2. Enhance the intent matching system
3. Add direct calculation functions
4. Improve response formatting
5. Add comprehensive tests
6. Update documentation

## Success Criteria

1. Accurate response to "how many" questions
2. Specific numbers in insights
3. Relevant context and visualization
4. Proper error handling
5. Comprehensive test coverage