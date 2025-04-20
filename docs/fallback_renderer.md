# Fallback Insight Renderer

The Fallback Insight Renderer is a component that handles error cases in the insight generation pipeline and provides user-friendly error messages. It ensures a positive user experience even when errors occur by presenting clear, actionable messages and logging detailed error information for debugging and improvement.

## Features

- Unified error taxonomy covering common failure scenarios
- User-friendly error messages with actionable suggestions
- Comprehensive error logging for debugging and improvement
- Integration with the insight flow pipeline
- Support for various error types (data validation, code generation, business logic, etc.)
- Performance monitoring for render operations
- Error rate tracking for system improvement

## Error Taxonomy

The fallback renderer uses a unified error taxonomy defined in the `ErrorCode` enum:

### Schema and Data Validation Errors
- `COLUMN_MISSING`: Required column not found in dataset
- `INVALID_DATA_TYPE`: Column contains data in unexpected format
- `DATA_CONVERSION_ERROR`: Failed to convert data values
- `NO_MATCHING_DATA`: No data found matching criteria

### LLM and Code Generation Errors
- `INVALID_LLM_CODE`: Generated code is invalid
- `CODE_EXECUTION_ERROR`: Error during code execution
- `MEMORY_ERROR`: Memory limit exceeded
- `TIMEOUT_ERROR`: Analysis timeout

### Business Logic Errors
- `INVALID_BUSINESS_RULE`: Violation of business rules
- `INSUFFICIENT_DATA`: Not enough data for analysis
- `AMBIGUOUS_RESULT`: Unclear or ambiguous results

### System Errors
- `SYSTEM_ERROR`: Unexpected system error
- `CONFIGURATION_ERROR`: System configuration issue
- `INTEGRATION_ERROR`: Service integration failure

## Usage

### Basic Usage

```python
from fallback_renderer import FallbackRenderer, ErrorCode, ErrorContext
from datetime import datetime

# Create a fallback renderer instance
renderer = FallbackRenderer()

# Create an error context
error_context = ErrorContext(
    error_code=ErrorCode.COLUMN_MISSING,
    error_message="Column 'sales' not found in dataset",
    details={"column_name": "sales"},
    timestamp=datetime.now().isoformat(),
    user_query="Show me sales trends",
    affected_columns=["sales"]
)

# Render the error message
result = renderer.render_fallback(error_context)
```

### Performance Monitoring

The fallback renderer includes built-in performance monitoring:

```python
# Render an error message
result = renderer.render_fallback(error_context)

# Access the render time
render_time = result["render_time"]
print(f"Error message rendered in {render_time:.3f} seconds")
```

Performance metrics are automatically logged:

```python
# In your logging configuration
logger.info(
    f"Fallback render time: {render_time:.3f}s",
    extra={
        "metric": "fallback_render_time",
        "value": render_time,
        "error_code": error_context.error_code.value
    }
)
```

### Error Rate Tracking

The fallback renderer tracks error rates for monitoring and improvement:

```python
# Get current error statistics
error_stats = renderer.get_error_stats()
print(f"Error counts: {error_stats}")

# Example output:
# {
#     "column_missing": 5,
#     "invalid_data_type": 2,
#     "no_matching_data": 3,
#     "system_error": 1
# }
```

This data can be used to:
- Identify the most common error types
- Prioritize improvements
- Monitor error trends over time
- Set up alerts for unusual error patterns

### Integration with Insight Flow

The fallback renderer is integrated with the insight flow pipeline in `insight_flow.py`. It handles various error scenarios:

1. Explicit error responses from the LLM or code generation
2. Missing or invalid insight content
3. Unexpected exceptions during insight generation

Example error handling in the insight flow:

```python
try:
    # Process insight response
    if response.get("type") == "error":
        error_context = ErrorContext(
            error_code=ErrorCode(response.get("error_code", "system_error")),
            error_message=response.get("error_message", "Unknown error occurred"),
            details=response.get("error_details", {}),
            timestamp=datetime.now().isoformat()
        )
        return fallback_renderer.render_fallback(error_context)
except Exception as e:
    # Handle unexpected errors
    error_context = ErrorContext(
        error_code=ErrorCode.SYSTEM_ERROR,
        error_message=str(e),
        details={"error_details": str(e)},
        timestamp=datetime.now().isoformat()
    )
    return fallback_renderer.render_fallback(error_context)
```

## Error Message Templates

Each error type has a template with the following components:

- `title`: Short, descriptive title
- `message`: User-friendly explanation of the error
- `action`: Suggested action to resolve the error
- `technical_details`: Technical information for logging

Example template for column missing error:

```python
{
    "title": "Required Data Not Found",
    "message": "We couldn't find the {column_name} column in your data. This information is needed to generate the insight.",
    "action": "Please check your data file and ensure it contains the {column_name} column.",
    "technical_details": "Missing column: {column_name}"
}
```

## Logging

The fallback renderer logs detailed error information for debugging and improvement:

```python
log_data = {
    "error_code": error_context.error_code.value,
    "error_message": error_context.error_message,
    "technical_details": technical_details,
    "timestamp": error_context.timestamp,
    "user_query": error_context.user_query,
    "affected_columns": error_context.affected_columns,
    "stack_trace": error_context.stack_trace
}
```

## Testing

The fallback renderer includes comprehensive unit tests and integration tests:

### Unit Tests

Run the basic unit tests:

```bash
pytest src/tests/test_fallback_renderer.py
```

### Integration Tests

Run the integration tests with real dealership data:

```bash
pytest src/tests/test_fallback_renderer_integration.py
```

The integration tests cover:
- Real-world data scenarios
- Complex error conditions
- Performance monitoring
- Error rate tracking

## Troubleshooting

Common issues and solutions:

1. **Missing Error Template Variables**
   - Ensure all required variables are provided in the error details
   - Use default values for optional variables

2. **Unknown Error Codes**
   - The renderer falls back to the system error template
   - Add new error codes to the `ErrorCode` enum

3. **Integration Issues**
   - Check error response format matches expected structure
   - Verify error code values match the enum

## Contributing

When adding new error types:

1. Add the error code to the `ErrorCode` enum
2. Create a template in the `error_templates` dictionary
3. Add test cases in `test_fallback_renderer.py`
4. Add integration tests in `test_fallback_renderer_integration.py`
5. Update this documentation

## Dependencies

- Python 3.7+
- `enum` (standard library)
- `logging` (standard library)
- `dataclasses` (standard library)
- `json` (standard library)
- `time` (standard library)
- `collections` (standard library)
- `threading` (standard library)
- `pandas` (for integration tests)
- `numpy` (for integration tests) 