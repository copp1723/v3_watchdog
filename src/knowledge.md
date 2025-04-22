# Watchdog AI Knowledge Base

## Metrics & Debug Dashboard

### Metrics Logger
- Use JSON format for structured logging
- Implement log rotation (10MB files, keep 5 backups)
- Track execution time, memory usage, token counts, cache hits
- Use trace IDs to link related operations
- Store logs in logs/metrics directory

### Debug Dashboard
- Three main sections: Execution Metrics, Trace Analysis, Cache Statistics
- Mobile-responsive layout with collapsible sections
- Use plotly for interactive visualizations
- Cache hit rate target: >80%
- Response time target: <1000ms
- Memory usage target: <500MB

### Traceability System
- Use watchdog_ai.insights.traceability.TraceabilityEngine for comprehensive tracing
- Store traces with steps, input/output data, and metrics
- Support trace versioning and comparison
- Persist traces to disk in JSON format
- Include query context and metadata

### Performance Thresholds
- Log slow operations (>5s) as warnings
- Alert on high error rates (>5%)
- Monitor cache hit rates hourly
- Track memory usage trends

### Best Practices
- Use trace IDs for all operations
- Record both success and error metrics
- Implement log rotation
- Keep UI responsive on mobile
- Add comprehensive test coverage

## DataFrame Display in Streamlit
- Most reliable approaches for displaying DataFrames with problematic types:
  1. Use st_aggrid package for interactive tables (best UX)
  2. Convert to records and use st.json() (most reliable)
  3. Use st.table() with pre-converted string data (simple but works)
  4. Use st.markdown() with DataFrame.to_markdown() (good for static data)
- Always convert data to strings at load time, not display time
- Use csv.reader for initial load to bypass pandas type inference
- Handle null values and complex types (dict/list) explicitly
- Provide fallback display options for each visualization

## Project Structure
- Main application entry: src/app.py
- UI components in watchdog_ai/ui/components/
- Page layouts in watchdog_ai/ui/pages/
- Core logic in watchdog_ai/

## Best Practices
- Use session state for data persistence
- Handle file uploads with proper validation
- Provide clear user feedback for data processing issues
- Include fallback display options for visualization errors