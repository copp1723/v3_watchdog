# Insight Engine System

This document provides a comprehensive overview of the Watchdog AI Insight Engine, including its architecture, components, and how to extend it with new metrics or anomaly checks.

## System Architecture

The Insight Engine follows a modular pipeline architecture that processes raw data into actionable insights through four main stages:

```
┌───────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
│           │     │          │     │          │     │           │
│ Contracts │ ──> │   Core   │ ──> │ Renderer │ ──> │    UI     │
│           │     │  Logic   │     │          │     │           │
└───────────┘     └──────────┘     └──────────┘     └───────────┘
```

### 1. Contracts

The Contracts layer defines the schema and validation rules for insights. It ensures that all insights adhere to a consistent structure and meet business requirements. This layer is implemented in `src/watchdog_ai/insights/contracts.py`.

Key components:
- `InsightContract`: Defines the expected input and output structure for insights
- `InsightMetrics`: Specifies metrics included in insight output
- `InsightContractEnforcer`: Validates insights against contract requirements

### 2. Core Logic

The Core Logic layer contains the business logic for generating insights. It processes input data, applies statistical methods, and produces structured insight objects. The main implementation is in `src/watchdog_ai/insights/insight_functions.py`.

Key components:
- `InsightFunctions`: Collection of functions for generating insights from data
- Data transformation utilities
- Anomaly detection algorithms
- Statistical analysis functions

### 3. Renderer

The Renderer layer formats insights for display. It transforms the structured insight objects into visual components, preparing them for presentation in the UI. This layer is implemented in `src/watchdog_ai/ui/components/sales_report_renderer.py`.

Key components:
- `render_insight_block`: Renders a complete insight block
- `render_metric_group`: Displays groups of metrics
- `render_breakdown_table`: Shows tabular data
- `render_chart`: Visualizes data in charts

### 4. UI

The UI layer presents the insights to users. It handles user interactions, layout, and styling. This layer includes both the visual presentation and interaction handling.

Key components:
- `InsightOutputFormatter`: Formats responses for consistent display
- `render_insight_card`: Renders an insight as an interactive card in the UI
- Interactive elements for user feedback and actions

## Adding New Metrics or Anomaly Checks

### Adding a New Metric

To add a new metric calculation to the Insight Engine, follow these steps:

1. **Define the metric function in `insight_functions.py`**:

```python
def compute_new_metric(self, df):
    """
    Compute a new metric from the dataframe.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Computed metric value or DataFrame
    """
    # Find relevant columns
    metric_col = self.find_column(df, ["MetricName", "metric_name"])
    if not metric_col:
        self.logger.warning("Required columns not found for new metric calculation")
        return None
    
    try:
        # Perform calculation
        result = df[metric_col].mean()  # Or any other calculation
        return result
    except Exception as e:
        self.logger.error(f"Error in compute_new_metric: {e}")
        return None
```

2. **Add the metric to export list at the bottom of the file**:

```python
# For backward compatibility
compute_new_metric = InsightFunctions().compute_new_metric
```

3. **Update contract validation (if necessary)**:

If your metric requires specific validation rules, add them to the `InsightContractEnforcer`:

```python
def validate_new_metric(df):
    """Validate that the data can support the new metric."""
    required_cols = ["MetricName", "SupportingColumn"]
    return all(col in df.columns for col in required_cols)

validator.add_rule(ValidationRule(
    name="new_metric_validation",
    description="Validates data for new metric calculation",
    severity="warning",
    validation_fn=validate_new_metric,
    error_message="Data may not support new metric calculation"
))
```

### Adding a New Anomaly Check

To add a new anomaly detection check:

1. **Define the anomaly detection function**:

```python
def detect_new_anomaly_type(self, df):
    """
    Detect a new type of anomaly in the data.
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of anomalies detected
    """
    anomalies = []
    try:
        # Find relevant columns
        target_col = self.find_column(df, ["TargetColumn", "target_column"])
        if not target_col:
            return anomalies
            
        # Calculate reference value (e.g., mean, median)
        reference = df[target_col].mean()
        
        # Identify anomalies (e.g., values > 3 standard deviations)
        std_dev = df[target_col].std()
        threshold = reference + (3 * std_dev)
        
        # Find anomalous records
        anomaly_records = df[df[target_col] > threshold]
        
        # Format anomalies
        for _, row in anomaly_records.iterrows():
            anomalies.append({
                "anomaly_type": "High Value Anomaly",
                "column": target_col,
                "value": row[target_col],
                "description": f"Value exceeds threshold of {threshold:.2f}",
                "is_problem": True,
                "recommendations": [
                    f"Investigate high {target_col} value of {row[target_col]}"
                ]
            })
            
        return anomalies
    except Exception as e:
        self.logger.error(f"Error in detect_new_anomaly_type: {e}")
        return anomalies
```

2. **Add anomaly check to the pipeline**:

In the integration module (e.g., `insights_integration.py`), add your new anomaly check to the processing flow:

```python
# Add to the anomaly processing section
anomaly_results = []
anomaly_results.extend(insight_functions.detect_new_anomaly_type(df))

# Process anomalies
if anomaly_results:
    for anomaly in anomaly_results:
        insight = anomaly_to_insight(anomaly)
        if anomaly.get("is_problem", True):
            problems.append(insight)
        else:
            opportunities.append(insight)
```

3. **Create a renderer for the new anomaly type (if needed)**:

If your anomaly requires special visualization, add a renderer function in `sales_report_renderer.py`:

```python
def render_new_anomaly_type(anomaly_data):
    """
    Render the new anomaly type with specialized visualization.
    
    Args:
        anomaly_data: The anomaly data to render
    """
    st.warning(f"**{anomaly_data['anomaly_type']}** detected")
    st.markdown(f"**Value:** {anomaly_data['value']}")
    st.markdown(f"**Description:** {anomaly_data['description']}")
    
    # Add specialized visualization
    if 'chart_data' in anomaly_data:
        st.line_chart(anomaly_data['chart_data'])
```

## Core Component Code Examples

### 1. Contract Definition (from `contracts.py`)

```python
class InsightContract(BaseModel):
    """Contract defining expected insight input and output."""
    # Metadata
    insight_id: str = Field(..., description="Unique identifier for the insight")
    insight_type: str = Field(..., description="Type of insight being generated")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Version of the insight contract")
    
    # Input Requirements
    required_columns: List[str] = Field(..., description="Required columns in input data")
    min_rows: int = Field(default=10, description="Minimum number of rows required")
    data_types: Dict[str, str] = Field(..., description="Expected data types for columns")
    
    # Output Requirements
    metrics: InsightMetrics
    summary: str = Field(..., min_length=10, max_length=1000)
    key_findings: List[str] = Field(..., min_items=1)
    recommendations: List[str] = Field(..., min_items=1)
    visualization: Optional[ChartSpec] = None
    
    # Execution Context
    execution_time_ms: float = Field(..., description="Time taken to generate insight")
    cache_hit: bool = Field(default=False, description="Whether result was cached")
    error: Optional[str] = None
```

### 2. Insight Function Example (from `insight_functions.py`)

```python
def compute_gross_profit_summary(self, df):
    """
    Compute a summary of gross profit.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with gross profit metrics
    """
    gross_profit_col = self.find_column(df, ["GrossProfit", "gross_profit", "Gross Profit"])
    if not gross_profit_col:
        return {"total_gross_profit": None, "average_gross_profit": None}
    total = df[gross_profit_col].sum()
    avg = df[gross_profit_col].mean()
    return {"total_gross_profit": total, "average_gross_profit": avg}
```

### 3. Renderer Example (from `sales_report_renderer.py`)

```python
def render_insight_block(insight_data: Dict[str, Any] = None):
    """
    Render a complete insight block with all components.
    
    Args:
        insight_data: Complete insight data dictionary
    """
    if not insight_data:
        st.warning("No insight data available")
        return
    
    # Main summary
    st.markdown(f"### {insight_data.get('summary', 'Insight Analysis')}")
    
    # Render metrics if present
    if primary_metrics := insight_data.get('primary_metrics'):
        render_metric_group(primary_metrics)
    
    # Render chart if present
    if chart_data := insight_data.get('chart_data'):
        render_chart(chart_data)
    
    # Render breakdown if present
    if breakdown := insight_data.get('performance_breakdown'):
        st.markdown("#### Performance Breakdown")
        render_breakdown_table(
            breakdown,
            key_column=next(iter(breakdown[0].keys())) if breakdown else None
        )
    
    # Render action items if present
    if actions := insight_data.get('actionable_flags'):
        render_action_items(actions)
    
    # Show confidence level if present
    if confidence := insight_data.get('confidence'):
        st.markdown(
            f"*Confidence Level: {confidence.title()}*",
            help="Indicates the reliability of the insight based on data quality"
        )
```

## Best Practices

When extending the Insight Engine, follow these best practices:

1. **Maintain Contract Compatibility**: Ensure new insights adhere to the existing contract structure or update the contracts appropriately.

2. **Error Handling**: Implement robust error handling in all insight functions to prevent pipeline failures.

3. **Logging**: Use the logger to document processing steps and errors for easier debugging.

4. **Performance**: Consider performance impact for large datasets; implement optimization where needed.

5. **Testing**: Write unit tests for new metrics and anomaly checks to validate their correctness.

6. **Documentation**: Update this document when adding significant new functionality.

## Common Troubleshooting

1. **Missing Data Errors**: Check that required columns exist and have the expected data types.

2. **Visualization Issues**: Ensure chart data is correctly formatted and within size limits.

3. **Performance Problems**: Add profiling to identify bottlenecks in complex calculations.

4. **Contract Validation Failures**: Review the error details to identify which contract requirement is not being met.

