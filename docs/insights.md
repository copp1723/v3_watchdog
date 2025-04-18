# Watchdog AI Insights Framework

This document provides a comprehensive overview of the Watchdog AI Insights Framework, its components, and instructions for extending it with new insights.

## Core Architecture

The Insights Framework is designed to generate high-value, actionable insights from automotive dealership data. It consists of several key components:

1. **Base Insight Classes** - The foundation for all insights
   - `InsightBase` - Abstract base class with core functionality
   - `ChartableInsight` - Extension with chart generation capabilities

2. **Insight Generation System**
   - `InsightGenerator` - Manages insight creation and dispatching
   - Intent-based natural language understanding for queries
   - Direct insight calculation for specific, predefined types

3. **Executive Insights** - High-value business analysis
   - Monthly Gross Margin vs. Target
   - Lead Conversion Rate by Source
   - Sales Performance Analysis
   - Inventory Aging Anomalies

4. **Visualization** - Chart generation for insights
   - Auto-detection of chart types
   - Standardized data structures
   - Interactive display in the UI

5. **Integration with InsightsDigest** - Unified reporting
   - Converts insights into problems and opportunities
   - Prioritizes by severity and impact
   - Provides recommendations

6. **Adaptive Learning System** - NEW
   - Customizes insights based on user interactions
   - Adjusts thresholds based on dealership-specific patterns
   - Improves relevance through usage feedback

7. **Predictive Forecasting** - COMING SOON
   - Time-series forecasting for sales and inventory metrics
   - Multiple algorithm support (ARIMA, Prophet, LSTM)
   - Anomaly detection in business metrics
   - Confidence intervals for business planning

## Developer Notes: Adaptive Learning System

The adaptive learning system enhances our insights framework by allowing it to adapt to each dealership's unique patterns and preferences. Here are the key components for developers:

### Usage Pattern Tracking

The system tracks:
- Which insights users interact with most frequently
- Which recommendations lead to actual actions
- How users modify default thresholds

```python
# Example of the interaction tracking schema
interaction_record = {
    "user_id": "string",
    "dealership_id": "string",
    "insight_type": "string",
    "action": "viewed|expanded|actioned|dismissed",
    "timestamp": "ISO-8601",
    "session_id": "string",
    "context": {
        "page": "string",
        "referrer": "string"
    },
    "custom_thresholds": {
        "threshold_name": value
    }
}
```

### Threshold Adaptation

The system automatically adjusts thresholds based on historical patterns:

1. **Baseline Calculation**: Each dealership starts with industry standard thresholds
2. **Pattern Detection**: System identifies normal ranges for each metric
3. **Outlier Adjustment**: Thresholds adapt to highlight true anomalies for each store
4. **Seasonality Awareness**: Adjusts for seasonal patterns in the business

Example implementation:

```python
def adapt_threshold(dealership_id, metric_name, base_threshold):
    """
    Adapt a threshold based on dealership-specific patterns.
    
    Args:
        dealership_id: Unique identifier for the dealership
        metric_name: Name of the metric (e.g., "gross_margin", "days_in_inventory")
        base_threshold: The starting threshold value
        
    Returns:
        Adapted threshold value
    """
    # Get historical data for this metric at this dealership
    history = get_metric_history(dealership_id, metric_name)
    
    if len(history) < MIN_HISTORY_POINTS:
        return base_threshold  # Not enough data to adapt
    
    # Calculate mean and standard deviation
    mean_value = np.mean(history)
    std_dev = np.std(history)
    
    # Adjust threshold based on historical patterns
    # Formula can be customized based on metric type
    adapted_threshold = mean_value + (std_dev * SENSITIVITY_FACTOR)
    
    # Ensure threshold is within reasonable bounds
    min_allowed = base_threshold * MIN_THRESHOLD_FACTOR
    max_allowed = base_threshold * MAX_THRESHOLD_FACTOR
    
    return np.clip(adapted_threshold, min_allowed, max_allowed)
```

### Relevance Feedback Loop

The system improves over time through explicit and implicit feedback:

1. **Explicit Feedback**: Users can rate insights or mark them as relevant/irrelevant
2. **Implicit Feedback**: System tracks which insights lead to actions
3. **Weight Adjustment**: Calculation factors are weighted based on feedback
4. **A/B Testing**: System tests variations in insight presentation and thresholds

Integration with insight generation:

```python
class AdaptiveInsight(ChartableInsight):
    """Base class for insights that adapt to user behavior."""
    
    def __init__(self, insight_type):
        super().__init__(insight_type)
        self.feedback_manager = FeedbackManager()
    
    def generate(self, df, dealership_id, **kwargs):
        """Generate insight with adaptive thresholds."""
        # Get base thresholds
        base_thresholds = self.get_base_thresholds()
        
        # Adapt thresholds for this dealership
        adapted_thresholds = {}
        for name, value in base_thresholds.items():
            adapted_thresholds[name] = adapt_threshold(
                dealership_id, 
                f"{self.insight_type}.{name}", 
                value
            )
            
        # Pass adapted thresholds to computation
        result = self.compute_insight(df, thresholds=adapted_thresholds, **kwargs)
        
        # Track generation for feedback loop
        self.feedback_manager.record_generation(
            dealership_id=dealership_id,
            insight_type=self.insight_type,
            thresholds=adapted_thresholds,
            result_summary=self.summarize_result(result)
        )
        
        return result
```

## Developer Notes: Upcoming Predictive Forecasting Module

The forecasting module is currently in development and will provide powerful predictive capabilities. Here's what to expect:

### Architecture

The module is designed with three tiers of forecasting models:

1. **Basic Forecasting** (ARIMA-based)
   - Suitable for stable metrics with clear seasonality
   - Lower computational requirements
   - Faster training and prediction times

2. **Advanced Forecasting** (Prophet-based)
   - Handles multiple seasonality patterns
   - Built-in holiday effects
   - Robust to missing data and outliers

3. **Deep Forecasting** (LSTM-based)
   - Captures complex non-linear patterns
   - Incorporates multiple feature inputs
   - Highest accuracy for complex scenarios

### Integration Points

When integrating with the forecasting module, use these patterns:

```python
# Basic forecast request
forecast_result = forecasting_service.forecast(
    metric_name="monthly_sales",
    historical_data=sales_df,
    forecast_periods=6,  # 6 months forward
    model_type="prophet",  # or "arima", "lstm"
    confidence_interval=0.80
)

# Accessing forecast results
predicted_values = forecast_result.predictions  # DataFrame with dates and values
upper_bounds = forecast_result.upper_bound
lower_bounds = forecast_result.lower_bound
anomalies = forecast_result.detect_anomalies(new_data)
```

### Forecasting APIs

The module will expose these key APIs:

1. **Time-series Forecasting**:
   ```python
   forecast(time_series, periods, model_type)
   ```

2. **Anomaly Detection**:
   ```python
   detect_anomalies(time_series, sensitivity, method)
   ```

3. **What-if Analysis**:
   ```python
   forecast_with_scenario(time_series, scenario_adjustments)
   ```

4. **Model Management**:
   ```python
   train_custom_model(dealership_id, metric, params)
   evaluate_model_accuracy(model_id, test_data)
   ```

5. **Ensemble Forecasting**:
   ```python
   ensemble_forecast(time_series, models=[...])
   ```

### Performance Considerations

- Forecasting is computationally intensive - use the async API for web requests
- Pre-compute common forecasts on a schedule rather than on-demand
- Cache forecast results with appropriate invalidation
- For real-time insights, use the "quick forecast" option with simplified models

### Upcoming Release Timeline

1. **Alpha (Internal)**: Basic forecasting with ARIMA and Prophet - Q3 2023
2. **Beta**: Advanced forecasting with Prophet, initial LSTM support - Q4 2023
3. **Production**: Full forecasting suite with ensemble methods - Q1 2024

## Adding New Insights

To add a new insight to the framework, follow these steps:

### 1. Create a new insight class

Create a new class in `src/insights/insight_functions.py` that inherits from either `InsightBase` or `ChartableInsight`.

```python
from .base_insight import ChartableInsight

class YourNewInsight(ChartableInsight):
    """
    Description of your new insight.
    """
    
    def __init__(self):
        """Initialize the insight."""
        super().__init__("your_insight_type")
    
    def _validate_insight_input(self, df, **kwargs):
        """Validate input data for your insight."""
        # Check for required columns or data characteristics
        # Return dict with "error" key if validation fails
        return {}
    
    def compute_insight(self, df, **kwargs):
        """Compute your insight from the data."""
        # Your insight logic goes here
        # Return a dict with your insight data
        return {
            "key_metrics": {...},
            "insights": [
                {
                    "type": "your_insight_type",
                    "title": "Insight Title",
                    "description": "Insight description"
                }
            ],
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ]
        }
    
    def create_chart_data(self, insight_result, original_df, **kwargs):
        """Create chart data for your insight."""
        # Create a DataFrame suitable for charting
        chart_df = pd.DataFrame(...)
        return chart_df
    
    def create_chart_encoding(self, insight_result, chart_data, **kwargs):
        """Create chart encoding specification."""
        return {
            "chart_type": "line",  # or bar, pie, etc.
            "x": "x_column_name",
            "y": "y_column_name",
            "title": "Chart Title"
        }
```

### 2. Register your insight with the InsightGenerator

Add your insight to the registry in `src/insights/insight_generator.py`:

```python
from .insight_functions import YourNewInsight

class InsightGenerator:
    def __init__(self):
        # Add your insight to the dictionary
        self.insights = {
            "monthly_gross_margin": MonthlyGrossMarginInsight(),
            "lead_conversion_rate": LeadConversionRateInsight(),
            "your_insight_type": YourNewInsight()
        }
```

### 3. Add query patterns for your insight

Update the pattern recognition in the `generate_insight` method:

```python
def generate_insight(self, prompt, df):
    # ...
    lower_prompt = prompt.lower()
    
    # Add pattern for your insight
    if "your_keyword" in lower_prompt and "your_other_keyword" in lower_prompt:
        insight_type = "your_insight_type"
    # ...
```

### 4. Add integration with InsightsDigest

Update the `executive_insight_to_digest_entry` function in `src/insights_integration.py` to handle your new insight type:

```python
def executive_insight_to_digest_entry(insight_result, insight_type):
    # ...
    elif insight_type == "your_insight_type":
        impact_area = "Your Impact Area"  # e.g., Sales, Finance, Inventory
        tags = ["tag1", "tag2"]
        
        # Extract title and description
        # ...
        
        # Extract metrics for severity
        # ...
        
        # Get recommendations
        # ...
    # ...
```

### 5. Create a UI component for your insight

Add a display function in `src/ui/pages/insights_view.py`:

```python
def display_your_insight(df):
    """
    Display your insight in the UI.
    
    Args:
        df: The DataFrame containing data
    """
    with st.spinner("Analyzing your insight..."):
        try:
            # Calculate the insight
            insight_result = insight_generator.generate_specific_insight("your_insight_type", df)
            
            if "error" in insight_result:
                st.warning(f"Could not generate insight: {insight_result['error']}")
                return
                
            # Display the summary
            st.subheader("Your Insight Title")
            
            # Create metrics
            # ...
            
            # Create charts
            # ...
            
            # Display insights and recommendations
            # ...
            
        except Exception as e:
            st.error(f"Error analyzing your insight: {str(e)}")
```

### 6. Add your insight to the UI tabs

Update the main UI function to include your new insight:

```python
def insights_view():
    # ...
    tabs = st.tabs([
        "Monthly Gross Margin", 
        "Lead Conversion Rate", 
        "Sales Rep Performance", 
        "Inventory Aging Alerts",
        "Your Insight Name"  # Add your insight here
    ])
    
    # ...
    with tabs[4]:  # Use correct index
        display_your_insight(df)
    # ...
```

### 7. Write tests for your insight

Create a test file in `tests/test_insights/` to test your insight:

```python
import pytest
from src.insights.insight_functions import YourNewInsight

@pytest.fixture
def sample_data():
    # Create sample data for testing
    return pd.DataFrame(...)

def test_your_insight_basic(sample_data):
    # Test basic functionality
    insight = YourNewInsight()
    result = insight.generate(sample_data)
    
    # Assert expected structure and values
    assert result["insight_type"] == "your_insight_type"
    assert "insights" in result
    # ...

def test_your_insight_analysis(sample_data):
    # Test specific analysis logic
    # ...
```

## Best Practices

1. **Focus on actionability**: Insights should lead to clear, actionable recommendations.

2. **Provide context**: Include relevant benchmarks and trends to make insights meaningful.

3. **Prioritize by impact**: Highlight high-impact issues and opportunities.

4. **Normalize data**: Handle variations in column names and data formats gracefully.

5. **Handle errors robustly**: Ensure insights degrade gracefully when data is missing or malformed.

6. **Optimize performance**: Use efficient algorithms and caching for large datasets.

7. **Include visualizations**: Charts make insights more accessible and impactful.

8. **Match business language**: Use terminology familiar to automotive dealership staff.

9. **Enable adaptability**: Use the adaptive learning system to tailor insights to each dealership.

10. **Prepare for forecasting**: Design insights to work with both historical and forecasted data.

## Core Insight Types

### Monthly Gross Margin vs. Target

Analyzes gross margin trends over time compared to targets, identifying issues in pricing, cost management, or sales mix that affect profitability.

**Key metrics:**
- Current month margin vs. target
- Margin trend direction
- Monthly gross profit

**Visualizations:**
- Line chart of margin vs. target over time
- Trend indicators for key metrics

### Lead Conversion Rate by Source

Analyzes lead conversion rates across different sources, identifying top and bottom performers, trends, and opportunities for improvement.

**Key metrics:**
- Overall conversion rate
- Best and worst performing sources
- Conversion rate trends

**Visualizations:**
- Bar chart of conversion rates by source
- Line chart of conversion rate trends

### Sales Performance Analysis

Analyzes sales rep performance, providing detailed metrics on gross profit, unit sales, and efficiency to identify coaching opportunities and best practices.

**Key metrics:**
- Sales rep rankings by volume and profit
- Top and bottom performers
- Negative gross percentage

**Visualizations:**
- Bar chart of sales by rep
- Performance distribution analysis

### Inventory Aging Anomalies

Identifies vehicles staying in inventory significantly longer than similar models, highlighting potential pricing, merchandising, or stocking issues.

**Key metrics:**
- Aged inventory percentage
- Model-specific aging patterns
- Outlier vehicles

**Visualizations:**
- Scatter plot of days on lot vs. model average
- Aging distribution analysis