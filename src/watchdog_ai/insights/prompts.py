#!/usr/bin/env python
"""
Prompt templates for the Insight Generation System.

This module provides templates and utility functions for generating prompts 
for the OpenAI API to produce structured insights from automotive dealership data.
"""

import json
from typing import Dict, Any, Optional, Union
import jsonschema

# System context definition
AUTOMOTIVE_ANALYST_SYSTEM_CTX = """
## System Context
You are an expert automotive dealership analyst providing data-driven insights to help dealership management make better business decisions. You specialize in analyzing sales data, customer information, inventory metrics, and financial performance to identify trends, opportunities, and potential issues.
"""

# Analysis approach guidelines
ANALYSIS_APPROACH = """
## Analysis Approach
1. **Focus on the data provided** - Base your analysis solely on the information available in the prompt and data context
2. **Be specific and quantitative** - Include actual numbers, percentages, and timeframes
3. **Identify patterns and anomalies** - Look for trends, outliers, and unexpected correlations
4. **Compare across dimensions** - Analyze performance across different categories, time periods, or segments
5. **Consider business impact** - Evaluate how insights affect dealership performance, profitability, and customer satisfaction
"""

# Output guidelines
OUTPUT_GUIDELINES = """
## Output Guidelines

### Summary Section
- Begin with a clear statement that directly answers the query
- Keep summaries concise (1-3 sentences)
- Include the most important metrics/KPIs that support your insight
- Use business language appropriate for dealership management

### Chart Data Section
- Choose the most appropriate chart type:
 - Bar charts for comparing categories
 - Line charts for time series data
 - Pie charts for showing composition
- Ensure chart data properly represents the most relevant aspect of your insight
- Include descriptive titles and axis labels
- Limit charts to 5-7 data points for clarity

### Recommendation Section
- Provide specific, actionable guidance based on the insight
- Focus on practical steps that dealership management can implement
- Include expected outcomes when possible
- When relevant, suggest follow-up analyses to gather more information

### Risk Flag Usage
Set to `true` when the insight reveals:
- Significant negative trends (sales declines, profitability issues)
- Data quality or compliance problems
- Potential fraud or unusual activities
- Customer satisfaction issues
- Inventory imbalances or aging issues
- Any matter requiring immediate attention
"""

# Response format description
RESPONSE_FORMAT = """
## Response Format
Structure your responses in JSON format to ensure consistency and enable programmatic processing. Use the following schema:

```json
{
   "summary": "A clear, concise summary of the main insight",
   "chart_data": {
       "type": "bar|line|pie",
       "data": {"x": [labels], "y": [values]} OR {"labels": [labels], "values": [values]},
       "title": "Chart title",
       "x_axis_label": "Label for X-axis (e.g., 'Model Year')",
       "y_axis_label": "Label for Y-axis (e.g., 'Units Sold')"
   },
   "recommendation": "A specific, actionable recommendation based on the data",
   "risk_flag": true|false,
   "confidence_score": 0.95
}
```

Always include a confidence score between 0 and 1 that represents your level of certainty in the analysis, based on data completeness, quality, and relevance to the question.
"""

# Examples to include in the prompt
PROMPT_EXAMPLES = """
## Example Queries and Responses

### Query Example 1
"Which sales representative has the highest number of deals this month?"

```json
{
   "summary": "James Smith leads all sales representatives with 18 deals this month, accounting for 22% of total sales volume.",
   "chart_data": {
       "type": "bar",
       "data": {
           "x": ["James Smith", "Sarah Johnson", "Mike Williams", "Lisa Brown", "Other Reps"],
           "y": [18, 14, 12, 10, 28]
       },
       "title": "Deals by Sales Representative - Current Month",
       "x_axis_label": "Sales Representative",
       "y_axis_label": "Number of Deals"
   },
   "recommendation": "Consider having James Smith conduct training sessions with newer sales representatives to share his successful techniques. Review his customer approach, follow-up practices, and lead handling methods.",
   "risk_flag": false,
   "confidence_score": 0.95
}
```

### Query Example 2
"What is the average front gross by lead source?"

```json
{
   "summary": "Website leads generate the highest average front gross at $1,284 per deal, 38% higher than the overall average of $932.",
   "chart_data": {
       "type": "bar",
       "data": {
           "x": ["Website", "Phone", "Walk-in", "Referral", "Other"],
           "y": [1284, 942, 875, 1105, 453]
       },
       "title": "Average Front Gross by Lead Source",
       "x_axis_label": "Lead Source",
       "y_axis_label": "Avg. Front Gross ($)"
   },
   "recommendation": "Increase investment in website optimization and digital marketing to drive more high-value leads. Consider implementing targeted promotions specifically for online customers.",
   "risk_flag": false,
   "confidence_score": 0.92
}
```
"""

# Data fields reference
DATA_FIELDS_REFERENCE = """
## Data Fields Reference
Common fields you may encounter in the dealership data:

- **DealNumber**: Unique identifier for each deal
- **DealDate**: Date when the deal was finalized
- **SalesRepName**: Name of the sales representative
- **VehicleYear/Make/Model**: Vehicle information
- **VIN**: Vehicle Identification Number
- **SellingPrice**: Final selling price of the vehicle
- **MSRP**: Manufacturer's Suggested Retail Price
- **Total Gross**: Total gross profit on the deal
- **Front Gross**: Front-end gross profit (vehicle sale)
- **Back Gross**: Back-end gross profit (F&I products)
- **LeadSource**: Origin of the customer lead
- **DaysInInventory**: How long the vehicle was in inventory
- **CustomerZip**: Customer's zip code
"""

# Tone and style guidelines
TONE_AND_STYLE = """
## Tone and Style Guidelines
- **Professional**: Use industry terminology appropriate for dealership management
- **Objective**: Present analysis without bias or assumptions
- **Actionable**: Ensure insights can be translated into concrete actions
- **Concise**: Be direct and to the point; avoid unnecessary elaboration
- **Confident**: When data clearly supports a conclusion, present it assertively
- **Measured**: When data is limited or inconclusive, acknowledge uncertainty

Remember that your primary goal is to help dealership management understand their business performance and make data-driven decisions to improve operations, increase profitability, and enhance customer satisfaction.
"""

# The complete prompt template
INSIGHT_PROMPT_TEMPLATE = f"""
{AUTOMOTIVE_ANALYST_SYSTEM_CTX}

{RESPONSE_FORMAT}

{ANALYSIS_APPROACH}

{OUTPUT_GUIDELINES}

{{custom_instructions}}

## Analytics Data
{{analytics_summary}}

## Query
{{query}}

{PROMPT_EXAMPLES}

{DATA_FIELDS_REFERENCE}

{TONE_AND_STYLE}

Remember to provide your response in the exact JSON format specified earlier, with no additional text outside the JSON structure.
"""

# JSON schema for response validation
INSIGHT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["summary", "chart_data", "recommendation", "risk_flag", "confidence_score"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "A clear, concise summary of the main insight"
        },
        "chart_data": {
            "type": "object",
            "required": ["type", "data", "title"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie"],
                    "description": "The type of chart to display"
                },
                "data": {
                    "type": "object",
                    "description": "The data to display in the chart",
                    "oneOf": [
                        {
                            "required": ["x", "y"],
                            "properties": {
                                "x": {"type": "array", "items": {"type": "string"}},
                                "y": {"type": "array", "items": {"type": "number"}}
                            }
                        },
                        {
                            "required": ["labels", "values"],
                            "properties": {
                                "labels": {"type": "array", "items": {"type": "string"}},
                                "values": {"type": "array", "items": {"type": "number"}}
                            }
                        }
                    ]
                },
                "title": {
                    "type": "string",
                    "description": "Chart title"
                },
                "x_axis_label": {
                    "type": "string",
                    "description": "Label for X-axis"
                },
                "y_axis_label": {
                    "type": "string",
                    "description": "Label for Y-axis"
                }
            }
        },
        "recommendation": {
            "type": "string",
            "description": "A specific, actionable recommendation based on the data"
        },
        "risk_flag": {
            "type": "boolean",
            "description": "Flag indicating whether the insight reveals a potential risk"
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "A score between 0 and 1 representing the confidence in the analysis"
        }
    }
}


def build_prompt(analytics_summary: Union[str, Dict[str, Any]], 
                query: str, 
                custom_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a prompt for the OpenAI API to generate insights.
    
    Args:
        analytics_summary: The analytics data in JSON format or as a dictionary
        query: The specific question or analysis request
        custom_params: Optional dictionary of custom parameters for the analysis
            This can include:
                - focus_metric: The specific metric to focus on
                - timeframe: The time period to analyze
                - chart_type: Override the default chart type
                - depth: Level of detail in the analysis (e.g., "high", "medium", "low")
                - comparison: What to compare against (e.g., "previous_period", "industry_average")
    
    Returns:
        A formatted prompt string ready to be sent to the OpenAI API
    
    Example:
        >>> analytics_data = {"sales": [{"rep": "John", "amount": 10000}, {"rep": "Jane", "amount": 15000}]}
        >>> query = "Who is the top performing sales representative?"
        >>> custom_params = {"chart_type": "bar", "depth": "high"}
        >>> prompt = build_prompt(analytics_data, query, custom_params)
    """
    # Convert analytics_summary to string if it's a dictionary
    if isinstance(analytics_summary, dict):
        analytics_summary = json.dumps(analytics_summary, indent=2)
    
    # Build custom instructions based on custom_params
    custom_instructions = "## Custom Analysis Parameters\n"
    if custom_params:
        for key, value in custom_params.items():
            custom_instructions += f"- {key}: {value}\n"
    else:
        custom_instructions = ""  # No custom parameters provided
    
    # Format the prompt
    return INSIGHT_PROMPT_TEMPLATE.format(
        analytics_summary=analytics_summary,
        query=query,
        custom_instructions=custom_instructions
    )


def validate_insight_response(response: Dict[str, Any]) -> None:
    """
    Validate that an insight response adheres to the expected schema.
    
    Args:
        response: The parsed JSON response from the OpenAI API
    
    Raises:
        jsonschema.exceptions.ValidationError: If the response fails validation
        
    Example:
        >>> response = {
        ...     "summary": "Sales are up 15% this month.",
        ...     "chart_data": {
        ...         "type": "bar",
        ...         "data": {"x": ["Jan", "Feb", "Mar"], "y": [10, 15, 20]},
        ...         "title": "Monthly Sales",
        ...         "x_axis_label": "Month",
        ...         "y_axis_label": "Sales ($)"
        ...     },
        ...     "recommendation": "Focus on high-performing products.",
        ...     "risk_flag": False,
        ...     "confidence_score": 0.85
        ... }
        >>> validate_insight_response(response)  # No exception raised if valid
    """
    try:
        jsonschema.validate(instance=response, schema=INSIGHT_RESPONSE_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        # Re-raise with a more user-friendly error message
        raise jsonschema.exceptions.ValidationError(
            f"Invalid insight response format: {str(e)}"
        ) from e

