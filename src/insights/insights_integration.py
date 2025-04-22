"""
Integration module for connecting Watchdog AI analytics with the InsightsDigest system.
This module provides the functions necessary to transform Watchdog AI's analysis results
into structured insights that can be used to generate comprehensive reports.
"""

from typing import List, Dict, Any, Optional, Tuple
from .insights_digest import (
    InsightsDigest, 
    create_insights_digest,
    SeverityLevel, 
    PriorityLevel
)


def anomaly_to_insight(
    anomaly_data: Dict[str, Any],
    impact_area: str = None,
    severity: str = "Medium"
) -> Dict[str, Any]:
    """
    Convert an anomaly detection result into an insight dictionary.
    
    Args:
        anomaly_data: Dictionary containing the anomaly detection results
        impact_area: The dealership area affected by this anomaly
        severity: Severity level (High, Medium, Low)
        
    Returns:
        A dictionary formatted for use in create_insights_digest
    """
    # Extract anomaly data from the result
    anomaly_type = anomaly_data.get("anomaly_type", "Unknown Anomaly")
    column = anomaly_data.get("column", "")
    value = anomaly_data.get("value", None)
    description = anomaly_data.get("description", "")
    
    # Determine if this is a problem or opportunity
    is_problem = anomaly_data.get("is_problem", True)
    
    # Generate a title based on the anomaly type
    title = anomaly_type if anomaly_type else f"Anomaly in {column}"
    
    # Create a more detailed description if needed
    full_description = description if description else f"An unusual value was detected in {column}"
    if value is not None:
        full_description += f": {value}"
    
    # Parse the severity - default to Medium if not valid
    try:
        severity_level = SeverityLevel(severity)
    except ValueError:
        severity_level = SeverityLevel.MEDIUM
    
    # Create tags based on the data
    tags = []
    if column:
        tags.append(column.lower().replace(" ", "_"))
    if impact_area:
        tags.append(impact_area.lower().replace(" ", "_"))
    if anomaly_type:
        tags.append(anomaly_type.lower().replace(" ", "_"))
    
    # Create recommendations (example - would be more sophisticated in practice)
    recommendations = []
    if anomaly_data.get("recommendations"):
        recommendations = anomaly_data["recommendations"]
    else:
        recommendations = [{
            "action": f"Investigate the {anomaly_type.lower()} detected in {column}",
            "priority": "Medium",
            "estimated_impact": "Potential improvement in data quality and decision-making",
            "tags": tags
        }]
    
    # Build the insight structure
    insight = {
        "title": title,
        "detail": {
            "description": full_description,
            "impact_area": impact_area,
            "severity": severity_level.value,
            "data_source": anomaly_data.get("data_source", "Anomaly Detection"),
            "detection_method": anomaly_data.get("detection_method", "Anomaly Detection"),
            "metric_value": value,
            "metric_unit": anomaly_data.get("unit", ""),
            "benchmark": anomaly_data.get("benchmark", None),
            "tags": tags
        },
        "recommendations": recommendations,
        "tags": tags
    }
    
    return insight


def comparison_to_insight(
    comparison_data: Dict[str, Any],
    impact_area: str = None
) -> Dict[str, Any]:
    """
    Convert a statistical comparison result into an insight dictionary.
    
    Args:
        comparison_data: Dictionary containing the comparison results
        impact_area: The dealership area affected by this comparison
        
    Returns:
        A dictionary formatted for use in create_insights_digest
    """
    # Extract comparison data
    comparison_type = comparison_data.get("comparison_type", "Statistical Comparison")
    p_value = comparison_data.get("p_value", 1.0)
    columns = comparison_data.get("columns", [])
    description = comparison_data.get("description", "")
    
    # Determine significance and severity based on p-value
    is_significant = p_value < 0.05
    if p_value < 0.01:
        significance = "highly significant"
        severity = SeverityLevel.HIGH.value
    elif p_value < 0.05:
        significance = "significant"
        severity = SeverityLevel.MEDIUM.value
    else:
        significance = "not significant"
        severity = SeverityLevel.LOW.value
    
    # Determine if this is a problem or opportunity
    is_problem = comparison_data.get("is_problem", not is_significant)
    
    # Generate a title
    column_str = " vs ".join(columns) if columns else "variables"
    title = f"{'Significant' if is_significant else 'No significant'} difference in {column_str}"
    
    # Create a detailed description
    full_description = description if description else f"The comparison between {column_str} is {significance} (p = {p_value:.4f})."
    
    # Create tags based on the data
    tags = []
    for col in columns:
        tags.append(col.lower().replace(" ", "_"))
    if impact_area:
        tags.append(impact_area.lower().replace(" ", "_"))
    
    # Create recommendations
    recommendations = []
    if comparison_data.get("recommendations"):
        recommendations = comparison_data["recommendations"]
    elif is_significant:
        recommendations = [{
            "action": f"Investigate the factors driving the difference in {column_str}",
            "priority": "High" if p_value < 0.01 else "Medium",
            "estimated_impact": "Potential optimization of business processes",
            "tags": tags
        }]
    else:
        recommendations = [{
            "action": f"Continue monitoring {column_str} for changes over time",
            "priority": "Low",
            "tags": tags
        }]
    
    # Build the insight structure
    insight = {
        "title": title,
        "detail": {
            "description": full_description,
            "impact_area": impact_area,
            "severity": severity,
            "data_source": comparison_data.get("data_source", "Statistical Analysis"),
            "detection_method": "Statistical Comparison",
            "tags": tags
        },
        "recommendations": recommendations,
        "tags": tags
    }
    
    return insight


def segment_to_insight(
    segment_data: Dict[str, Any],
    impact_area: str = None
) -> List[Dict[str, Any]]:
    """
    Convert customer segmentation results into insight dictionaries.
    
    Args:
        segment_data: Dictionary containing the segmentation results
        impact_area: The dealership area affected by this segmentation
        
    Returns:
        A list of dictionaries formatted for use in create_insights_digest
    """
    insights = []
    
    # Extract segments
    segments = segment_data.get("segments", [])
    
    for idx, segment in enumerate(segments):
        # Skip noise segments (usually ID < 0 in DBSCAN)
        if segment.get("id", 0) < 0:
            continue
            
        segment_name = segment.get("name", f"Segment {idx+1}")
        segment_size = segment.get("size", 0)
        segment_pct = segment.get("percentage", 0)
        characteristics = segment.get("characteristics", {})
        categorical_insights = segment.get("categorical_insights", [])
        
        # Determine if this segment represents a problem or opportunity
        # This would be based on business rules in a real implementation
        is_problem = False  # For segmentation, usually we treat all segments as opportunities
        
        # Generate a title
        title = f"Customer Segment: {segment_name} ({segment_pct:.1f}% of customers)"
        
        # Create a detailed description
        char_strings = []
        for field, values in characteristics.items():
            if values.get("diff_pct", 0) != 0:
                sign = "+" if values.get("diff_pct", 0) > 0 else ""
                char_strings.append(
                    f"{field}: {values.get('segment_avg', 0):.2f} ({sign}{values.get('diff_pct', 0):.1f}% vs. average)"
                )
                
        characteristics_text = ", ".join(char_strings[:3])  # Top 3 characteristics
        
        categorical_text = ""
        if categorical_insights:
            categorical_text = " Additional traits: " + "; ".join(categorical_insights[:2])
            
        full_description = (
            f"A distinct customer segment representing {segment_pct:.1f}% of the customer base. " +
            f"Key characteristics: {characteristics_text}.{categorical_text}"
        )
        
        # Create tags based on segment characteristics
        tags = ["segmentation", segment_name.lower().replace(" ", "_")]
        if impact_area:
            tags.append(impact_area.lower().replace(" ", "_"))
        
        # Determine severity based on segment size
        if segment_pct > 30:
            severity = SeverityLevel.HIGH.value
        elif segment_pct > 10:
            severity = SeverityLevel.MEDIUM.value
        else:
            severity = SeverityLevel.LOW.value
        
        # Create segment-specific recommendations
        # In a real implementation, these would be derived from segment characteristics
        recommendations = []
        if segment_data.get("recommendations", {}).get(segment_name):
            recommendations = segment_data["recommendations"][segment_name]
        else:
            recommendations = [{
                "action": f"Develop targeted marketing for the {segment_name} segment",
                "priority": "High" if segment_pct > 25 else "Medium",
                "estimated_impact": f"Improved conversion rates for {segment_pct:.1f}% of customer base",
                "tags": tags + ["marketing"]
            }]
            
            # Add a second recommendation based on segment size
            if segment_pct > 20:
                recommendations.append({
                    "action": f"Refine product offerings based on {segment_name} preferences",
                    "priority": "Medium",
                    "estimated_impact": "Increased customer satisfaction and retention",
                    "tags": tags + ["product"]
                })
        
        # Build the insight structure
        insight = {
            "title": title,
            "detail": {
                "description": full_description,
                "impact_area": impact_area,
                "severity": severity,
                "data_source": segment_data.get("data_source", "Customer Data"),
                "detection_method": "Customer Segmentation",
                "metric_value": segment_pct,
                "metric_unit": "%",
                "tags": tags
            },
            "recommendations": recommendations,
            "tags": tags
        }
        
        insights.append(insight)
    
    return insights


def generate_insights_from_analysis(
    anomalies: List[Dict[str, Any]] = None,
    comparisons: List[Dict[str, Any]] = None,
    segments: List[Dict[str, Any]] = None,
    dealer_name: str = None,
    data_time_range: str = None
) -> InsightsDigest:
    """
    Generate a comprehensive insights digest from various analysis results.
    
    Args:
        anomalies: List of anomaly detection results
        comparisons: List of statistical comparison results
        segments: List of customer segmentation results
        dealer_name: Name of the dealership
        data_time_range: Time period the data covers
        
    Returns:
        An InsightsDigest object with problems and opportunities
    """
    problems = []
    opportunities = []
    
    # Process anomalies
    if anomalies:
        for anomaly in anomalies:
            insight = anomaly_to_insight(anomaly)
            if anomaly.get("is_problem", True):
                problems.append(insight)
            else:
                opportunities.append(insight)
    
    # Process comparisons
    if comparisons:
        for comparison in comparisons:
            insight = comparison_to_insight(comparison)
            if comparison.get("is_problem", comparison.get("p_value", 1.0) >= 0.05):
                problems.append(insight)
            else:
                opportunities.append(insight)
    
    # Process segments - these typically generate multiple insights
    if segments:
        for segment_result in segments:
            segment_insights = segment_to_insight(segment_result)
            # For segments, we typically treat them as opportunities
            opportunities.extend(segment_insights)
    
    # Create the digest
    digest = create_insights_digest(problems, opportunities)
    
    # Set additional metadata
    if dealer_name:
        digest.dealer_name = dealer_name
    if data_time_range:
        digest.data_time_range = data_time_range
    
    return digest


# Example function for tool integration with ScriptRunner
def integrate_with_watchdog_tools(script_runner):
    """
    Example of how to integrate the insights digest with Watchdog AI tools.
    
    Args:
        script_runner: The ScriptRunner instance from Watchdog AI server
        
    Returns:
        A function that can be called to generate insights from analysis results
    """
    def generate_insights(
        anomalies=None, 
        comparisons=None, 
        segments=None, 
        dealer_name=None, 
        data_time_range=None
    ):
        # Use the data from ScriptRunner's memory if available
        if not anomalies and 'anomalies' in script_runner.data:
            anomalies = script_runner.data['anomalies'].to_dict('records')
            
        if not comparisons and 'comparisons' in script_runner.data:
            comparisons = script_runner.data['comparisons'].to_dict('records')
            
        if not segments and 'segments' in script_runner.data:
            segments = script_runner.data['segments'].to_dict('records')
            
        # Generate the insights digest
        digest = generate_insights_from_analysis(
            anomalies=anomalies,
            comparisons=comparisons,
            segments=segments,
            dealer_name=dealer_name,
            data_time_range=data_time_range
        )
        
        # Store the digest in memory for later use
        script_runner.data['insights_digest'] = digest
        
        # Return the digest in various formats
        return {
            "digest": digest,
            "json": digest.to_json(),
            "markdown": digest.to_markdown(),
            "health_score": digest.get_overall_health_score()
        }
    
    return generate_insights


def create_tool_output_from_digest(digest: InsightsDigest):
    """
    Creates a formatted tool output for Claude based on an InsightsDigest.
    
    Args:
        digest: The InsightsDigest object to format
        
    Returns:
        A dictionary with the formatted output for Claude
    """
    markdown = digest.to_markdown()
    health_score = digest.get_overall_health_score()
    
    # Create a summary section
    summary = "## Dealership Insights Summary\n\n"
    summary += f"**Overall Health Score:** {health_score:.1f}/100\n\n"
    summary += f"**Problems Identified:** {len(digest.top_problems)}\n"
    summary += f"**Opportunities Identified:** {len(digest.top_opportunities)}\n\n"
    
    # Add problem categories
    if digest.top_problems:
        problem_areas = {}
        for problem in digest.top_problems:
            area = problem.detail.impact_area or "General"
            problem_areas[area] = problem_areas.get(area, 0) + 1
        
        summary += "**Problem Areas:**\n"
        for area, count in problem_areas.items():
            summary += f"- {area}: {count} issue(s)\n"
        summary += "\n"
    
    # Add opportunity categories
    if digest.top_opportunities:
        opportunity_areas = {}
        for opportunity in digest.top_opportunities:
            area = opportunity.detail.impact_area or "General"
            opportunity_areas[area] = opportunity_areas.get(area, 0) + 1
        
        summary += "**Opportunity Areas:**\n"
        for area, count in opportunity_areas.items():
            summary += f"- {area}: {count} opportunity(s)\n"
        summary += "\n"
    
    # Add high priority recommendations
    high_priority_recs = []
    for entry in digest.top_problems + digest.top_opportunities:
        for rec in entry.recommendations:
            if rec.priority == PriorityLevel.HIGH:
                high_priority_recs.append((entry.title, rec.action))
    
    if high_priority_recs:
        summary += "**Top Priority Recommendations:**\n"
        for title, action in high_priority_recs[:5]:  # Top 5 high priority recommendations
            summary += f"- **{title}**: {action}\n"
        summary += "\n"
    
    # Combine summary and full report
    full_report = summary + "\n---\n\n" + markdown
    
    return {
        "content": [
            {
                "type": "text",
                "text": "Insights analysis completed successfully."
            },
            {
                "type": "text",
                "text": full_report
            }
        ],
        "metadata": {
            "insight_count": len(digest.top_problems) + len(digest.top_opportunities),
            "health_score": health_score,
            "digest_id": str(digest.generated_at)
        }
    }
