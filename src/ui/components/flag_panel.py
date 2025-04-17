"""
Flag Panel Component for Watchdog AI.
Displays validation flags and metrics in a structured panel.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional
import re

def _create_issue_chart(flag_counts: Dict[str, int]) -> alt.Chart:
    """
    Create an Altair chart showing issue distribution.
    
    Args:
        flag_counts: Dictionary mapping flag types to counts
        
    Returns:
        Altair chart object
    """
    # Convert flag counts to DataFrame
    df = pd.DataFrame([
        {"Flag": k, "Count": v}
        for k, v in flag_counts.items()
        if v > 0  # Only include flags with counts > 0
    ])
    
    if df.empty:
        return None
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Count:Q', title='Number of Issues'),
        y=alt.Y('Flag:N', title=None, sort='-x'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='reds')),
        tooltip=[
            alt.Tooltip('Flag:N', title='Issue Type'),
            alt.Tooltip('Count:Q', title='Count')
        ]
    ).properties(
        width=400,
        height=min(len(df) * 40, 300)  # Dynamic height based on number of flags
    )
    
    return chart

def _format_markdown_for_streamlit(text: str) -> str:
    """
    Format markdown text for Streamlit display.
    
    Args:
        text: Markdown text to format
        
    Returns:
        Formatted markdown text
    """
    # Add emphasis to numbers
    text = re.sub(r'(\d+)', r'**\1**', text)
    
    # Add emphasis to percentages
    text = re.sub(r'(\d+\.?\d*%)', r'**\1**', text)
    
    # Add emphasis to dollar amounts
    text = re.sub(r'\$(\d+,?\d*\.?\d*)', r'**$\1**', text)
    
    return text

def render_flag_summary(flag_counts: Dict[str, int], total_records: int):
    """
    Render a summary of validation flags.
    
    Args:
        flag_counts: Dictionary mapping flag types to counts
        total_records: Total number of records analyzed
    """
    st.markdown("### Data Quality Summary")
    
    # Calculate metrics
    total_issues = sum(flag_counts.values())
    issue_rate = (total_issues / total_records * 100) if total_records > 0 else 0
    clean_records = total_records - total_issues
    clean_rate = (clean_records / total_records * 100) if total_records > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Records",
            f"{total_records:,}",
            help="Total number of records analyzed"
        )
    
    with col2:
        st.metric(
            "Clean Records",
            f"{clean_records:,}",
            f"{clean_rate:.1f}%",
            help="Records with no validation issues"
        )
    
    with col3:
        st.metric(
            "Issues Found",
            f"{total_issues:,}",
            f"{issue_rate:.1f}%",
            help="Records with one or more validation issues",
            delta_color="inverse"  # Red for positive (more issues)
        )
    
    # Create and display chart if there are issues
    if total_issues > 0:
        chart = _create_issue_chart(flag_counts)
        if chart:
            st.altair_chart(chart, use_container_width=True)
    else:
        st.success("✅ No validation issues found!")

def render_flag_metrics(metrics: Dict[str, Any]):
    """
    Render metrics about validation flags.
    
    Args:
        metrics: Dictionary containing validation metrics
    """
    st.markdown("### Validation Metrics")
    
    # Group metrics by category
    categories = {
        "Data Quality": [
            ("Missing Values", metrics.get("missing_value_rate", 0), "%"),
            ("Invalid Values", metrics.get("invalid_value_rate", 0), "%"),
            ("Duplicate Records", metrics.get("duplicate_rate", 0), "%")
        ],
        "Business Rules": [
            ("Negative Gross", metrics.get("negative_gross_count", 0), "records"),
            ("Missing Lead Source", metrics.get("missing_lead_source_count", 0), "records"),
            ("Invalid VINs", metrics.get("invalid_vin_count", 0), "records")
        ],
        "Performance": [
            ("Processing Time", metrics.get("processing_time", 0), "seconds"),
            ("Memory Usage", metrics.get("memory_usage", 0), "MB"),
            ("Cache Hit Rate", metrics.get("cache_hit_rate", 0), "%")
        ]
    }
    
    # Display metrics by category
    for category, category_metrics in categories.items():
        st.markdown(f"#### {category}")
        cols = st.columns(len(category_metrics))
        
        for col, (label, value, unit) in zip(cols, category_metrics):
            with col:
                if unit == "%":
                    st.metric(
                        label,
                        f"{value:.1f}%",
                        help=f"Percentage of records with {label.lower()}"
                    )
                else:
                    st.metric(
                        label,
                        f"{value:,} {unit}",
                        help=f"Number of {label.lower()}"
                    )
