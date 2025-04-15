"""
Flag Panel Component for Watchdog AI.

This module provides UI components for displaying data quality flags and issues
detected by the insight validator module. It creates a clean, structured panel
that highlights data quality issues and provides options to filter and clean the data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import altair as alt
import re

# Import the insight validator
from ...validators.insight_validator import (
    flag_all_issues,
    summarize_flags,
    generate_flag_summary
)


def _create_issue_chart(df: pd.DataFrame, flag_column: str, title: str) -> alt.Chart:
    """
    Create a small chart visualizing the distribution of flagged vs. non-flagged records.
    
    Args:
        df: DataFrame with flag columns
        flag_column: Name of the flag column to visualize
        title: Title for the chart
        
    Returns:
        Altair chart object
    """
    if flag_column not in df.columns:
        return None
    
    # Count flagged vs non-flagged records
    flag_counts = df[flag_column].value_counts().reset_index()
    flag_counts.columns = ['Flag', 'Count']
    
    # Map boolean values to readable labels
    flag_counts['Flag'] = flag_counts['Flag'].map({True: 'Issue', False: 'Clean'})
    
    # Create the chart
    chart = alt.Chart(flag_counts).mark_bar().encode(
        x='Flag:N',
        y='Count:Q',
        color=alt.Color('Flag:N', scale=alt.Scale(
            domain=['Issue', 'Clean'],
            range=['#FF6961', '#77DD77']  # Red for issues, green for clean
        )),
        tooltip=['Flag:N', 'Count:Q']
    ).properties(
        title=title,
        width=200,
        height=150
    )
    
    return chart


def _format_markdown_for_streamlit(markdown_text: str) -> str:
    """
    Format markdown text for better display in Streamlit.
    
    Args:
        markdown_text: Raw markdown text from generate_flag_summary
        
    Returns:
        Formatted markdown text
    """
    # Replace h1 with h2 to fit better in the UI
    markdown_text = re.sub(r'^# (.+)$', r'## \1', markdown_text, flags=re.MULTILINE)
    
    # Add emoji to section headers
    markdown_text = re.sub(r'^## Issue Summary$', r'## 📊 Issue Summary', markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r'^## Recommendations$', r'## 💡 Recommendations', markdown_text, flags=re.MULTILINE)
    
    return markdown_text


def render_flag_summary(df: pd.DataFrame, 
                        on_clean_click: Optional[Callable[[pd.DataFrame], None]] = None) -> Tuple[pd.DataFrame, bool]:
    """
    Render a user-friendly panel showing data quality issues and flags.
    
    Args:
        df: The input DataFrame to analyze
        on_clean_click: Optional callback function to execute when the "Send Cleaned Data" button is clicked
        
    Returns:
        Tuple containing (potentially cleaned dataframe, boolean indicating if cleaning was performed)
    """
    st.subheader("🔍 Data Quality Assessment")
    
    # Apply flags to the DataFrame if they don't already exist
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    if not flag_columns:
        with st.spinner("Analyzing data quality..."):
            df = flag_all_issues(df)
    
    # Get the summary information
    summary = summarize_flags(df)
    
    # Create tabs for summary and detailed view
    tab1, tab2 = st.tabs(["Summary", "Detailed Report"])
    
    with tab1:
        # Display overview stats in a clean layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Records", 
                value=summary['total_records']
            )
        
        with col2:
            st.metric(
                label="Records with Issues", 
                value=summary['total_issues'],
                delta=f"{100 - summary['percentage_clean']:.1f}% of data",
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                label="Data Quality Score", 
                value=f"{summary['percentage_clean']:.1f}%",
                delta=None if summary['percentage_clean'] > 95 else f"{95 - summary['percentage_clean']:.1f}% below target",
                delta_color="normal" if summary['percentage_clean'] > 95 else "inverse"
            )
        
        # Display individual issue sections
        if summary['total_issues'] > 0:
            st.write("---")
            
            # Process each flag type
            cleaned_df = df.copy()
            cleaning_applied = False
            
            # 1. Negative Gross
            if 'negative_gross_count' in summary and summary['negative_gross_count'] > 0:
                with st.expander(f"📌 Negative Gross Profit ({summary['negative_gross_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {summary['negative_gross_count']} records ({summary['negative_gross_count']/summary['total_records']*100:.1f}% of data) 
                    have negative gross profit values, which may indicate pricing errors or special circumstances.
                    """)
                    
                    # Add chart showing distribution of flagged vs non-flagged records
                    chart = _create_issue_chart(df, 'flag_negative_gross', 'Negative Gross Distribution')
                    if chart:
                        st.altair_chart(chart)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_negative_gross"):
                        st.dataframe(df[df['flag_negative_gross']], use_container_width=True)
                    
                    # Add option to fix negative gross by setting to 0
                    if st.checkbox("📝 Convert negative gross values to zero", key="fix_negative_gross"):
                        # Find gross column(s)
                        gross_cols = [col for col in df.columns if 'gross' in col.lower() and 'flag' not in col.lower()]
                        if gross_cols:
                            for col in gross_cols:
                                cleaned_df.loc[cleaned_df[col] < 0, col] = 0
                            st.success(f"Negative values in {', '.join(gross_cols)} will be converted to zero when cleaned.")
                            cleaning_applied = True
            
            # 2. Missing Lead Source
            if 'missing_lead_source_count' in summary and summary['missing_lead_source_count'] > 0:
                with st.expander(f"📌 Missing Lead Sources ({summary['missing_lead_source_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {summary['missing_lead_source_count']} records ({summary['missing_lead_source_count']/summary['total_records']*100:.1f}% of data) 
                    are missing lead source information, which prevents accurate marketing ROI analysis.
                    """)
                    
                    # Add chart
                    chart = _create_issue_chart(df, 'flag_missing_lead_source', 'Missing Lead Source Distribution')
                    if chart:
                        st.altair_chart(chart)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_missing_lead"):
                        st.dataframe(df[df['flag_missing_lead_source']], use_container_width=True)
                    
                    # Add option to set a default lead source
                    if st.checkbox("📝 Set missing lead sources to 'Unknown'", key="fix_missing_lead"):
                        # Find lead source column(s)
                        lead_cols = [col for col in df.columns if ('lead' in col.lower() or 'source' in col.lower()) and 'flag' not in col.lower()]
                        if lead_cols:
                            for col in lead_cols:
                                cleaned_df.loc[cleaned_df[col].isna() | (cleaned_df[col] == ''), col] = 'Unknown'
                            st.success(f"Missing values in {', '.join(lead_cols)} will be set to 'Unknown' when cleaned.")
                            cleaning_applied = True
            
            # 3. Duplicate VINs
            if 'duplicate_vins_count' in summary and summary['duplicate_vins_count'] > 0:
                with st.expander(f"📌 Duplicate VINs ({summary['duplicate_vins_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {summary['duplicate_vins_count']} records ({summary['duplicate_vins_count']/summary['total_records']*100:.1f}% of data) 
                    have duplicate VIN numbers, which may indicate data entry errors or multiple transactions on the same vehicle.
                    """)
                    
                    # Add chart
                    chart = _create_issue_chart(df, 'flag_duplicate_vin', 'Duplicate VIN Distribution')
                    if chart:
                        st.altair_chart(chart)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_dup_vins"):
                        st.dataframe(df[df['flag_duplicate_vin']], use_container_width=True)
                    
                    # Add option to keep only the latest transaction for each VIN
                    if st.checkbox("📝 Keep only the latest transaction for each VIN", key="fix_dup_vins"):
                        # Find VIN and date columns
                        vin_cols = [col for col in df.columns if 'vin' in col.lower() and 'flag' not in col.lower()]
                        date_cols = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']) and 'flag' not in col.lower()]
                        
                        if vin_cols and date_cols:
                            vin_col = vin_cols[0]
                            date_col = date_cols[0]
                            
                            # Convert date column to datetime if needed
                            if not pd.api.types.is_datetime64_dtype(cleaned_df[date_col]):
                                try:
                                    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
                                except:
                                    st.error(f"Could not convert {date_col} to date format.")
                                    
                            # Sort by date and keep last entry for each VIN
                            cleaned_df = cleaned_df.sort_values(date_col).drop_duplicates(subset=[vin_col], keep='last')
                            st.success(f"Duplicate VINs will be resolved by keeping only the latest transaction when cleaned.")
                            cleaning_applied = True
            
            # 4. Missing/Invalid VINs
            if 'missing_vins_count' in summary and summary['missing_vins_count'] > 0:
                with st.expander(f"📌 Missing/Invalid VINs ({summary['missing_vins_count']} records)", expanded=True):
                    st.markdown(f"""
                    **Issue**: {summary['missing_vins_count']} records ({summary['missing_vins_count']/summary['total_records']*100:.1f}% of data) 
                    have missing or invalid VIN numbers, which complicates inventory tracking and reporting.
                    """)
                    
                    # Add chart
                    chart = _create_issue_chart(df, 'flag_missing_vin', 'Missing/Invalid VIN Distribution')
                    if chart:
                        st.altair_chart(chart)
                    
                    # Show sample of problematic records
                    if st.checkbox("Show affected records", key="show_missing_vins"):
                        st.dataframe(df[df['flag_missing_vin']], use_container_width=True)
                    
                    # Add option to flag for manual review
                    if st.checkbox("📝 Mark records with missing VINs for review", key="fix_missing_vins"):
                        # Add a review flag column
                        cleaned_df['needs_vin_review'] = df['flag_missing_vin']
                        st.success(f"Records with missing/invalid VINs will be marked for review when cleaned.")
                        cleaning_applied = True
            
            # Add the cleanup button
            st.write("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if cleaning_applied:
                    st.info("Cleaning operations are ready to be applied.")
                else:
                    st.info("No cleaning operations selected.")
            
            with col2:
                clean_button = st.button("✅ Send Cleaned Data to Insight Engine", 
                                        type="primary" if cleaning_applied else "secondary",
                                        disabled=not cleaning_applied)
            
            # Handle clean button click
            if clean_button and cleaning_applied:
                st.success("Data cleaned successfully!")
                
                # Remove flag columns from the cleaned data
                cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.startswith('flag_')]
                
                # Execute callback if provided
                if on_clean_click:
                    on_clean_click(cleaned_df)
                
                return cleaned_df, True
            
        else:
            st.success("✨ No data quality issues detected. Data is ready for analysis!")
    
    with tab2:
        # Generate and display the full markdown report
        markdown_report = generate_flag_summary(df)
        formatted_report = _format_markdown_for_streamlit(markdown_report)
        st.markdown(formatted_report)
    
    return df, False


def render_flag_metrics(df: pd.DataFrame) -> None:
    """
    Render a compact metrics panel showing data quality statistics.
    Useful for dashboards where space is limited.
    
    Args:
        df: The input DataFrame to analyze
    """
    # Apply flags to the DataFrame if they don't already exist
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    if not flag_columns:
        df = flag_all_issues(df)
    
    # Get the summary information
    summary = summarize_flags(df)
    
    # Create a clean metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Quality Score", 
            value=f"{summary['percentage_clean']:.1f}%"
        )
        
    with col2:
        st.metric(
            label="Negative Gross", 
            value=summary.get('negative_gross_count', 0)
        )
        
    with col3:
        st.metric(
            label="Missing Sources", 
            value=summary.get('missing_lead_source_count', 0)
        )
        
    with col4:
        st.metric(
            label="VIN Issues", 
            value=summary.get('duplicate_vins_count', 0) + summary.get('missing_vins_count', 0)
        )


if __name__ == "__main__":
    # Sample code for testing
    import streamlit as st
    
    st.set_page_config(page_title="Flag Panel Demo", layout="wide")
    st.title("Watchdog AI - Data Quality Panel Demo")
    
    # Create sample data
    data = {
        'VIN': ['1HGCM82633A123456', '1HGCM82633A123456', '5TFBW5F13AX123457', '789', '', 'WBAGH83576D123458'],
        'Make': ['Honda', 'Honda', 'Toyota', 'Chevrolet', 'Ford', 'BMW'],
        'Model': ['Accord', 'Accord', 'Tundra', 'Malibu', 'F-150', '7 Series'],
        'Year': [2019, 2019, 2020, 2018, 2021, 2018],
        'Sale_Date': ['2023-01-15', '2023-02-10', '2023-02-20', '2023-03-01', '2023-03-15', '2023-03-05'],
        'Sale_Price': [28500.00, 27000.00, 45750.00, 22000.00, 35000.00, 62000.00],
        'Cost': [25000.00, 28000.00, 40000.00, 20000.00, 32000.00, 55000.00],
        'Gross_Profit': [3500.00, -1000.00, 5750.00, 2000.00, 3000.00, 7000.00],
        'Lead_Source': ['Website', None, '', 'Google', 'Autotrader', 'Walk-in'],
        'Salesperson': ['John Smith', 'Jane Doe', 'Jane Doe', 'Bob Johnson', 'John Smith', 'Bob Johnson']
    }
    df = pd.DataFrame(data)
    
    # Demo the flag panel
    cleaned_df, was_cleaned = render_flag_summary(df)
    
    if was_cleaned:
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df)
