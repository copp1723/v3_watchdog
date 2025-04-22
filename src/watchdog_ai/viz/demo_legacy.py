"""
Demo script for Watchdog AI visualization layer.

This script demonstrates how to use the chart builders with 
mobile-responsive sizing in a Streamlit application.

Run with:
    streamlit run src/watchdog_ai/viz/demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import logging
from datetime import datetime, timedelta

# Import chart builders from viz package
from watchdog_ai.viz import (
    sales_trend_plot,
    category_distribution_plot,
    get_responsive_dimensions,
    DEFAULT_COLOR_SCHEME,
    SEQUENTIAL_COLOR_SCHEME,
    DIVERGING_COLOR_SCHEME
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_sales_data(days=90, categories=None):
    """
    Generate sample sales data for demonstration.
    
    Args:
        days: Number of days of data to generate
        categories: List of product categories (default provides 5 categories)
        
    Returns:
        DataFrame with synthetic sales data
    """
    logger.info(f"Generating sample data for {days} days across {len(categories) if categories else 5} categories")
    
    if categories is None:
        categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Books']
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Create data for each category
    data = []
    for category in categories:
        # Category Distribution
        st.header("📊 Category Distribution")
        st.markdown("Compare bar and pie chart visualizations with responsive sizing.")
        
        # Split view for comparison charts
        # Note: On mobile, these will stack vertically
        # and use the full container width
        col1, col2 = st.columns(2)
            # Weekly seasonality (weekend boost)
            weekly_effect = np.sin(date.weekday() * np.pi / 7) * 100
            
            # Monthly seasonality (different patterns for each category)
            monthly_effect = 0
            if category in ['Clothing', 'Electronics']:
                monthly_effect = np.sin(date.day * np.pi / 30) * 150
            elif category in ['Food', 'Books']:
                monthly_effect = np.cos(date.day * np.pi / 30) * 100
            
            # Calculate units and price
            units = max(0, int(base + weekly_effect + monthly_effect + np.random.normal(0, 50)))
            price = np.random.uniform(10, 100)  # Random price between 10 and 100
            revenue = units * price
            
            # Weekend boost
            if date.weekday() >= 5:  # Saturday or Sunday
                units = int(units * 1.3)  # 30% boost on weekends
                revenue = units * price
            
            data.append({
                'date': date,
                'category': category,
                'units': units,
                'price': price,
                'revenue': revenue
            })
    
    return pd.DataFrame(data)

def generate_category_summary(df):
    """Generate category summary data for visualizations."""
    return df.groupby('category').agg({
        'units': 'sum',
        'revenue': 'sum'
    }).reset_index()

def show_code_example(code):
    """Display formatted code example."""
    st.code(textwrap.dedent(code), language='python')

def main():
    """Main demo application."""
    st.title("📊 Watchdog AI Visualization Demo")
    
    try:
        st.markdown("""
        This demo showcases the visualization capabilities of Watchdog AI with responsive chart builders.
        The charts automatically adapt to different screen sizes - try adjusting your browser window!
        """)
        
        # Sidebar configuration
        st.sidebar.header("📈 Configuration")
        days = st.sidebar.slider("Days of Data", 30, 365, 90)
        num_categories = st.sidebar.slider("Number of Categories", 3, 8, 5)
        categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Books', 'Sports', 'Toys', 'Health'][:num_categories]
        
        # Chart settings
        st.sidebar.subheader("Chart Settings")
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            [DEFAULT_COLOR_SCHEME, SEQUENTIAL_COLOR_SCHEME, DIVERGING_COLOR_SCHEME],
            format_func=lambda x: x.title()
        )
        
        show_points = st.sidebar.checkbox("Show Points", value=True)
        animate = st.sidebar.checkbox("Enable Animation", value=True)
        interactive = st.sidebar.checkbox("Interactive (pan/zoom)", value=True)
        
        # Generate sample data
        with st.spinner("Generating sample data..."):
            sales_df = generate_sample_sales_data(days=days, categories=categories)
            category_df = generate_category_summary(sales_df)
        
        # Get container width for responsive sizing
        container_width = st.container().width
        chart_height = 400 if container_width > 768 else 300
        
        # Sales Trend Visualization
        st.header("📈 Sales Trend")
        st.markdown("Interactive time series visualization with mobile-responsive sizing.")
        
        with st.expander("Show code"):
            show_code_example("""
            sales_chart = sales_trend_plot(
                df=sales_df,
                date_column='date',
                value_column='revenue',
                category_column='category',
                title='Sales Trend by Category',
                container_width=container_width,
                show_points=show_points,
                animate=animate
            )
            st.altair_chart(sales_chart)
            """)
        
        sales_chart = sales_trend_plot(
            df=sales_df,
            date_column='date',
            value_column='revenue',
            category_column='category',
            title='Sales Trend by Category',
            container_width=container_width,
            height=chart_height,
            color_scheme=color_scheme,
            show_points=show_points,
            animate=animate
        )
        st.altair_chart(sales_chart)
        
        # Category Distribution
        st.header("📊 Category Distribution")
        st.markdown("Compare bar and pie chart visualizations with responsive sizing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bar_chart = category_distribution_plot(
                df=category_df,
                category_column='category',
                value_column='revenue',
                chart_type='bar',
                title='Revenue by Category',
                container_width=container_width / 2,
                height=chart_height,
                color_scheme=color_scheme
            )
            st.altair_chart(bar_chart)
        
        with col2:
            pie_chart = category_distribution_plot(
                df=category_df,
                category_column='category',
                value_column='revenue',
                chart_type='pie',
                title='Revenue Distribution',
                container_width=container_width / 2,
                height=chart_height,
                color_scheme=color_scheme
            )
            st.altair_chart(pie_chart)
        
        # Error handling examples
        st.header("🐛 Error Handling")
        with st.expander("View error handling examples"):
            error_tabs = st.tabs(["Empty Data", "Missing Columns", "Invalid Types"])
            
            with error_tabs[0]:
                st.markdown("Empty DataFrame example:")
                empty_df = pd.DataFrame()
                error_chart = sales_trend_plot(empty_df, container_width=container_width)
                st.altair_chart(error_chart)
            
            with error_tabs[1]:
                st.markdown("Missing required columns example:")
                bad_df = sales_df.drop(columns=['date'])
                error_chart = sales_trend_plot(bad_df, container_width=container_width)
                st.altair_chart(error_chart)
            
            with error_tabs[2]:
                st.markdown("Invalid data types example:")
                invalid_df = sales_df.copy()
                invalid_df['date'] = 'invalid'
                error_chart = sales_trend_plot(invalid_df, container_width=container_width)
                st.altair_chart(error_chart)
        
        # Resources
        st.header("💡 Resources")
        st.markdown("""
        ### Documentation
        - [Watchdog AI Visualization Guide](https://docs.watchdogai.com/visualization)
        - [Chart Builder API Reference](https://docs.watchdogai.com/api/viz)
        
        ### Mobile Optimization Tips
        - Charts automatically adjust to screen size
        - Font sizes and margins are optimized for mobile
        - Touch-friendly interactions on mobile devices
        """)
        
        # Version info
        st.sidebar.markdown("---")
        st.sidebar.caption("Watchdog AI Visualization Layer v0.1.0")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

