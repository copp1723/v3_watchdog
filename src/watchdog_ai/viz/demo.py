"""
Demo script for Watchdog AI visualization layer.

This script demonstrates how to use the chart builders with 
mobile-responsive sizing in a Streamlit application.

Run with:
    streamlit run src/watchdog_ai/viz/demo.py
"""

# Standard library imports
import logging
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from watchdog_ai.viz import (
    DEFAULT_COLOR_SCHEME,
    DIVERGING_COLOR_SCHEME,
    SEQUENTIAL_COLOR_SCHEME,
    category_distribution_plot,
    get_responsive_dimensions,
    sales_trend_plot,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################
# DATA GENERATION LAYER
###############################

def _validate_input_parameters(days: int, categories: List[str]) -> None:
    """
    Validate input parameters for data generation.
    
    Args:
        days: Number of days of data to generate
        categories: List of product categories
        
    Raises:
        ValueError: If input parameters are invalid
    """
    if not isinstance(days, int) or days <= 0:
        raise ValueError(f"Days must be a positive integer, got {days}")
    
    if not categories or not isinstance(categories, list) or len(categories) == 0:
        raise ValueError("Categories must be a non-empty list")


def generate_sample_sales_data(
    days: int = 90, 
    categories: Optional[List[str]] = None,
    base_unit_range: tuple = (500, 1000),
    price_range: tuple = (10, 100),
    weekend_multiplier: float = 1.3,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate sample sales data for demonstration.
    
    This function creates synthetic time series data representing sales across
    different product categories. It includes seasonal patterns (weekly and monthly)
    and random variations to simulate realistic sales data.
    
    Args:
        days: Number of days of data to generate
        categories: List of product categories (default provides 5 categories)
        base_unit_range: Tuple of (min, max) for base unit sales
        price_range: Tuple of (min, max) for price range
        weekend_multiplier: Multiplier for weekend sales boost (e.g. 1.3 = 30% increase)
        random_seed: Optional seed for reproducible results
        
    Returns:
        DataFrame with synthetic sales data containing columns:
        - date: Datetime of the sale
        - category: Product category
        - units: Number of units sold
        - price: Unit price
        - revenue: Total revenue (units * price)
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Default categories if none provided
    if categories is None:
        categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Books']
    
    # Validate inputs
    _validate_input_parameters(days, categories)
    
    logger.info(f"Generating sample data for {days} days across {len(categories)} categories")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Create data for each category
    data = []
    for category in categories:
        # Base sales (different for each category)
        base = np.random.randint(base_unit_range[0], base_unit_range[1])
        
        # Generate data for each date
        for date in dates:
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
            price = np.random.uniform(price_range[0], price_range[1])
            
            # Weekend boost
            if date.weekday() >= 5:  # Saturday or Sunday
                units = int(units * weekend_multiplier)
            
            # Calculate revenue after any adjustments to units
            revenue = units * price
            
            data.append({
                'date': date,
                'category': category,
                'units': units,
                'price': price,
                'revenue': revenue
            })
    
    # Convert to DataFrame and ensure proper data types
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['units'] = df['units'].astype(int)
    df['price'] = df['price'].astype(float)
    df['revenue'] = df['revenue'].astype(float)
    
    return df

###############################
# DATA PROCESSING LAYER
###############################

def _validate_sales_df(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame has the required structure for sales data.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If DataFrame doesn't meet requirements
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_columns = ['date', 'category', 'units', 'revenue']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("'date' column must be datetime type")
    
    if not pd.api.types.is_numeric_dtype(df['units']):
        raise ValueError("'units' column must be numeric type")
        
    if not pd.api.types.is_numeric_dtype(df['revenue']):
        raise ValueError("'revenue' column must be numeric type")


def generate_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of sales data aggregated by category.
    
    This function takes sales data and computes aggregate statistics for each category,
    including total revenue, total units, average price, and sales date range.
    
    Args:
        df: DataFrame with sales data, must contain 'category', 'units', 'revenue', 'date' columns
        
    Returns:
        DataFrame with category summary data
        
    Raises:
        ValueError: If input DataFrame doesn't meet requirements
    """
    # Validate input DataFrame
    _validate_sales_df(df)
    
    # Create the summary
    summary = df.groupby('category').agg({
        'units': 'sum',
        'revenue': 'sum',
        'price': 'mean',
        'date': ['min', 'max']
    })
    
    # Flatten the MultiIndex columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Calculate average price (if not already present)
    if 'price_mean' not in summary.columns:
        summary['price_mean'] = summary['revenue_sum'] / summary['units_sum']
    
    # Reset index to make category a column
    summary = summary.reset_index()
    
    return summary


def show_code_example(code: str) -> None:
    """
    Display formatted code example with proper indentation.
    
    Args:
        code: String containing code to be displayed
    """
    st.code(textwrap.dedent(code), language='python')


###############################
# UI / VISUALIZATION LAYER
###############################

def main():
    """
    Main demo application with Streamlit UI.
    
    This function contains all UI and visualization components, organized into sections:
    - Configuration sidebar
    - Sales trend visualization
    - Category distribution visualization
    - Error handling examples
    - Resources and documentation
    
    The UI is designed to be mobile-responsive and handle errors gracefully.
    """
    try:
        st.title("📊 Watchdog AI Visualization Demo")
        
        st.markdown("""
        This demo showcases the visualization capabilities of Watchdog AI with responsive chart builders.
        The charts automatically adapt to different screen sizes - try adjusting your browser window!
        """)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("📈 Configuration")
            
            # Data generation parameters
            days = st.slider("Days of Data", 30, 365, 90)
            num_categories = st.slider("Number of Categories", 3, 8, 5)
            all_categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Books', 'Sports', 'Toys', 'Health']
            categories = all_categories[:num_categories]
            
            # Advanced options in expander
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    min_price = st.number_input("Min Price", 1, 50, 10)
                    weekend_boost = st.slider("Weekend Boost", 1.0, 2.0, 1.3, 0.1)
                with col2:
                    max_price = st.number_input("Max Price", 51, 500, 100)
                    random_seed = st.number_input("Random Seed", 0, 9999, 42)
                
                if min_price >= max_price:
                    st.warning("Min price must be less than max price")
                    min_price, max_price = 10, 100
            
            # Chart settings
            st.subheader("Chart Settings")
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["default", "sequential", "diverging"],
                format_func=lambda x: x.title(),
                index=0
            )
            
            # Map selection to actual color scheme
            color_scheme_map = {
                "default": DEFAULT_COLOR_SCHEME,
                "sequential": SEQUENTIAL_COLOR_SCHEME,
                "diverging": DIVERGING_COLOR_SCHEME
            }
            selected_color_scheme = color_scheme_map[color_scheme]
            
            show_points = st.checkbox("Show Points", value=True)
            animate = st.checkbox("Enable Animation", value=True)
            interactive = st.checkbox("Interactive (pan/zoom)", value=True)
            
            # Add data regeneration button
            if st.button("Regenerate Data"):
                st.session_state.data_generated = False
        
        # Initialize session state for data caching
        if 'data_generated' not in st.session_state:
            st.session_state.data_generated = False
        
        # Generate sample data
        if not st.session_state.data_generated:
            with st.spinner("Generating sample data..."):
                try:
                    sales_df = generate_sample_sales_data(
                        days=days, 
                        categories=categories,
                        price_range=(min_price, max_price),
                        weekend_multiplier=weekend_boost,
                        random_seed=random_seed
                    )
                    category_df = generate_category_summary(sales_df)
                    
                    # Store in session state
                    st.session_state.sales_df = sales_df
                    st.session_state.category_df = category_df
                    st.session_state.data_generated = True
                    
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
                    logger.error(f"Data generation error: {str(e)}")
                    return
        else:
            # Use cached data
            sales_df = st.session_state.sales_df
            category_df = st.session_state.category_df
        
        # Get container width for responsive sizing
        container_width = st.container().width
        chart_height = 400 if container_width > 768 else 300
        
        # Data Overview
        st.header("📋 Data Overview")
        with st.expander("View sample data"):
            st.dataframe(sales_df.head(10))
            
            # Add download button for CSV
            csv = sales_df.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset (CSV)",
                data=csv,
                file_name="sales_data.csv",
                mime="text/csv"
            )
        
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
        
        try:
            sales_chart = sales_trend_plot(
                df=sales_df,
                date_column='date',
                value_column='revenue',
                category_column='category',
                title='Sales Trend by Category',
                container_width=container_width,
                height=chart_height,
                color_scheme=selected_color_scheme,
                show_points=show_points,
                animate=animate,
                interactive=interactive
            )
            st.altair_chart(sales_chart, use_container_width=True)
            
            # Add rolling average chart
            st.subheader("7-Day Rolling Average")
            # Calculate 7-day rolling average by category
            rolling_df = sales_df.copy()
            rolling_df['date'] = pd.to_datetime(rolling_df['date'])
            rolling_df = rolling_df.sort_values('date')
            
            # Group by category and date, then calculate rolling average
            rolling_data = []
            for category in rolling_df['category'].unique():
                cat_data = rolling_df[rolling_df['category'] == category].copy()
                cat_data['rolling_avg'] = cat_data['revenue'].rolling(window=7, min_periods=1).mean()
                rolling_data.append(cat_data)
            
            rolling_df = pd.concat(rolling_data)
            
            rolling_chart = sales_trend_plot(
                df=rolling_df,
                date_column='date',
                value_column='rolling_avg',
                category_column='category',
                title='7-Day Rolling Average by Category',
                container_width=container_width,
                height=chart_height,
                color_scheme=selected_color_scheme,
                show_points=False,
                animate=animate,
                interactive=interactive
            )
            st.altair_chart(rolling_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating sales trend chart: {str(e)}")
            logger.error(f"Chart creation error: {str(e)}")
        
        # Category Distribution
        st.header("📊 Category Distribution")
        st.markdown("Compare bar and pie chart visualizations with responsive sizing.")
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                bar_chart = category_distribution_plot(
                    df=category_df,
                    category_column='category',
                    value_column='revenue_sum',
                    chart_type='bar',
                    title='Revenue by Category',
                    container_width=container_width / 2,
                    height=chart_height,
                    color_scheme=selected_color_scheme
                )
                st.altair_chart(bar_chart, use_container_width=True)
            
            with col2:
                pie_chart = category_distribution_plot(
                    df=category_df,
                    category_column='category',
                    value_column='revenue_sum',
                    chart_type='pie',
                    title='Revenue Distribution',
                    container_width=container_width / 2,
                    height=chart_height,
                    color_scheme=selected_color_scheme
                )
                st.altair_chart(pie_chart, use_container_width=True)
                
            # Add units view as well
            st.subheader("Units Sold by Category")
            col1, col2 = st.columns(2)
            
            with col1:
                units_bar = category_distribution_plot(
                    df=category_df,
                    category_column='category',
                    value_column='units_sum',
                    chart_type='bar',
                    title='Units Sold by Category',
                    container_width=container_width / 2,
                    height=chart_height,
                    color_scheme=selected_color_scheme
                )
                st.altair_chart(units_bar, use_container_width=True)
            
            with col2:
                units_pie = category_distribution_plot(
                    df=category_df,
                    category_column='category',
                    value_column='units_sum',
                    chart_type='pie',
                    title='Units Distribution',
                    container_width=container_width / 2,
                    height=chart_height,
                    color_scheme=selected_color_scheme
                )
                st.altair_chart(units_pie, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating category distribution charts: {str(e)}")
            logger.error(f"Chart creation error: {str(e)}")
        
        # Error handling examples
        st.header("🐛 Error Handling")
        with st.expander("View error handling examples"):
            error_tabs = st.tabs(["Empty Data", "Missing Columns", "Invalid Types"])
            
            with error_tabs[0]:
                st.markdown("Empty DataFrame example:")
                empty_df = pd.DataFrame()
                try:
                    error_chart = sales_trend_plot(
                        df=empty_df, 
                        container_width=container_width
                    )
                    st.altair_chart(error_chart)
                except Exception as e:
                    st.error(f"Expected error: {str(e)}")
            
            with error_tabs[1]:
                st.markdown("Missing required columns example:")
                bad_df = sales_df.drop(columns=['date']).copy()
                try:
                    error_chart = sales_trend_plot(
                        df=bad_df, 
                        container_width=container_width
                    )
                    st.altair_chart(error_chart)
                except Exception as e:
                    st.error(f"Expected error: {str(e)}")
            
            with error_tabs[2]:
                st.markdown("Invalid data types example:")
                invalid_df = sales_df.copy()
                invalid_df['date'] = 'invalid'
                try:
                    error_chart = sales_trend_plot(
                        df=invalid_df, 
                        container_width=container_width
                    )
                    st.altair_chart(error_chart)
                except Exception as e:
                    st.error(f"Expected error: {str(e)}")
        
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
        logger.error(f"Unexpected error in main: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the logs for more details.")


if __name__ == "__main__":
    main()
