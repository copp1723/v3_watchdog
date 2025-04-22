"""
Integration example demonstrating all error handling, validation, and logging components.

This example shows how to use:
1. Global error handling decorator (@with_global_error_handling)
2. Validation status badges
3. Log management
4. Error surfacing in the UI

Run this file directly with Streamlit to see a complete working example:
streamlit run src/watchdog_ai/examples/integration_example.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import time
import os
import logging
import random
from datetime import datetime
import traceback

# Import our components
from watchdog_ai.utils.global_error_handler import (
    with_global_error_handling,
    global_try_except,
    handle_api_errors
)
from watchdog_ai.ui.components.validation_status_badge import (
    ValidationStatusBadge, 
    create_validation_badge,
    render_validation_summary,
    render_flag_column_badges
)
from watchdog_ai.ui.components.log_manager import (
    LogManager,
    render_settings_page_logs_section
)
from watchdog_ai.core.config.logging import setup_logging, LOG_DIR
from src.utils.errors import APIError  # Import custom error classes

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
setup_logging(
    log_level="INFO",
    log_file=os.path.join(LOG_DIR, "integration_example.log"),
    max_bytes=1048576,  # 1MB
    backup_count=3
)
logger = logging.getLogger("integration_example")


#
# Define example processing functions with error handling
#
@with_global_error_handling(friendly_message="Error during data loading")
def load_sample_data(rows=100, error_probability=0):
    """
    Load sample data with optional error injection.
    
    Args:
        rows: Number of rows to generate
        error_probability: Probability of injecting an error (0-1)
        
    Returns:
        pandas DataFrame with sample data
    """
    logger.info(f"Loading sample data with {rows} rows")
    
    # Possibly inject an error
    if random.random() < error_probability:
        logger.warning("Injecting a random error in data loading")
        error_type = random.choice([
            "value_error", "key_error", "type_error", "index_error", "zero_division"
        ])
        
        if error_type == "value_error":
            raise ValueError("Invalid value in data generation")
        elif error_type == "key_error":
            raise KeyError("Missing key in data dictionary")
        elif error_type == "type_error":
            raise TypeError("Incorrect data type for operation")
        elif error_type == "index_error":
            raise IndexError("Index out of bounds in data array")
        elif error_type == "zero_division":
            return 1 / 0  # ZeroDivisionError
    
    # Generate sample data
    data = {
        'ID': range(1, rows + 1),
        'Name': [f"Item {i}" for i in range(1, rows + 1)],
        'Value': np.random.normal(100, 25, rows),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], rows),
        'Date': [datetime.now().strftime("%Y-%m-%d") for _ in range(rows)],
    }
    
    # Add some null values for validation testing
    for i in range(int(rows * 0.1)):  # 10% of rows
        idx = random.randint(0, rows - 1)
        col = random.choice(['Name', 'Value', 'Category'])
        data[col][idx] = None if col != 'Value' else np.nan
    
    # Add some invalid values
    for i in range(int(rows * 0.05)):  # 5% of rows
        idx = random.randint(0, rows - 1)
        data['Value'][idx] = -1  # Negative values (for demo validation)
    
    df = pd.DataFrame(data)
    logger.info(f"Successfully generated DataFrame with shape {df.shape}")
    
    return df


@global_try_except(friendly_message="Error during data processing")
def process_data(df, operation=None):
    """
    Process data with various operations.
    
    Args:
        df: Input DataFrame
        operation: Processing operation to perform
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Processing data with operation: {operation}")
    
    if df is None or df.empty:
        logger.error("Cannot process empty DataFrame")
        raise ValueError("DataFrame is empty or None")
    
    if operation == "normalize":
        # Normalize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'ID':  # Skip ID column
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                    
    elif operation == "filter_invalid":
        # Filter out rows with negative values
        df = df[df['Value'] >= 0]
        
    elif operation == "calculate_metrics":
        # Add new calculated columns
        df['Value_Squared'] = df['Value'] ** 2
        df['Category_Code'] = df['Category'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        
    elif operation == "error_operation":
        # Intentionally cause an error
        logger.warning("Executing operation known to cause errors")
        # Try to access a non-existent column
        df['NonExistentColumn'] = df['Value'] / 0
    
    else:
        logger.warning(f"Unknown operation: {operation}, returning original data")
    
    logger.info(f"Processing complete, resulting shape: {df.shape}")
    return df


@handle_api_errors(friendly_message="API Error")
def simulate_api_call(data_id, should_fail=False):
    """
    Simulate an API call that might fail.
    
    Args:
        data_id: ID for the data to fetch
        should_fail: Whether the call should fail
        
    Returns:
        Response data dictionary
    """
    logger.info(f"Making API call for data_id: {data_id}")
    
    if should_fail:
        logger.error(f"API call failure for data_id: {data_id}")
        raise APIError(f"Failed to retrieve data for ID: {data_id}")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Return mock data
    return {
        "id": data_id,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "data": {
            "value": random.randint(1, 1000),
            "name": f"API Result {data_id}",
            "metadata": {
                "source": "integration_example",
                "version": "1.0.0"
            }
        }
    }


#
# Data validation functions
#
def validate_data(df):
    """
    Validate data and return validation results.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating data")
    
    if df is None or df.empty:
        return {
            "status": "error",
            "message": "No data to validate",
            "errors": [{
                "type": "empty_data",
                "message": "The DataFrame is empty or None"
            }],
            "warnings": []
        }
    
    # Initialize results
    errors = []
    warnings = []
    
    # Check for null values
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        error_msg = f"Found null values in {len(columns_with_nulls)} columns"
        errors.append({
            "type": "null_values",
            "message": error_msg,
            "columns": columns_with_nulls.to_dict(),
            "rule": "no_nulls_allowed",
            "invalid_rows": [
                {"row": i, "message": f"Null value in column {col}"} 
                for col in columns_with_nulls.index 
                for i in df[df[col].isnull()].index
            ]
        })
        logger.warning(error_msg)
    
    # Check for negative values
    if 'Value' in df.columns:
        negative_values = df[df['Value'] < 0]
        if not negative_values.empty:
            warning_msg = f"Found {len(negative_values)} rows with negative values"
            warnings.append({
                "type": "negative_values",
                "message": warning_msg,
                "rule": "positive_values",
                "invalid_rows": [
                    {"row": i, "message": f"Negative value: {df.loc[i, 'Value']}"} 
                    for i in negative_values.index
                ]
            })
            logger.warning(warning_msg)
    
    # Check for category distribution
    if 'Category' in df.columns:
        category_counts = df['Category'].value_counts()
        min_category_count = category_counts.min()
        if min_category_count < len(df) * 0.1:  # Less than 10% of data
            warnings.append({
                "type": "unbalanced_categories",
                "message": f"Category distribution is unbalanced. Minimum count: {min_category_count}",
                "rule": "balanced_categories",
                "invalid_rows": []
            })
    
    # Add flag columns for visualization
    result_df = df.copy()
    
    # Add flag for null values
    result_df['flag_null_values'] = df.isnull().any(axis=1)
    
    # Add flag for negative values
    if 'Value' in df.columns:
        result_df['flag_negative_values'] = df['Value'] < 0
    
    # Determine overall status
    status = "success"
    if warnings:
        status = "warning"
    if errors:
        status = "error"
    
    # Log validation results
    logger.info(f"Validation complete: status={status}, {len(errors)} errors, {len(warnings)} warnings")
    
    return {
        "status": status,
        "message": f"Validation found {len(errors)} errors and {len(warnings)} warnings",
        "errors": errors,
        "warnings": warnings,
        "validated_df": result_df
    }


#
# UI Components
#
def render_data_section():
    """Render the data generation and processing section."""
    st.header("1. Data Generation & Processing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rows = st.number_input("Number of rows", min_value=10, max_value=1000, value=100, step=10)
    
    with col2:
        error_prob = st.slider("Error probability", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    with col3:
        operation = st.selectbox(
            "Processing operation",
            options=[
                "none", 
                "normalize", 
                "filter_invalid", 
                "calculate_metrics", 
                "error_operation"
            ]
        )
    
    # Generate button
    if st.button("Generate & Process Data"):
        with st.spinner("Generating data..."):
            # Step 1: Load data (with possible errors)
            result = load_sample_data(rows=rows, error_probability=error_prob)
            
            if isinstance(result, dict) and "error_type" in result:
                # An error occurred, result contains error details
                st.error(f"Error during data generation: {result['message']}")
                if "ui_feedback" in result:
                    st.write(result["ui_feedback"]["content"]["message"])
                # Log errors to help debug
                logger.error(f"Data generation error: {result['message']}")
                
                # Store nothing in session state
                st.session_state.df = None
                st.session_state.validation_results = None
            else:
                # Result is a DataFrame
                df = result
                st.session_state.df = df
                
                # Store original data for display
                st.session_state.original_df = df.copy()
                
                # Show success message
                st.success(f"Successfully generated {len(df)} rows of data")
                
                # Step 2: Process data if requested
                if operation != "none":
                    with st.spinner(f"Processing data with {operation}..."):
                        process_result = process_data(df, operation=operation)
                        
                        if isinstance(process_result, dict) and "error_type" in process_result:
                            # An error occurred during processing
                            st.error(f"Error during data processing: {process_result['message']}")
                            if "ui_feedback" in process_result:
                                st.write(process_result["ui_feedback"]["content"]["message"])
                            # Keep original data
                            st.session_state.df = df
                        else:
                            # Processing successful
                            st.session_state.df = process_result
                            st.success(f"Data processing with '{operation}' completed successfully")
                
                # Step 3: Validate data
                with st.spinner("Validating data..."):
                    validation_results = validate_data(st.session_state.df)
                    st.session_state.validation_results = validation_results
                    
                    # Update df to the flagged version
                    if "validated_df" in validation_results:
                        st.session_state.df = validation_results["validated_df"]
    
    # Display data if available
    if "df" in st.session_state and st.session_state.df is not None:
        st.subheader("Generated Data")
        st.dataframe(st.session_state.df, use_container_width=True)


def render_validation_section():
    """Render the validation results section with badges."""
    st.header("2. Validation Results")
    
    if "validation_results" not in st.session_state or st.session_state.validation_results is None:
        st.info("Generate data first to see validation results")
        return
    
    # Get validation results
    validation_results = st.session_state.validation_results
    
    # Render validation summary badges
    render_validation_summary(validation_results)
    
    # Display more detailed validation information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Validation Details")
        
        # Display flag columns
        if "validated_df" in validation_results:
            flag_columns = [col for col in validation_results["validated_df"].columns if col.startswith("flag_")]
            for flag_col in flag_columns:
                df = validation_results["validated_df"]
                flag_count = int(df[flag_col].sum())
                flag_name = flag_col.replace("flag_", "").replace("_", " ").title()
                
                if flag_count > 0:
                    status = "error"
                    message = f"{flag_name}: {flag_count} issues found"
                    details = [f"Row {i}: {flag_col} issue" for i in df[df[flag_col]].index[:5]]
                    
                    # Add ellipsis if there are more than 5 issues
                    if flag_count > 5:
                        details.append(f"... and {flag_count - 5} more")
                    
                    # Create a badge for this flag
                    create_validation_badge(
                        status=status,
                        message=message,
                        details=details,
                        count=flag_count
                    )
    
    with col2:
        st.subheader("Issue Summary")
        
        # Count issues by type
        error_count = sum(len(e.get("invalid_rows", [])) for e in validation_results.get("errors", []))
        warning_count = sum(len(w.get("invalid_rows", [])) for w in validation_results.get("warnings", []))
        
        # Show metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            df = validation_results.get("validated_df", pd.DataFrame())
            st.metric("Total Rows", len(df) if not df.empty else 0)
        
        with metrics_col2:
            st.metric("Errors", error_count, delta=-error_count, delta_color="inverse")
        
        with metrics_col3:
            st.metric("Warnings", warning_count, delta=-warning_count, delta_color="inverse")
        
        # Show flag column distribution
        if "validated_df" in validation_results:
            df = validation_results["validated_df"]
            flag_columns = [col for col in df.columns if col.startswith("flag_")]
            
            if flag_columns:
                flag_stats = {}
                for flag_col in flag_columns:
                    flag_stats[flag_col.replace("flag_", "").replace("_", " ").title()] = int(df[flag_col].sum())
                
                # Create a bar chart of flag counts
                if flag_stats:
                    st.write("### Issues by Type")
                    flag_df = pd.DataFrame({'Issue Type': list(flag_stats.keys()), 'Count': list(flag_stats.values())})
                    st.bar_chart(flag_df.set_index('Issue Type'))


def render_api_section():
    """Render the API simulation section to demonstrate error handling."""
    st.header("3. API Simulation")
    st.write("This section demonstrates API error handling with the `@handle_api_errors` decorator.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_id = st.text_input("Data ID", value="12345")
    
    with col2:
        should_fail = st.checkbox("Simulate API Error")
    
    if st.button("Make API Call"):
        with st.spinner("Calling API..."):
            try:
                # Make API call - this uses the @handle_api_errors decorator
                result = simulate_api_call(data_id, should_fail=should_fail)
                
                # Check if result is an error response
                if isinstance(result, dict) and result.get("status") == "error":
                    st.error(f"API Error: {result.get('message', 'Unknown error')}")
                    st.json(result)
                    # Add error badge
                    create_validation_badge(
                        status="error",
                        message=f"API Error: {result.get('error_code', 'UNKNOWN_ERROR')}",
                        details=[result.get("message", "Unknown error")]
                    )
                else:
                    # Success - show result
                    st.success("API call successful!")
                    st.json(result)
                    # Add success badge
                    create_validation_badge(
                        status="success",
                        message="API Request Completed",
                        details=[f"Data ID: {data_id}", f"Timestamp: {result.get('timestamp')}"]
                    )
            except Exception as e:
                # This shouldn't happen due to the decorator, but just in case
                st.error(f"Unexpected error: {str(e)}")
                logger.exception("Unexpected error in API simulation")


def render_logs_section():
    """Render the logs section to demonstrate log management."""
    st.header("4. Log Management")
    
    # Display log management interface
    # Use the simplified view for settings page
    render_settings_page_logs_section()
    
    # Add logging examples
    st.subheader("Generate Example Logs")
    st.write("Click the buttons below to generate different types of log entries.")
    
    log_col1, log_col2, log_col3 = st.columns(3)
    
    with log_col1:
        if st.button("Log Info Message"):
            logger.info(f"Info log message generated at {datetime.now().isoformat()}")
            st.success("Info message logged")
    
    with log_col2:
        if st.button("Log Warning Message"):
            logger.warning(f"Warning log message generated at {datetime.now().isoformat()}")
            st.warning("Warning message logged")
    
    with log_col3:
        if st.button("Log Error Message"):
            try:
                # Simulate an error
                raise ValueError("Simulated error for logging demonstration")
            except Exception as e:
                logger.error(f"Error log message: {str(e)}", exc_info=True)
                st.error("Error message logged with traceback")


def render_documentation():
    """Render documentation and usage instructions."""
    with st.expander("📚 Documentation & Usage Instructions", expanded=False):
        st.markdown("""
        # Watchdog AI Error Handling & Logging Demo
        
        This example demonstrates the integration of three key components:
        
        1. **Global Error Handling** - Using decorators to catch and handle exceptions consistently
        2. **Validation Badges** - Visual feedback for data validation issues
        3. **Log Management** - Centralized logging with file rotation and UI access
        
        ## Key Components Used
        
        ### Error Handling Decorators
        
        Three decorators are demonstrated:
        
        - `@with_global_error_handling` - Basic decorator for error handling
        - `@global_try_except` - Simplified syntax that works with or without arguments
        - `@handle_api_errors` - Specialized for API endpoints
        
        ```python
        @with_global_error_handling(friendly_message="Custom error message")
        def my_function():
            # Function code
        ```
        
        ### Validation Badges
        
        Badges for showing validation status with three levels:
        
        - **Success** (green) - No issues found
        - **Warning** (yellow) - Minor issues found
        - **Error** (red) - Critical issues found
        
        ### Log Management
        
        Features demonstrated:
        
        - Log rotation
        - Log level management
        - UI access to logs
        """)
def main():
    """Main function to render the Streamlit app."""
    st.set_page_config(
        page_title="Watchdog AI Integration Example",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Watchdog AI Integration Example")
    st.write("This example demonstrates error handling, validation, and logging components.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Processing", 
        "Validation", 
        "API Simulation", 
        "Logs", 
        "Documentation"
    ])
    
    with tab1:
        render_data_section()
    
    with tab2:
        render_validation_section()
    
    with tab3:
        render_api_section()
    
    with tab4:
        render_logs_section()
    
    with tab5:
        render_documentation()


if __name__ == "__main__":
    main()
