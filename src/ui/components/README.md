# UI Components for Watchdog AI

This directory contains reusable UI components for building Watchdog AI dashboards and interactive interfaces.

## Components

### Flag Panel

The `flag_panel.py` module provides a rich UI panel for visualizing data quality issues identified by the insight validator. It creates an interactive dashboard that allows users to:

- View a summary of data quality metrics
- Explore specific data quality issues
- Apply data cleaning operations
- Send cleaned data to the insight engine

#### Key Features

- **Data Quality Assessment**: Shows overall data quality metrics and issue counts
- **Interactive Issue Exploration**: Expandable sections for each type of data issue
- **Visualization**: Charts displaying the distribution of clean vs. problematic data
- **Data Cleaning Options**: Interactive controls for applying cleaning operations
- **Detailed Reporting**: Markdown-formatted detailed reports
- **Streamlit Integration**: Built for seamless integration with Streamlit apps

#### Usage Example

```python
import pandas as pd
import streamlit as st
from watchdog_ai.ui.components.flag_panel import render_flag_summary

# Load your data
df = pd.read_csv("dealership_data.csv")

# Define a callback function for when the clean button is clicked
def on_data_cleaned(cleaned_df):
    st.session_state['cleaned_data'] = cleaned_df
    st.success("Data cleaning operations applied!")

# Render the flag panel
cleaned_df, was_cleaned = render_flag_summary(df, on_clean_click=on_data_cleaned)

# Use the cleaned data for further analysis if it was cleaned
if was_cleaned:
    # Continue with analysis on cleaned_df
    pass
```

#### Compact Metrics View

For dashboards where space is limited, you can use the compact metrics view:

```python
from watchdog_ai.ui.components.flag_panel import render_flag_metrics

# Render just the metrics section
render_flag_metrics(df)
```

## Integration with Insight Validator

The Flag Panel component integrates directly with the Insight Validator module, allowing for a seamless flow from data validation to visualization to action:

1. **Validation**: The Insight Validator analyzes the data and flags issues
2. **Visualization**: The Flag Panel displays these issues in a user-friendly format
3. **Cleaning**: The user interactively selects cleaning operations
4. **Action**: Cleaned data is passed to the Insight Engine for further analysis
