"""
Example script demonstrating the Flag Panel UI component for Watchdog AI.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the flag_panel module
from ..src.ui.components.flag_panel import render_flag_summary, render_flag_metrics
from ..src.validators.insight_validator import flag_all_issues, summarize_flags

def main():
    """Main function to demonstrate the flag panel UI component."""
    st.set_page_config(page_title="Watchdog AI - Flag Panel Demo", layout="wide")
    st.title("Watchdog AI - Data Quality Panel Demo")
    st.write("This demo shows how the flag panel component can be used to visualize and address data quality issues.")
    
    # Get the path to the sample data
    sample_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'tests',
        'assets',
        'sample_dealership_data.csv'
    )
    
    # Check if the file exists
    if not os.path.exists(sample_data_path):
        # Use sample data from code if file doesn't exist
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
        st.info("Using built-in sample data since sample CSV file was not found.")
    else:
        # Load the sample data
        df = pd.read_csv(sample_data_path)
        st.success(f"Loaded sample data from: {sample_data_path}")
    
    # Show the original data
    st.subheader("Original Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Add a separator
    st.write("---")
    
    # Define a callback function for when the clean button is clicked
    def on_data_cleaned(cleaned_df):
        st.session_state['cleaned_data'] = cleaned_df
        st.session_state['insight_ready'] = True
    
    # Render the flag panel with the callback
    cleaned_df, was_cleaned = render_flag_summary(df, on_clean_click=on_data_cleaned)
    
    # Display the cleaned data if it was cleaned
    if was_cleaned or st.session_state.get('insight_ready', False):
        display_df = st.session_state.get('cleaned_data', cleaned_df)
        
        st.subheader("Data Ready for Insight Engine")
        st.dataframe(display_df, use_container_width=True)
        
        # Show some analytics options
        st.subheader("Ready for Analysis")
        st.write("Now that the data has been cleaned, it's ready for deeper analysis.")
        
        analysis_options = st.multiselect(
            "Select analysis options:",
            ["Sales Performance", "Inventory Trends", "Lead Source ROI", "Salesperson Performance"]
        )
        
        if analysis_options:
            st.info(f"Selected analyses: {', '.join(analysis_options)}. In a full implementation, this would trigger the selected analysis modules.")
            
            # Add a button to run the analysis
            if st.button("Run Selected Analyses"):
                st.success("Analysis complete! Check the Insights Dashboard for results.")
                
                # Show a placeholder for the insights dashboard
                st.subheader("Insights Dashboard Preview")
                
                # Create compact metrics view
                render_flag_metrics(display_df)
                
                # Add some placeholder charts
                import numpy as np
                import matplotlib.pyplot as plt
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Placeholder chart 1
                    fig, ax = plt.subplots()
                    makes = display_df['Make'].value_counts()
                    ax.pie(makes, labels=makes.index, autopct='%1.1f%%')
                    ax.set_title('Sales by Make')
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    # Placeholder chart 2
                    fig, ax = plt.subplots()
                    x = np.array(['Jan', 'Feb', 'Mar', 'Apr'])
                    y = np.array([3, 5, 4, 6])
                    ax.bar(x, y)
                    ax.set_title('Monthly Sales Trend')
                    st.pyplot(fig, use_container_width=True)

if __name__ == "__main__":
    # Initialize session state if needed
    if 'insight_ready' not in st.session_state:
        st.session_state['insight_ready'] = False
    
    if 'cleaned_data' not in st.session_state:
        st.session_state['cleaned_data'] = None
    
    main()
