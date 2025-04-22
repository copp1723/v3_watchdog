"""Sidebar component for the Watchdog AI UI."""

import streamlit as st
import pandas as pd

def render_sidebar():
    """Render the application sidebar with settings and tools."""
    with st.sidebar:
        st.header("⚙️ Settings & Tools")
        
        # Model settings
        st.subheader("Model Settings")
        model = st.selectbox(
            "LLM Model",
            ["gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower values = more deterministic responses"
        )
        
        # Apply model settings button
        if st.button("Apply Settings"):
            # Store settings in session state
            st.session_state.model_settings = {
                "model": model,
                "temperature": temperature
            }
            st.success("Settings applied!")
        
        # Sample datasets
        st.subheader("Sample Datasets")
        sample_sets = ["Sales Data", "Inventory Data", "Web Traffic Data"]
        selected_sample = st.selectbox("Load Sample Dataset", ["None"] + sample_sets)
        
        if selected_sample != "None" and st.button("Load Sample"):
            # Create a simple sample dataset
            if selected_sample == "Sales Data":
                df = pd.DataFrame({
                    "Date": pd.date_range(start="2023-01-01", periods=20),
                    "Product": ["Product A", "Product B"] * 10,
                    "Sales": [100, 200, 150, 300, 250, 400, 350, 450, 500, 600] * 2,
                    "Revenue": [1000, 2000, 1500, 3000, 2500, 4000, 3500, 4500, 5000, 6000] * 2
                })
                
                # Store in session state
                st.session_state.sample_data = df
                st.session_state.validated_data = df
                st.success(f"Loaded {selected_sample} successfully!")
            elif selected_sample == "Inventory Data":
                # Create a simple inventory dataset
                df = pd.DataFrame({
                    "ProductID": range(1, 21),
                    "ProductName": [f"Product {chr(65+i)}" for i in range(20)],
                    "Category": ["Electronics", "Clothing", "Food", "Books"] * 5,
                    "QuantityInStock": [100, 50, 75, 30, 45, 60, 25, 80, 95, 110, 40, 55, 70, 85, 90, 105, 120, 65, 35, 15],
                    "UnitPrice": [10.99, 19.99, 5.49, 12.99, 8.99, 15.99, 22.99, 7.49, 9.99, 14.99, 11.99, 17.99, 6.99, 13.49, 20.99, 4.99, 18.99, 16.49, 24.99, 13.99]
                })
                
                # Store in session state
                st.session_state.sample_data = df
                st.session_state.validated_data = df
                st.success(f"Loaded {selected_sample} successfully!")
        
        # Help section
        st.markdown("---")
        with st.expander("❓ Help & Information", expanded=False):
            st.markdown("""
            **Getting Started**
            1. Upload your data in the Data Upload tab
            2. Switch to the Analysis tab
            3. Ask questions about your data
            
            **Example Questions**
            - What are the top performing sales reps?
            - Show me revenue trends by month
            - What products have the lowest inventory?
            """)
            
        # About section
        st.markdown("---")
        st.caption("Watchdog AI v3.0 | © 2023-2025")