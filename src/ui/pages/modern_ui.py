"""
Modern UI implementation for Watchdog AI.
Provides a streamlined, modern interface for data analysis.
"""

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import os
from PIL import Image

from ..components.data_upload_enhanced import DataUploadManager
from ..components.flag_panel import render_flag_summary
from ..components.dashboard import (
    render_dashboard_from_insight,
    render_kpi_metrics,
    render_sales_dashboard,
    render_inventory_dashboard
)
from ..components.chat_interface import ChatInterface
from ..components.insight_generator import InsightGenerator
from ..components.system_connect import render_system_connect

def get_logo():
    """Load and return the logo image."""
    try:
        # Get the absolute path to the assets directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        logo_path = os.path.join(project_root, "assets", "watchdog_logo.png")
        
        if not os.path.exists(logo_path):
            print(f"[ERROR] Logo not found at: {logo_path}")
            return None
            
        return Image.open(logo_path)
    except Exception as e:
        print(f"[ERROR] Failed to load logo: {str(e)}")
        return None

def _initialize_session_state():
    """Initialize session state variables."""
    if 'validated_data' not in st.session_state:
        st.session_state.validated_data = {}
    if 'active_upload' not in st.session_state:
        st.session_state.active_upload = None
    if 'validation_summary' not in st.session_state:
        st.session_state.validation_summary = {
            'flag_counts': {},
            'total_records': 0,
            'total_issues': 0,
            'percentage_clean': 100
        }
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = None
    if 'current_insight' not in st.session_state:
        st.session_state.current_insight = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'upload_metadata' not in st.session_state:
        st.session_state.upload_metadata = None

def modern_analyst_ui() -> None:
    """Render the modern analyst UI interface."""
    # Initialize session state
    _initialize_session_state()
    
    # Page config
    st.set_page_config(
        page_title="Watchdog AI",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize components
    chat_interface = ChatInterface()
    insight_generator = InsightGenerator()
    data_upload_manager = DataUploadManager()
    
    # Custom CSS for modern look
    st.markdown("""
        <style>
        /* Modern theme colors */
        :root {
            --primary: #0A84FF;
            --background: #1A1A1A;
            --surface: #252526;
            --text: #D4D4D4;
            --accent: #0066cc;
        }
        
        /* Base styles */
        .stApp {
            background-color: var(--background);
            color: var(--text);
        }
        
        /* Header styling */
        .header-text {
            margin: 0;
            padding: 0;
        }
        
        .header-text h1 {
            color: #E0E0E0;
            font-size: 2.5rem;
            margin: 0;
            line-height: 1.2;
        }
        
        .header-text p {
            color: #00FFFF;
            font-size: 1rem;
            margin: 4px 0 0;
        }
        
        /* Center the columns container */
        [data-testid="column-container"] {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        
        /* General UI styles from original code */
        .header { color: #00FF00; font-size: 24px; font-family: 'Arial', sans-serif; margin-top: 20px; }
        .text { color: #FFFFFF; font-size: 16px; font-family: 'Arial', sans-serif; }
        .insight-card { background-color: #1A2526; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
        .button { background-color: #00FF00; color: #000000; border-radius: 5px; padding: 10px; }
        .button:hover { background-color: #00FFFF; }

        /* Streamlit specific overrides if needed */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px; /* Add gap between tabs */
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            white-space: pre-wrap;
            background-color: #252526; /* Match card background */
            border-radius: 4px 4px 0 0;
            color: #AAA; /* Muted tab text */
        }
        .stTabs [aria-selected="true"] {
             background-color: #1A2526; /* Darker selected tab */
             color: #00FFFF; /* Accent color for selected tab text */
        }

        /* Ensure columns align items at the top/start for header */
        [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
           display: flex;
           flex-direction: column;
           align-items: flex-start; /* Align items to start */
        }

        /* Styles from Ticket Step 3 - Moved here as theme.py modification is complex */
        /* Center page blocks */
        /* .block-container {
          display: flex;
          flex-direction: column;
          align-items: center;
        } */ /* Commented out - potentially too broad */
        
        /* Adjust default Streamlit column gaps */
        /* .css-1lcbmhc .css-1rr5i01 { gap: 24px !important; } */ /* Commented out - too specific/brittle */

        </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and text using Streamlit columns
    header_cols = st.columns([1, 4], gap="small")
    
    # Logo column
    with header_cols[0]:
        logo = get_logo()
        if logo:
            st.image(logo, width=180, use_container_width=False)
        else:
            # Fallback if logo not found
            st.markdown("""
                <div style='background: #252526; width: 180px; height: 180px; display: flex; align-items: center; justify-content: center; border-radius: 8px;'>
                    <span style='font-size: 48px;'>🐾</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Text column
    with header_cols[1]:
        st.markdown("""
            <div class='header-text'>
                <h1>Watchdog AI</h1>
                <p>Intelligent Dealership Analytics</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    with st.container():
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs([
            "Data & Insights",
            "Chat Analysis",
            "System Connect"
        ])
        
        # Data & Insights Tab
        with tab1:
            st.markdown('<div class="header">Upload Your Data</div>', unsafe_allow_html=True)
            st.markdown('<div class="text">Drag and drop CSV or Excel file to begin analysis</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'], help="Limit 200MB per file • CSV, XLSX, XLS")

            if uploaded_file is not None:
                st.session_state['file_uploaded'] = True
                st.success("File uploaded successfully!")
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    st.session_state['dataframe'] = df # Store df in session state

                    # Data Preview
                    st.markdown('<div class="header">Data Preview</div>', unsafe_allow_html=True)
                    st.dataframe(df.head(), use_container_width=True)

                    # Column Information
                    st.markdown('<div class="header">Column Information</div>', unsafe_allow_html=True)
                    col_info = pd.DataFrame({
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.notnull().sum(),
                        'Null Count': df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
                except Exception as e:
                     st.error(f"Error reading file: {e}")
                     st.session_state['file_uploaded'] = False
                     if 'dataframe' in st.session_state: del st.session_state['dataframe']

        # Chat Analysis Tab
        with tab2:
            st.markdown('<div class="header">Chat Analysis</div>', unsafe_allow_html=True)
            user_input = st.text_input("Ask a question about your data", placeholder="e.g., What lead source had the lowest total gross?", key="chat_input")
            col1, col2 = st.columns([3, 1])
            with col1:
                generate_clicked = st.button("Generate Insight", key="generate")
            with col2:
                clear_clicked = st.button("Clear", key="clear")

            if clear_clicked:
                st.session_state['user_input'] = ""
                st.rerun()

            # Generate Insights from Uploaded Data
            if generate_clicked and user_input:
                if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
                    st.markdown('<div class="header">Analysis Results</div>', unsafe_allow_html=True)
                    df = st.session_state['dataframe']
                    
                    # --- Enhanced Insight Logic ---
                    # Look for gross and lead source columns with flexible naming
                    gross_cols = [col for col in df.columns if 'gross' in col.lower()]
                    lead_cols = [col for col in df.columns if 'lead' in col.lower() or 'source' in col.lower()]
                    rep_cols = [col for col in df.columns if any(term in col.lower() for term in ['rep', 'sales', 'salesperson'])]
                    
                    try:
                        user_query = user_input.lower()
                        
                        if "lowest total gross" in user_query:
                            # Use the first matching column names found
                            gross_col = gross_cols[0]
                            lead_col = lead_cols[0]
                            
                            # Clean and convert gross values
                            raw_gross = df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True)
                            df[gross_col] = pd.to_numeric(raw_gross, errors='coerce')
                            
                            # Drop rows where gross couldn't be converted
                            df_numeric = df.dropna(subset=[gross_col])
                            
                            if not df_numeric.empty:
                                lead_gross = df_numeric.groupby(lead_col)[gross_col].sum().reset_index()
                                if not lead_gross.empty:
                                    lowest_gross_lead = lead_gross.loc[lead_gross[gross_col].idxmin()]
                                    st.markdown(f'<div class="insight-card">'
                                                f'<b>Lead Source with Lowest Total Gross</b><br><br>'
                                                f'• Lead Source: {lowest_gross_lead[lead_col]}<br>'
                                                f'• Total Gross: ${lowest_gross_lead[gross_col]:,.2f}<br><br>'
                                                f'<u>Action Items:</u><br>'
                                                f'• Review lead generation strategies for {lowest_gross_lead[lead_col]} to improve performance.<br>'
                                                f'• Consider reallocating resources to higher-performing lead sources.'
                                                f'</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("Could not group by lead source or calculate sum.")
                            else:
                                st.error(
                                    f"No valid numeric data found in '{gross_col}' column. "
                                    f"Sample values: {df[gross_col].head().tolist()}"
                                )
                        elif any(term in user_query for term in ["highest gross", "top gross"]) and any(term in user_query for term in ["sales rep", "rep", "salesperson"]):
                            if gross_cols and rep_cols:
                                # Use the first matching column names found
                                gross_col = gross_cols[0]
                                rep_col = rep_cols[0]
                                
                                # Clean and convert gross values
                                raw_gross = df[gross_col].astype(str).str.replace(r'[\$,]', '', regex=True)
                                df[gross_col] = pd.to_numeric(raw_gross, errors='coerce')
                                
                                # Drop rows where gross couldn't be converted
                                df_numeric = df.dropna(subset=[gross_col])
                                
                                if not df_numeric.empty:
                                    rep_gross = df_numeric.groupby(rep_col)[gross_col].sum().reset_index()
                                    if not rep_gross.empty:
                                        top_gross_rep = rep_gross.loc[rep_gross[gross_col].idxmax()]
                                        st.markdown(f'<div class="insight-card">'
                                                    f'<b>Sales Rep with Highest Total Gross</b><br><br>'
                                                    f'• Sales Rep: {top_gross_rep[rep_col]}<br>'
                                                    f'• Total Gross: ${top_gross_rep[gross_col]:,.2f}<br><br>'
                                                    f'<u>Action Items:</u><br>'
                                                    f'• Study {top_gross_rep[rep_col]}\'s sales strategies for team training.<br>'
                                                    f'• Analyze their lead source performance for optimization.<br>'
                                                    f'• Consider having them mentor other team members.'
                                                    f'</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("Could not group by sales rep or calculate sum.")
                            else:
                                missing = []
                                if not gross_cols:
                                    missing.append("gross profit")
                                if not rep_cols:
                                    missing.append("sales rep")
                                st.error(f"Could not find required columns for: {', '.join(missing)}. Available columns: {list(df.columns)}")
                        else:
                            st.info("Try asking about 'highest gross by sales rep' or 'lowest total gross' to see example insights.")
                        
                    except Exception as e:
                        st.error(
                            f"Error analyzing data: {str(e)}. "
                            f"Found columns: {list(df.columns)}. "
                            f"Gross columns found: {gross_cols}. "
                            f"Rep columns found: {rep_cols}."
                        )

                elif 'file_uploaded' not in st.session_state or not st.session_state['file_uploaded']:
                    st.warning('Please upload a data file first on the "Data & Insights" tab.')
                else:
                    st.warning("Could not find the DataFrame in session state. Please re-upload the file.")
            elif not user_input and generate_clicked:
                st.info("Please enter a question to generate an insight.")
            else:
                if 'file_uploaded' in st.session_state and st.session_state['file_uploaded']:
                    st.markdown('<div class="text">Ask a question to get started!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="text">Upload a file on the "Data & Insights" tab to enable chat analysis.</div>', unsafe_allow_html=True)

        # System Connect Tab
        with tab3:
            with st.expander("Connect My Systems"):
                st.markdown('<div class="text">Placeholder for system integration (future development).</div>', unsafe_allow_html=True)

    # Session state management / cleanup (ensure keys exist before use)
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    if 'dataframe' not in st.session_state:
        st.session_state['dataframe'] = None

if __name__ == "__main__":
    modern_analyst_ui()