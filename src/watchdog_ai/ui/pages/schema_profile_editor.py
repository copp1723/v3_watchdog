"""
Schema profile editor UI component with benchmarking capabilities.
"""

import streamlit as st
import json
import pandas as pd
import altair as alt
from typing import Dict, List, Any, Optional
from datetime import datetime

from validators.schema_manager import SchemaProfileManager
from insights.benchmarking import BenchmarkEngine

class SchemaProfileEditor:
    """UI component for editing schema profiles with benchmarking."""
    
    def __init__(self, schema_manager: Optional[SchemaProfileManager] = None):
        """Initialize the editor."""
        self.schema_manager = schema_manager or SchemaProfileManager()
        self.benchmark_engine = BenchmarkEngine()
    
    def render(self) -> None:
        """Render the schema profile editor."""
        st.title("Schema Profile Editor")
        
        # Get list of profiles
        profiles = self.schema_manager.list_profiles()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Edit Profile", "Create Profile", "Preview", "Benchmarks"
        ])
        
        with tab1:
            self._render_edit_profile(profiles)
        
        with tab2:
            self._render_create_profile()
        
        with tab3:
            self._render_preview()
            
        with tab4:
            self._render_benchmarks()
    
    def _render_benchmarks(self) -> None:
        """Render the benchmarking interface."""
        st.header("Schema Benchmarks")
        
        # Dealership selector
        dealership_id = st.text_input("Dealership ID")
        
        # Comparison group
        st.subheader("Comparison Group")
        comparison_type = st.radio(
            "Compare Against:",
            ["All Dealerships", "Custom Group", "Similar Size", "Same Region"]
        )
        
        comparison_group = None
        if comparison_type == "Custom Group":
            comparison_group = st.multiselect(
                "Select Dealerships",
                ["d1", "d2", "d3", "d4", "d5"]  # Would be dynamic in production
            )
        
        # Time period
        st.subheader("Time Period")
        period = st.selectbox(
            "Comparison Period",
            ["30d", "90d", "180d", "365d"],
            index=0
        )
        
        # Metrics
        st.subheader("Metrics")
        metrics = st.multiselect(
            "Select Metrics to Compare",
            [
                "closing_rate",
                "avg_gross_profit",
                "lead_response_time",
                "inventory_turn_rate",
                "customer_satisfaction"
            ]
        )
        
        if st.button("Calculate Benchmarks") and dealership_id and metrics:
            try:
                # Get benchmark results
                results = self.benchmark_engine.calculate_benchmarks(
                    dealership_id=dealership_id,
                    metrics=metrics,
                    comparison_group=comparison_group,
                    period=period
                )
                
                # Display results
                st.subheader("Benchmark Results")
                
                # Create metrics display
                cols = st.columns(len(results.metrics))
                for i, metric in enumerate(results.metrics):
                    with cols[i]:
                        st.metric(
                            metric.name,
                            f"{metric.value:.1f}",
                            f"{metric.trend:.1f}%" if metric.trend else None
                        )
                        
                        # Create percentile gauge chart
                        gauge_data = pd.DataFrame({
                            'value': [metric.percentile],
                            'color': ['#1f77b4']
                        })
                        
                        gauge_chart = alt.Chart(gauge_data).mark_arc(
                            innerRadius=50,
                            outerRadius=80,
                            startAngle=0,
                            endAngle=alt.expr.scale(
                                'value',
                                domain=[0, 100],
                                range=[0, 6.28]  # 2π
                            )
                        ).encode(
                            theta='value:Q',
                            color=alt.value('#1f77b4')
                        ).properties(
                            width=150,
                            height=150
                        )
                        
                        st.altair_chart(gauge_chart)
                        st.write(f"Percentile: {metric.percentile:.1f}%")
                
                # Display improvement targets
                st.subheader("Improvement Targets")
                targets = self.benchmark_engine.get_improvement_targets(results)
                if targets:
                    target_df = pd.DataFrame([
                        {"Metric": k, "Improvement Needed": f"{v:.1f}"}
                        for k, v in targets.items()
                    ])
                    st.table(target_df)
                else:
                    st.success("All metrics are performing above the 75th percentile!")
                
                # Display anomalies
                anomalies = self.benchmark_engine.detect_anomalies(results)
                if anomalies:
                    st.subheader("Detected Anomalies")
                    for anomaly in anomalies:
                        st.warning(anomaly["description"])
                
            except Exception as e:
                st.error(f"Error calculating benchmarks: {str(e)}")
    
    def _render_edit_profile(self, profiles: List[Dict[str, Any]]) -> None:
        """
        Render the profile editing interface.
        
        Args:
            profiles: List of available profiles
        """
        st.header("Edit Schema Profile")
        
        # Profile selector
        profile_names = [p['name'] for p in profiles]
        selected_profile = st.selectbox(
            "Select Profile",
            options=profile_names
        )
        
        if selected_profile:
            try:
                # Load the profile
                profile = self.schema_manager.load_profile(selected_profile)
                
                # Show current profile as JSON
                st.subheader("Current Profile")
                st.json(profile)
                
                # Edit interface
                st.subheader("Edit Profile")
                
                # Use a code editor for JSON editing
                edited_json = st.text_area(
                    "Edit JSON",
                    value=json.dumps(profile, indent=2),
                    height=400
                )
                
                # Save changes
                if st.button("Save Changes"):
                    try:
                        # Parse and validate JSON
                        edited_profile = json.loads(edited_json)
                        
                        # Save profile
                        self.schema_manager.save_profile(selected_profile, edited_profile)
                        
                        st.success("✅ Profile saved successfully!")
                        
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {str(e)}")
                    except ValueError as e:
                        st.error(f"Invalid profile structure: {str(e)}")
                    except Exception as e:
                        st.error(f"Error saving profile: {str(e)}")
                
            except Exception as e:
                st.error(f"Error loading profile: {str(e)}")
    
    def _render_create_profile(self) -> None:
        """Render the profile creation interface."""
        st.header("Create New Profile")
        
        # Basic information
        profile_id = st.text_input("Profile ID", placeholder="e.g., dealersocket")
        profile_name = st.text_input("Display Name", placeholder="e.g., DealerSocket Schema")
        description = st.text_area("Description", placeholder="Describe the purpose of this profile")
        role = st.text_input("Role", placeholder="e.g., admin")
        
        # Column editor
        st.subheader("Columns")
        
        if 'columns' not in st.session_state:
            st.session_state.columns = []
        
        # Add new column
        with st.expander("Add Column", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                new_col_name = st.text_input("Column Name", key="new_col_name")
                new_col_type = st.selectbox(
                    "Data Type",
                    options=["string", "integer", "float", "datetime", "boolean"],
                    key="new_col_type"
                )
            
            with col2:
                new_col_display = st.text_input("Display Name", key="new_col_display")
                new_col_format = st.text_input("Format", key="new_col_format")
            
            new_col_desc = st.text_area("Description", key="new_col_desc")
            new_col_aliases = st.text_input(
                "Aliases (comma-separated)",
                key="new_col_aliases"
            )
            
            if st.button("Add Column"):
                if new_col_name and new_col_display:
                    st.session_state.columns.append({
                        "name": new_col_name,
                        "display_name": new_col_display,
                        "description": new_col_desc,
                        "data_type": new_col_type,
                        "format": new_col_format,
                        "aliases": [a.strip() for a in new_col_aliases.split(",") if a.strip()],
                        "visibility": "public"
                    })
                    st.success("Column added!")
                else:
                    st.warning("Column name and display name are required.")
        
        # Show current columns
        if st.session_state.columns:
            st.subheader("Current Columns")
            for i, col in enumerate(st.session_state.columns):
                with st.expander(f"{col['display_name']} ({col['name']})"):
                    st.json(col)
                    if st.button(f"Remove Column {i}"):
                        st.session_state.columns.pop(i)
                        st.rerun()
        
        # Create profile
        if st.button("Create Profile"):
            if not profile_id or not profile_name:
                st.warning("Profile ID and name are required.")
                return
            
            if not st.session_state.columns:
                st.warning("At least one column is required.")
                return
            
            try:
                # Create profile dictionary
                profile = {
                    "id": profile_id,
                    "name": profile_name,
                    "description": description,
                    "role": role,
                    "columns": st.session_state.columns,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                # Save profile
                self.schema_manager.save_profile(profile_id, profile)
                
                st.success("✅ Profile created successfully!")
                
                # Clear form
                st.session_state.columns = []
                
            except Exception as e:
                st.error(f"Error creating profile: {str(e)}")
    
    def _render_preview(self) -> None:
        """Render the profile preview interface."""
        st.header("Preview Schema Profile")
        
        # Upload sample data
        uploaded_file = st.file_uploader(
            "Upload sample data to preview mapping",
            type=['csv', 'xlsx']
        )
        
        if uploaded_file:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Get available profiles
                profiles = self.schema_manager.list_profiles()
                profile_names = [p['name'] for p in profiles]
                
                # Select profile to preview
                selected_profile = st.selectbox(
                    "Select Profile",
                    options=profile_names
                )
                
                if selected_profile:
                    from validators.column_mapper import ColumnMapper
                    
                    # Create mapper
                    mapper = ColumnMapper(self.schema_manager)
                    
                    # Map columns
                    mapped_df, results = mapper.map_columns(df, selected_profile)
                    
                    # Show results
                    st.subheader("Mapping Results")
                    
                    # Successful mappings
                    st.write("✅ Successfully Mapped Columns:")
                    mapped_data = []
                    for target, source in results['mapped'].items():
                        mapped_data.append({
                            "Target Column": target,
                            "Source Column": source,
                            "Confidence": f"{results['confidence_scores'][target]:.1f}%"
                        })
                    
                    if mapped_data:
                        st.dataframe(pd.DataFrame(mapped_data))
                    
                    # Unmapped columns
                    if results['unmapped']:
                        st.write("❌ Unmapped Columns:")
                        st.write(", ".join(results['unmapped']))
                    
                    # Preview mapped data
                    st.subheader("Mapped Data Preview")
                    st.dataframe(mapped_df.head())
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("Upload a CSV or Excel file to preview schema mapping.")