"""
Schema Profile Editor UI Component for Watchdog AI.

This module provides the Streamlit UI for editing and previewing schema profiles.
"""

import os
import streamlit as st
import pandas as pd
import json
import yaml
from typing import Optional, Dict, Any, List
from datetime import datetime

from ...utils.schema_profile_editor import SchemaProfileEditor
from ...utils.adaptive_schema import SchemaProfile, SchemaColumn, ExecRole

def render_profile_editor(editor: SchemaProfileEditor,
                        sample_data: Optional[pd.DataFrame] = None) -> None:
    """
    Render the schema profile editor interface.
    
    Args:
        editor: SchemaProfileEditor instance
        sample_data: Optional sample DataFrame for preview
    """
    st.subheader("Schema Profile Editor")
    
    # Profile selection/creation
    col1, col2 = st.columns([3, 1])
    with col1:
        action = st.radio(
            "Action",
            ["Edit Existing Profile", "Create New Profile", "Import Profile"],
            horizontal=True
        )
    
    if action == "Edit Existing Profile":
        _render_profile_selector(editor)
    elif action == "Create New Profile":
        _render_new_profile_form(editor)
    else:
        _render_import_interface(editor)
    
    # Show editor if we have a profile
    if editor.current_profile:
        _render_profile_editor_form(editor, sample_data)

def _render_profile_selector(editor: SchemaProfileEditor) -> None:
    """Render the profile selection interface."""
    # Get list of profiles
    profiles = []
    for filename in os.listdir(editor.profiles_dir):
        if filename.endswith(('.json', '.yml', '.yaml')):
            profile_id = os.path.splitext(filename)[0]
            profile = editor.load_profile(profile_id)
            if profile:
                profiles.append(profile)
    
    if not profiles:
        st.warning("No profiles found")
        return
    
    # Create selection box
    profile_names = [f"{p.name} ({p.id})" for p in profiles]
    selected = st.selectbox("Select Profile", profile_names)
    
    if selected:
        # Get selected profile ID
        profile_id = selected.split('(')[-1].strip(')')
        profile = editor.load_profile(profile_id)
        
        if profile:
            # Show profile actions
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Duplicate"):
                    new_id = f"{profile.id}_copy"
                    new_profile = editor.duplicate_profile(profile, new_id)
                    if new_profile:
                        st.success(f"Created duplicate profile: {new_id}")
                        editor.current_profile = new_profile
            with col2:
                format = st.selectbox("Export Format", ['json', 'yaml'])
                if st.button("Export"):
                    exported = editor.export_profile(profile, format)
                    if exported:
                        st.download_button(
                            "Download Profile",
                            exported,
                            file_name=f"{profile.id}.{format}",
                            mime=f"application/{format}"
                        )
            with col3:
                if st.button("Delete", type="secondary"):
                    if editor.delete_profile(profile.id):
                        st.success(f"Deleted profile: {profile.id}")
                        editor.current_profile = None
                        st.rerun()

def _render_new_profile_form(editor: SchemaProfileEditor) -> None:
    """Render the new profile creation form."""
    with st.form("new_profile"):
        profile_id = st.text_input("Profile ID")
        name = st.text_input("Profile Name")
        description = st.text_area("Description")
        role = st.selectbox("Role", [r.value for r in ExecRole])
        
        if st.form_submit_button("Create Profile"):
            if not profile_id or not name:
                st.error("Profile ID and Name are required")
                return
            
            # Create new profile
            profile = SchemaProfile(
                id=profile_id,
                name=name,
                description=description,
                role=role,
                columns=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            # Validate and save
            validation = editor.validate_profile(profile)
            if validation["is_valid"]:
                if editor.save_profile(profile):
                    st.success("Profile created successfully")
                    editor.current_profile = profile
                else:
                    st.error("Error saving profile")
            else:
                st.error("\n".join(validation["errors"]))

def _render_import_interface(editor: SchemaProfileEditor) -> None:
    """Render the profile import interface."""
    format = st.selectbox("Import Format", ['json', 'yaml'])
    data = st.text_area("Profile Data", height=300)
    
    if st.button("Import Profile"):
        if not data:
            st.error("Please enter profile data")
            return
        
        profile = editor.import_profile(data, format)
        if profile:
            if editor.save_profile(profile):
                st.success(f"Profile '{profile.name}' imported successfully")
                editor.current_profile = profile
            else:
                st.error("Error saving imported profile")
        else:
            st.error("Invalid profile data")

def _render_profile_editor_form(editor: SchemaProfileEditor,
                             sample_data: Optional[pd.DataFrame] = None) -> None:
    """Render the main profile editor form."""
    profile = editor.current_profile
    if not profile:
        return
    
    st.write("---")
    st.subheader(f"Editing Profile: {profile.name}")
    
    # Basic info
    with st.expander("Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            profile.name = st.text_input("Name", profile.name)
        with col2:
            profile.role = st.selectbox("Role", [r.value for r in ExecRole], 
                                      index=[r.value for r in ExecRole].index(profile.role))
        profile.description = st.text_area("Description", profile.description)
    
    # Columns
    with st.expander("Columns", expanded=True):
        st.write("#### Columns")
        
        # Add new column button
        if st.button("Add Column"):
            profile.columns.append(SchemaColumn(
                name="new_column",
                display_name="New Column",
                description="",
                data_type="string",
                visibility="public"
            ))
        
        # Edit existing columns
        for i, col in enumerate(profile.columns):
            with st.expander(f"Column: {col.name}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    col.name = st.text_input("Name", col.name, key=f"col_name_{i}")
                    col.display_name = st.text_input("Display Name", col.display_name, key=f"col_display_{i}")
                with col2:
                    col.data_type = st.selectbox("Data Type", 
                                               ['string', 'integer', 'float', 'date', 'boolean'],
                                               index=['string', 'integer', 'float', 'date', 'boolean'].index(col.data_type),
                                               key=f"col_type_{i}")
                    col.visibility = st.selectbox("Visibility",
                                               ['public', 'restricted', 'private'],
                                               index=['public', 'restricted', 'private'].index(col.visibility),
                                               key=f"col_vis_{i}")
                
                col.description = st.text_area("Description", col.description, key=f"col_desc_{i}")
                
                # Aliases
                aliases = st.text_input("Aliases (comma-separated)", 
                                     ",".join(col.aliases),
                                     key=f"col_aliases_{i}")
                col.aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                
                # Delete column button
                if st.button("Delete Column", key=f"del_col_{i}"):
                    profile.columns.pop(i)
                    st.rerun()
    
    # Preview section
    if sample_data is not None:
        with st.expander("Live Preview", expanded=True):
            st.write("#### Preview with Sample Data")
            preview = editor.preview_validation(profile, sample_data)
            
            if preview["success"]:
                # Show sample data transformation
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Data")
                    st.dataframe(pd.DataFrame(preview["sample_data"]["original"]))
                with col2:
                    st.write("Normalized Data")
                    st.dataframe(pd.DataFrame(preview["sample_data"]["normalized"]))
                
                # Show mapping summary
                st.write("#### Column Mapping")
                for src, target in preview["normalization_summary"].get("column_mappings", {}).items():
                    st.write(f"- {src} → {target}")
                
                # Show unmapped columns
                unmapped = preview["normalization_summary"].get("unmapped_columns", [])
                if unmapped:
                    st.warning("Unmapped Columns: " + ", ".join(unmapped))
            else:
                st.error(f"Preview error: {preview.get('error', 'Unknown error')}")
    
    # Save button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Save Changes", type="primary"):
            # Validate and save
            validation = editor.validate_profile(profile)
            if validation["is_valid"]:
                if editor.save_profile(profile):
                    st.success("Changes saved successfully")
                else:
                    st.error("Error saving changes")
            else:
                st.error("\n".join(validation["errors"]))