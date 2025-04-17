import streamlit as st
import time
from datetime import datetime
import os
import logging
from typing import Dict, Any, List, Optional, Callable

class NovaFallback:
    """
    Handles fallback scenarios when automated Nova Act operations fail.
    Provides UI components and workflows for manual intervention and recovery.
    """
    
    def __init__(self, title: str = "Nova Act Manual Intervention Required"):
        """Initialize the fallback handler"""
        self.title = title
        self.logger = logging.getLogger("NovaFallback")
        self._pending_actions: Dict[str, Dict[str, Any]] = {}
        self._completed_actions: List[Dict[str, Any]] = []
    
    def register_fallback_action(self, 
                               action_id: str, 
                               title: str, 
                               description: str,
                               action_type: str,
                               retry_callback: Optional[Callable] = None,
                               skip_callback: Optional[Callable] = None,
                               details: Dict[str, Any] = None) -> str:
        """
        Register a fallback action that requires manual intervention
        
        Args:
            action_id: Unique identifier for this action
            title: Short title describing the issue
            description: Detailed description of what went wrong and how to fix it
            action_type: Type of action required ('2fa', 'manual_upload', 'data_correction', etc.)
            retry_callback: Function to call when user wants to retry
            skip_callback: Function to call when user wants to skip
            details: Additional details about the issue
            
        Returns:
            action_id: The registered action ID
        """
        self._pending_actions[action_id] = {
            'id': action_id,
            'title': title,
            'description': description,
            'action_type': action_type,
            'retry_callback': retry_callback,
            'skip_callback': skip_callback,
            'details': details or {},
            'status': 'pending',
            'resolution': None
        }
        
        self.logger.info(f"Registered fallback action {action_id}: {title}")
        return action_id
    
    def complete_action(self, action_id: str, resolution: str, notes: str = None) -> bool:
        """
        Mark a fallback action as completed
        
        Args:
            action_id: The ID of the action to complete
            resolution: How it was resolved ('manual', 'retry', 'skip', 'fixed')
            notes: Optional notes about resolution
            
        Returns:
            bool: True if action was found and completed, False otherwise
        """
        if action_id not in self._pending_actions:
            return False
            
        action = self._pending_actions.pop(action_id)
        action['status'] = 'completed'
        action['resolution'] = resolution
        action['notes'] = notes
        
        self._completed_actions.append(action)
        self.logger.info(f"Action {action_id} completed with resolution: {resolution}")
        return True
    
    def render_fallback_ui(self):
        """Render the fallback UI in Streamlit"""
        if not self._pending_actions:
            return
            
        st.error(self.title)
        
        tabs = st.tabs([f"Issue {i+1}: {action['title']}" 
                        for i, action in enumerate(self._pending_actions.values())])
        
        for i, (action_id, action) in enumerate(self._pending_actions.items()):
            with tabs[i]:
                self._render_action_ui(action_id, action)
    
    def _render_action_ui(self, action_id: str, action: Dict[str, Any]):
        """Render UI for a specific action"""
        st.markdown(f"### {action['title']}")
        st.markdown(action['description'])
        
        # Display details relevant to this action type
        self._render_action_details(action)
        
        # Action buttons
        cols = st.columns(3)
        
        with cols[0]:
            if action['retry_callback'] and st.button("Retry", key=f"retry_{action_id}"):
                st.session_state[f"action_{action_id}_status"] = "retrying"
                success = action['retry_callback']()
                if success:
                    self.complete_action(action_id, "retry")
                    st.success("Retry successful!")
                    st.session_state[f"action_{action_id}_status"] = "completed"
                else:
                    st.error("Retry failed. Please try manual resolution.")
                    st.session_state[f"action_{action_id}_status"] = "failed"
        
        with cols[1]:
            if st.button("Mark as Resolved", key=f"resolve_{action_id}"):
                notes = st.session_state.get(f"notes_{action_id}", "")
                self.complete_action(action_id, "manual", notes)
                st.success("Action marked as resolved!")
                st.session_state[f"action_{action_id}_status"] = "completed"
        
        with cols[2]:
            if action['skip_callback'] and st.button("Skip", key=f"skip_{action_id}"):
                action['skip_callback']()
                self.complete_action(action_id, "skipped")
                st.info("Action skipped.")
                st.session_state[f"action_{action_id}_status"] = "skipped"
        
        # Notes input
        st.text_area("Resolution notes", key=f"notes_{action_id}", 
                    placeholder="Describe how you resolved this issue...")
    
    def _render_action_details(self, action: Dict[str, Any]):
        """Render details specific to action type"""
        action_type = action['action_type']
        details = action['details']
        
        if action_type == '2fa':
            st.info("Two-Factor Authentication Required")
            st.markdown("""
            **Steps to complete 2FA:**
            1. Check your email or authentication app for the verification code
            2. Enter the code below
            3. Click 'Submit Code'
            """)
            
            code = st.text_input("Enter 2FA code", key=f"2fa_code_{action['id']}")
            if 'verify_2fa' in details and st.button("Submit Code"):
                if details['verify_2fa'](code):
                    self.complete_action(action['id'], "manual", f"2FA code submitted: {code}")
                    st.success("2FA verification successful!")
                else:
                    st.error("Invalid 2FA code. Please try again.")
        
        elif action_type == 'manual_upload':
            st.info("Manual Data Upload Required")
            st.markdown("""
            **Steps to manually upload data:**
            1. Log into the vendor portal using the credentials below
            2. Navigate to the export/report section
            3. Download the required data
            4. Upload the file below
            """)
            
            if 'credentials' in details:
                with st.expander("Portal Credentials"):
                    st.code(f"""
                    URL: {details.get('url', 'N/A')}
                    Username: {details.get('credentials', {}).get('username', 'N/A')}
                    Password: {details.get('credentials', {}).get('password', '********')}
                    """)
            
            uploaded_file = st.file_uploader("Upload data file", key=f"manual_upload_{action['id']}")
            if uploaded_file and 'process_upload' in details:
                if details['process_upload'](uploaded_file):
                    self.complete_action(action['id'], "manual", f"File uploaded: {uploaded_file.name}")
                    st.success("File processed successfully!")
                else:
                    st.error("Error processing file. Please ensure it's the correct format.")
        
        elif action_type == 'data_correction':
            st.info("Data Correction Required")
            st.markdown("""
            **Data inconsistencies were detected and require manual correction.**
            Please review the issues below and make necessary corrections.
            """)
            
            if 'issues' in details:
                for idx, issue in enumerate(details['issues']):
                    with st.expander(f"Issue #{idx+1}: {issue.get('title', 'Data issue')}"):
                        st.markdown(issue.get('description', ''))
                        st.text_area("Corrected value", 
                                    key=f"correction_{action['id']}_{idx}",
                                    value=issue.get('current_value', ''))
            
            if st.button("Apply Corrections", key=f"apply_corrections_{action['id']}"):
                corrections = {}
                for idx, _ in enumerate(details.get('issues', [])):
                    corrections[idx] = st.session_state.get(f"correction_{action['id']}_{idx}")
                
                if 'apply_corrections' in details and details['apply_corrections'](corrections):
                    self.complete_action(action['id'], "manual", "Data corrections applied")
                    st.success("Corrections applied successfully!")
                else:
                    st.error("Error applying corrections.")
        
        else:
            # Generic fallback for other action types
            st.markdown("**Details:**")
            for key, value in details.items():
                if isinstance(value, str):
                    st.text(f"{key}: {value}")
    
    def has_pending_actions(self) -> bool:
        """Check if there are any pending actions"""
        return len(self._pending_actions) > 0
    
    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific action by ID"""
        return self._pending_actions.get(action_id)
    
    def get_all_pending_actions(self) -> List[Dict[str, Any]]:
        """Get all pending actions"""
        return list(self._pending_actions.values())
    
    def get_all_completed_actions(self) -> List[Dict[str, Any]]:
        """Get all completed actions"""
        return self._completed_actions

# Example Usage (for demonstration only)
if __name__ == "__main__":
    st.set_page_config(page_title="Nova Act Manual Intervention", page_icon="🔒")
    
    st.title("Nova Act Manual Intervention Demo")
    
    # Example of testing different fallback types
    error_types = ["2fa", "captcha", "credentials", "generic"]
    vendor = "VinSolutions"
    credentials = {"username": "demo_user@example.com"}
    
    error_type = st.selectbox("Select Error Type to Simulate:", error_types)
    
    if st.button("Simulate Error"):
        NovaFallback.handle_login_friction(
            vendor=vendor,
            error=f"Simulated {error_type} error for testing purposes",
            error_type=error_type,
            credentials=credentials
        )
    
    # Clear session state button for testing
    if st.button("Clear Session State"):
        for key in list(st.session_state.keys()):
            if key.startswith("nova_act_"):
                del st.session_state[key]
        st.success("Session state cleared!") 