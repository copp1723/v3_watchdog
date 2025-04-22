"""
Admin Preferences System for Watchdog AI.

This component allows administrators to configure notification
preferences, including delivery frequency, insight types, and recipients.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Import necessary modules from the project
try:
    from ...insights.insight_functions import get_available_insight_types
except ImportError:
    # Fallback if function not found in the module
    logger.warning("get_available_insight_types not found, using mock implementation")
    def get_available_insight_types():
        """Fallback function for insight types."""
        return {
            "roi_analysis": "Lead Source ROI Analysis",
            "sales_performance": "Sales Performance",
            "inventory_health": "Inventory Health",
            "lead_conversion": "Lead Conversion",
            "trending_models": "Trending Models",
            "price_competitiveness": "Price Competitiveness Analysis",
        }

try:
    from ....scheduler.notification_service import NotificationService
except ImportError:
    # Fallback if service not found
    logger.warning("NotificationService not found, using mock implementation")
    class NotificationService:
        """Mock notification service for testing."""
        def __init__(self, reports_dir=None, templates_dir=None):
            pass
            
        def send_insight_email(self, recipients, insights, subject=None, parameters=None):
            logger.info(f"Mock notification to {recipients} with {len(insights)} insights")
            return "mock_message_id"

# Default preferences
DEFAULT_PREFERENCES = {
    "delivery": {
        "frequency": "weekly",
        "day_of_week": "Monday",
        "time": "08:00",
    },
    "insight_types": {
        "roi_analysis": True,
        "sales_performance": True,
        "inventory_health": True,
        "lead_conversion": True,
        "trending_models": False,
        "price_competitiveness": False,
    },
    "notifications": {
        "email": True,
        "dashboard": True,
        "alerts_only": False,
    },
    "recipients": [],
    "updated_at": datetime.now().isoformat(),
    "last_run": None
}


class AdminPreferences:
    """Admin preferences management system."""
    
    def __init__(self, preferences_path: Optional[str] = None):
        """
        Initialize the preferences manager.
        
        Args:
            preferences_path: Path to preferences file
        """
        self.preferences_path = preferences_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config",
            "admin_preferences.json"
        )
        
        # Load or create preferences
        self.preferences = self._load_preferences()
        
        # Create notification service
        self.notification_service = NotificationService()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """
        Load preferences from file or create defaults.
        
        Returns:
            Dictionary with preferences
        """
        try:
            if os.path.exists(self.preferences_path):
                with open(self.preferences_path, 'r') as f:
                    preferences = json.load(f)
                logger.info(f"Loaded preferences from {self.preferences_path}")
                
                # Update with any missing keys from defaults
                updated = False
                for section, values in DEFAULT_PREFERENCES.items():
                    if section not in preferences:
                        preferences[section] = values
                        updated = True
                    elif isinstance(values, dict):
                        for key, value in values.items():
                            if key not in preferences[section]:
                                preferences[section][key] = value
                                updated = True
                
                if updated:
                    logger.info("Updated preferences with new default values")
                    self._save_preferences(preferences)
                
                return preferences
            else:
                logger.info(f"Preferences file not found, creating default at {self.preferences_path}")
                self._save_preferences(DEFAULT_PREFERENCES)
                return DEFAULT_PREFERENCES.copy()
                
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            return DEFAULT_PREFERENCES.copy()
    
    def _save_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Save preferences to file.
        
        Args:
            preferences: Preferences to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.preferences_path), exist_ok=True)
            
            # Update timestamp
            preferences["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.preferences_path, 'w') as f:
                json.dump(preferences, f, indent=2)
                
            logger.info(f"Saved preferences to {self.preferences_path}")
                
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def update_preferences(self, new_preferences: Dict[str, Any]) -> None:
        """
        Update preferences with new values.
        
        Args:
            new_preferences: New preference values
        """
        self.preferences.update(new_preferences)
        self._save_preferences(self.preferences)
    
    def get_preferences(self) -> Dict[str, Any]:
        """
        Get current preferences.
        
        Returns:
            Dictionary with current preferences
        """
        return self.preferences.copy()
    
    def add_recipient(self, email: str, name: Optional[str] = None) -> None:
        """
        Add a notification recipient.
        
        Args:
            email: Recipient email address
            name: Optional recipient name
        """
        # Validate email format
        if not self._is_valid_email(email):
            raise ValueError(f"Invalid email address: {email}")
            
        # Check if already exists
        for recipient in self.preferences["recipients"]:
            if recipient["email"] == email:
                # Update name if provided
                if name and recipient["name"] != name:
                    recipient["name"] = name
                    self._save_preferences(self.preferences)
                return
        
        # Add new recipient
        self.preferences["recipients"].append({
            "email": email,
            "name": name or email.split('@')[0],
            "added_at": datetime.now().isoformat()
        })
        
        self._save_preferences(self.preferences)
    
    def remove_recipient(self, email: str) -> None:
        """
        Remove a notification recipient.
        
        Args:
            email: Recipient email address
        """
        self.preferences["recipients"] = [
            r for r in self.preferences["recipients"] 
            if r["email"] != email
        ]
        
        self._save_preferences(self.preferences)
    
    def _is_valid_email(self, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def send_test_notification(self) -> bool:
        """
        Send a test notification to configured recipients.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            recipients = [r["email"] for r in self.preferences["recipients"]]
            if not recipients:
                logger.warning("No recipients configured for test notification")
                return False
            
            # Create test insight
            test_insight = {
                "title": "Test Notification",
                "summary": "This is a test notification from Watchdog AI.",
                "metrics": [
                    {"label": "Test Metric", "value": "100%"},
                    {"label": "System Status", "value": "Online"}
                ],
                "recommendations": [
                    "Configure your notification preferences as needed.",
                    "Add additional recipients if necessary."
                ]
            }
            
            # Send test notification
            message_id = self.notification_service.send_insight_email(
                recipients=recipients,
                insights=[test_insight],
                subject="Watchdog AI Test Notification",
                parameters={
                    "test_mode": True,
                    "sent_by": "Admin Preferences System"
                }
            )
            
            if message_id:
                logger.info(f"Test notification sent successfully: {message_id}")
                return True
            else:
                logger.error("Failed to send test notification")
                return False
                
        except Exception as e:
            logger.error(f"Error sending test notification: {e}")
            return False


def render_admin_preferences_page():
    """Render the admin preferences page in Streamlit."""
    st.title("Admin Preferences")
    
    # Initialize preferences manager
    preferences_manager = AdminPreferences()
    preferences = preferences_manager.get_preferences()
    
    # Create tabs for different preference sections
    tabs = st.tabs(["Delivery Settings", "Insight Types", "Recipients", "Test & Save"])
    
    with tabs[0]:  # Delivery Settings
        st.header("Delivery Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            frequency = st.selectbox(
                "Delivery Frequency",
                options=["daily", "weekly", "monthly", "alerts_only"],
                index=["daily", "weekly", "monthly", "alerts_only"].index(
                    preferences["delivery"]["frequency"]
                ),
                help="How often to deliver insights"
            )
            
            if frequency == "weekly":
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=[
                        "Monday", "Tuesday", "Wednesday", 
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ],
                    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(
                        preferences["delivery"].get("day_of_week", "Monday")
                    )
                )
            else:
                day_of_week = preferences["delivery"].get("day_of_week", "Monday")
                
            if frequency == "monthly":
                day_of_month = st.slider(
                    "Day of Month",
                    min_value=1,
                    max_value=28,
                    value=preferences["delivery"].get("day_of_month", 1)
                )
            else:
                day_of_month = preferences["delivery"].get("day_of_month", 1)
        
        with col2:
            delivery_time = st.time_input(
                "Delivery Time",
                value=datetime.strptime(
                    preferences["delivery"].get("time", "08:00"), 
                    "%H:%M"
                ).time()
            )
            
            st.write("Notification Methods:")
            email_notify = st.checkbox(
                "Email",
                value=preferences["notifications"].get("email", True)
            )
            dashboard_notify = st.checkbox(
                "Dashboard",
                value=preferences["notifications"].get("dashboard", True)
            )
            alerts_only = st.checkbox(
                "Send Alerts Only",
                value=preferences["notifications"].get("alerts_only", False),
                help="Only send notifications for critical alerts, not regular reports"
            )
    
    with tabs[1]:  # Insight Types
        st.header("Insight Types")
        st.write("Select which types of insights you want to receive:")
        
        # Get available insight types
        insight_types = get_available_insight_types()
        
        # Organize in columns
        col1, col2 = st.columns(2)
        
        # Split the insight types between columns
        half = len(insight_types) // 2 + len(insight_types) % 2
        
        selected_insights = {}
        
        with col1:
            for insight in list(insight_types.keys())[:half]:
                selected_insights[insight] = st.checkbox(
                    insight_types[insight],
                    value=preferences["insight_types"].get(insight, False)
                )
        
        with col2:
            for insight in list(insight_types.keys())[half:]:
                selected_insights[insight] = st.checkbox(
                    insight_types[insight],
                    value=preferences["insight_types"].get(insight, False)
                )
    
    with tabs[2]:  # Recipients
        st.header("Notification Recipients")
        
        # Display current recipients
        if preferences["recipients"]:
            st.write("Current Recipients:")
            
            recipient_data = []
            for recipient in preferences["recipients"]:
                recipient_data.append({
                    "Name": recipient["name"],
                    "Email": recipient["email"],
                    "Added": recipient.get("added_at", "Unknown")
                })
            
            recipient_df = pd.DataFrame(recipient_data)
            st.dataframe(recipient_df)
            
            # Remove recipients
            if recipient_data:
                to_remove = st.multiselect(
                    "Select recipients to remove:",
                    options=[r["Email"] for r in recipient_data]
                )
                
                if to_remove and st.button("Remove Selected Recipients"):
                    for email in to_remove:
                        preferences_manager.remove_recipient(email)
                    st.success(f"Removed {len(to_remove)} recipients")
                    st.rerun()
        else:
            st.info("No recipients configured. Add at least one recipient below.")
        
        # Add new recipient
        st.subheader("Add New Recipient")
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("Name")
        
        with col2:
            new_email = st.text_input("Email")
        
        if st.button("Add Recipient"):
            if not new_email:
                st.error("Email is required")
            else:
                try:
                    preferences_manager.add_recipient(new_email, new_name)
                    st.success(f"Added {new_email} to recipients")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    
    with tabs[3]:  # Test & Save
        st.header("Test & Save Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Send Test Notification"):
                if not preferences["recipients"]:
                    st.error("No recipients configured. Please add at least one recipient.")
                else:
                    success = preferences_manager.send_test_notification()
                    if success:
                        st.success("Test notification sent successfully!")
                    else:
                        st.error("Failed to send test notification. Check logs for details.")
        
        with col2:
            if st.button("Save Preferences"):
                # Update preferences
                new_preferences = {
                    "delivery": {
                        "frequency": frequency,
                        "day_of_week": day_of_week,
                        "day_of_month": day_of_month,
                        "time": delivery_time.strftime("%H:%M")
                    },
                    "insight_types": selected_insights,
                    "notifications": {
                        "email": email_notify,
                        "dashboard": dashboard_notify,
                        "alerts_only": alerts_only
                    }
                }
                
                preferences_manager.update_preferences(new_preferences)
                st.success("Preferences saved successfully!")
        
        # Show current settings summary
        st.subheader("Current Settings Summary")
        st.json(preferences)


# Helper function to get available insight types
def get_available_insight_types() -> Dict[str, str]:
    """
    Get available insight types for UI.
    
    Returns:
        Dictionary of insight type IDs to display names
    """
    # In a real implementation, this would be integrated with the actual
    # insight types available in the system. For the demo, we'll use static values.
    return {
        "roi_analysis": "Lead Source ROI Analysis",
        "sales_performance": "Sales Performance",
        "inventory_health": "Inventory Health",
        "lead_conversion": "Lead Conversion",
        "trending_models": "Trending Models",
        "price_competitiveness": "Price Competitiveness Analysis",
        "market_comparison": "Market Comparison",
        "customer_satisfaction": "Customer Satisfaction",
        "service_department": "Service Department KPIs",
        "finance_insurance": "Finance & Insurance Performance"
    }