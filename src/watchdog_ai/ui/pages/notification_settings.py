"""
Notification Settings UI for configuring delivery preferences and alert escalations.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime, time, timedelta
import json
import os

from ...scheduler.notification_service import NotificationService
from ...ui.utils.status_formatter import StatusType, format_status_text
from ...utils.config import save_user_preferences, get_user_preferences
from ....notifications.escalation import AlertEscalationRouter

class NotificationSettings:
    """Settings interface for notification preferences and escalation settings."""
    
    def __init__(self, 
                notification_service: Optional[NotificationService] = None,
                escalation_router: Optional[AlertEscalationRouter] = None):
        """
        Initialize settings interface.
        
        Args:
            notification_service: Optional notification service instance
            escalation_router: Optional escalation router instance
        """
        self.notification_service = notification_service or NotificationService()
        self.escalation_router = escalation_router or AlertEscalationRouter()
    
    def render(self) -> None:
        """Render the notification settings interface."""
        st.title("Notification Settings")
        
        # Get current user
        if 'user' not in st.session_state:
            st.warning("Please log in to configure notification settings.")
            return
        
        user = st.session_state.user
        
        # Load current preferences
        preferences = get_user_preferences(user.username)
        
        # Create tabs for different settings
        tab1, tab2, tab3, tab4 = st.tabs(["Delivery Preferences", "Schedule", "Templates", "Lead Escalation"])
        
        with tab1:
            self._render_delivery_preferences(preferences)
        
        with tab2:
            self._render_schedule_settings(preferences)
        
        with tab3:
            self._render_template_settings(preferences)
            
        with tab4:
            self._render_escalation_settings(preferences)
    
    def _render_delivery_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Render delivery preferences section.
        
        Args:
            preferences: Current user preferences
        """
        st.header("Delivery Preferences")
        
        with st.form("delivery_preferences"):
            # Channel selection
            st.subheader("Notification Channels")
            
            email_enabled = st.checkbox(
                "Email Notifications",
                value=preferences.get('channels', {}).get('email', True)
            )
            
            if email_enabled:
                email = st.text_input(
                    "Email Address",
                    value=preferences.get('email', '')
                )
            
            slack_enabled = st.checkbox(
                "Slack Notifications",
                value=preferences.get('channels', {}).get('slack', False)
            )
            
            if slack_enabled:
                slack_channel = st.text_input(
                    "Slack Channel",
                    value=preferences.get('slack_channel', '#notifications')
                )
            
            sms_enabled = st.checkbox(
                "SMS Notifications",
                value=preferences.get('channels', {}).get('sms', False)
            )
            
            if sms_enabled:
                phone = st.text_input(
                    "Phone Number",
                    value=preferences.get('phone', '')
                )
            
            # Notification types
            st.subheader("Notification Types")
            
            daily_summary = st.checkbox(
                "Daily Summary",
                value=preferences.get('types', {}).get('daily_summary', True)
            )
            
            weekly_exec = st.checkbox(
                "Weekly Executive Report",
                value=preferences.get('types', {}).get('weekly_exec', False)
            )
            
            critical_alerts = st.checkbox(
                "Critical Alerts",
                value=preferences.get('types', {}).get('critical_alerts', True)
            )
            
            # Save changes
            if st.form_submit_button("Save Preferences"):
                # Update preferences
                new_preferences = {
                    'channels': {
                        'email': email_enabled,
                        'slack': slack_enabled,
                        'sms': sms_enabled
                    },
                    'types': {
                        'daily_summary': daily_summary,
                        'weekly_exec': weekly_exec,
                        'critical_alerts': critical_alerts
                    }
                }
                
                if email_enabled:
                    new_preferences['email'] = email
                if slack_enabled:
                    new_preferences['slack_channel'] = slack_channel
                if sms_enabled:
                    new_preferences['phone'] = phone
                
                # Save to user preferences
                # Save to user preferences
                save_user_preferences(st.session_state.user.username, new_preferences)
                success_text = format_status_text(StatusType.SUCCESS, custom_text="Preferences saved successfully!")
                st.markdown(success_text, unsafe_allow_html=True)
    def _render_schedule_settings(self, preferences: Dict[str, Any]) -> None:
        """
        Render schedule settings section.
        
        Args:
            preferences: Current user preferences
        """
        st.header("Schedule Settings")
        
        with st.form("schedule_settings"):
            # Daily summary timing
            st.subheader("Daily Summary")
            
            daily_time = st.time_input(
                "Delivery Time",
                value=datetime.strptime(
                    preferences.get('schedule', {}).get('daily_time', '09:00'),
                    '%H:%M'
                ).time()
            )
            
            # Weekly report timing
            st.subheader("Weekly Executive Report")
            
            weekly_day = st.selectbox(
                "Delivery Day",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'].index(
                    preferences.get('schedule', {}).get('weekly_day', 'Monday')
                )
            )
            
            weekly_time = st.time_input(
                "Delivery Time",
                value=datetime.strptime(
                    preferences.get('schedule', {}).get('weekly_time', '08:00'),
                    '%H:%M'
                ).time()
            )
            
            # Timezone
            timezone = st.selectbox(
                "Timezone",
                options=['UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific'],
                index=['UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific'].index(
                    preferences.get('schedule', {}).get('timezone', 'UTC')
                )
            )
            
            # Save changes
            if st.form_submit_button("Save Schedule"):
                # Update preferences
                new_schedule = {
                    'daily_time': daily_time.strftime('%H:%M'),
                    'weekly_day': weekly_day,
                    'weekly_time': weekly_time.strftime('%H:%M'),
                    'timezone': timezone
                }
                
                # Update preferences
                # Update preferences
                preferences['schedule'] = new_schedule
                save_user_preferences(st.session_state.user.username, preferences)
                success_text = format_status_text(StatusType.SUCCESS, custom_text="Schedule saved successfully!")
                st.markdown(success_text, unsafe_allow_html=True)
        # Preview next deliveries
        st.subheader("Next Scheduled Deliveries")
        
        next_daily = self._calculate_next_delivery(
            daily_time,
            None,
            timezone
        )
        
        next_weekly = self._calculate_next_delivery(
            weekly_time,
            weekly_day,
            timezone
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Next Daily Summary",
                next_daily.strftime("%Y-%m-%d %H:%M %Z")
            )
        
        with col2:
            st.metric(
                "Next Weekly Report",
                next_weekly.strftime("%Y-%m-%d %H:%M %Z")
            )
    
    def _render_template_settings(self, preferences: Dict[str, Any]) -> None:
        """
        Render template settings section.
        
        Args:
            preferences: Current user preferences
        """
        st.header("Template Settings")
        
        # Template preview
        template_type = st.selectbox(
            "Preview Template",
            options=['Daily Summary', 'Weekly Executive', 'Alert']
        )
        
        # Show template preview
        with st.expander("Template Preview", expanded=True):
            if template_type == 'Daily Summary':
                template = self.notification_service.jinja_env.get_template('daily_summary.html')
                html = template.render(
                    title="Daily Summary Preview",
                    date=datetime.now().strftime("%B %d, %Y"),
                    year=datetime.now().year,
                    insights=[
                        {
                            "title": "Sample Insight",
                            "summary": "This is a preview of how insights will appear.",
                            "metrics": [
                                {"label": "Metric 1", "value": "$1,234"},
                                {"label": "Metric 2", "value": "56.7%"}
                            ],
                            "recommendations": [
                                "Sample recommendation 1",
                                "Sample recommendation 2"
                            ]
                        }
                    ]
                )
                st.components.v1.html(html, height=600)
    
    def _calculate_next_delivery(self, delivery_time: time, 
                               day: Optional[str], timezone: str) -> datetime:
        """
        Calculate the next delivery time.
        
        Args:
            delivery_time: Time of day for delivery
            day: Optional day of week for delivery
            timezone: Timezone for delivery
            
        Returns:
            Datetime of next delivery
        """
        # This is a simplified calculation - in production, use proper timezone handling
        now = datetime.now()
        next_delivery = datetime.combine(now.date(), delivery_time)
        
        if next_delivery <= now:
            next_delivery += timedelta(days=1)
        
        if day:
            # Adjust to next occurrence of specified day
            days_ahead = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                         'Thursday': 3, 'Friday': 4}[day] - next_delivery.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_delivery += timedelta(days=days_ahead)
        
        return next_delivery
        
    def _render_escalation_settings(self, preferences: Dict[str, Any]) -> None:
        """
        Render lead escalation settings section.
        
        Args:
            preferences: Current user preferences
        """
        st.header("Lead Escalation Settings")
        
        # Get current escalation config
        config = self.escalation_router.config
        
        with st.form("escalation_settings"):
            # Thresholds section
            st.subheader("Escalation Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                low_prob = st.slider(
                    "Low Probability Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=config['thresholds'].get('low_probability', 0.3),
                    step=0.05,
                    help="Leads with sale probability below this threshold will trigger escalation"
                )
                
            with col2:
                critical_prob = st.slider(
                    "Critical Probability Threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=config['thresholds'].get('critical_probability', 0.1),
                    step=0.05,
                    help="Leads with sale probability below this threshold will trigger critical escalation"
                )
            
            # Recipient configuration
            st.subheader("Escalation Recipients")
            
            # Manager recipients
            st.write("Manager Recipients (for high-priority escalations)")
            manager_emails = st.text_area(
                "Manager Emails",
                value="\n".join([r.get('email', '') for r in config['routing'].get('manager_recipients', [])]),
                help="Enter one email address per line"
            )
            
            # Fallback recipients
            st.write("Fallback Recipients (when no direct recipient is available)")
            fallback_emails = st.text_area(
                "Fallback Emails",
                value="\n".join([r.get('email', '') for r in config['routing'].get('fallback_recipients', [])]),
                help="Enter one email address per line"
            )
            
            # Escalation delays
            st.subheader("Escalation Delays")
            
            col1, col2 = st.columns(2)
            
            with col1:
                default_delay = st.number_input(
                    "Default Delay (minutes)",
                    min_value=0,
                    max_value=1440,  # 24 hours
                    value=config['escalation_rules']['default'].get('delay_minutes', 60),
                    help="Time to wait before sending standard escalations"
                )
                
            with col2:
                high_value_delay = st.number_input(
                    "High-Value Delay (minutes)",
                    min_value=0,
                    max_value=1440,  # 24 hours
                    value=config['escalation_rules']['high_value'].get('delay_minutes', 30),
                    help="Time to wait before sending high-value escalations"
                )
            
            # Working hours configuration
            st.subheader("Working Hours")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_hour = st.slider(
                    "Start Hour",
                    min_value=0,
                    max_value=23,
                    value=config['working_hours'].get('start_hour', 8),
                    help="Start of business hours (24-hour format)"
                )
                
            with col2:
                end_hour = st.slider(
                    "End Hour",
                    min_value=0,
                    max_value=23,
                    value=config['working_hours'].get('end_hour', 18),
                    help="End of business hours (24-hour format)"
                )
            
            work_days = st.multiselect(
                "Work Days",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=[
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
                ] if 5 in config['working_hours'].get('work_days', [0,1,2,3,4]) else 
                [
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
                ],
                help="Days considered as working days"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                reassign = st.checkbox(
                    "Auto-reassign leads",
                    value=config['routing'].get('auto_reassign', True),
                    help="Automatically reassign leads if primary rep is unavailable"
                )
                
                after_hours = st.checkbox(
                    "After-hours routing",
                    value=config['routing'].get('reassign_after_hours', True),
                    help="Route escalations to designated after-hours reps outside working hours"
                )
                
                webhook_url = st.text_input(
                    "Webhook URL",
                    value=config['webhook_urls'].get('default', ''),
                    help="URL for webhook notifications (leave empty to disable)"
                )
            
            # Save changes
            if st.form_submit_button("Save Escalation Settings"):
                # Convert day names to weekday indices
                day_map = {
                    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                    "Friday": 4, "Saturday": 5, "Sunday": 6
                }
                work_day_indices = [day_map[day] for day in work_days]
                
                # Process email lists
                manager_recipients = []
                for email in manager_emails.strip().split('\n'):
                    if email:
                        manager_recipients.append({"name": "Manager", "email": email, "type": "manager"})
                
                fallback_recipients = []
                for email in fallback_emails.strip().split('\n'):
                    if email:
                        fallback_recipients.append({"name": "Fallback", "email": email, "type": "fallback"})
                
                # Update config
                new_config = {
                    "thresholds": {
                        "low_probability": low_prob,
                        "critical_probability": critical_prob
                    },
                    "escalation_rules": {
                        "default": {
                            "delay_minutes": default_delay,
                        },
                        "high_value": {
                            "delay_minutes": high_value_delay,
                        }
                    },
                    "routing": {
                        "manager_recipients": manager_recipients,
                        "fallback_recipients": fallback_recipients,
                        "auto_reassign": reassign,
                        "reassign_after_hours": after_hours
                    },
                    "webhook_urls": {
                        "default": webhook_url
                    },
                    "working_hours": {
                        "start_hour": start_hour,
                        "end_hour": end_hour,
                        "work_days": work_day_indices
                    }
                }
                
                # Update router configuration
                self.escalation_router.update_config(new_config)
                
                # Update user preferences
                preferences['escalation'] = {
                    'last_updated': datetime.now().isoformat(),
                    'thresholds': {
                        'low_probability': low_prob,
                        'critical_probability': critical_prob
                    }
                }
                save_user_preferences(st.session_state.user.username, preferences)
                
                success_text = format_status_text(StatusType.SUCCESS, custom_text="Escalation settings saved successfully!")
                st.markdown(success_text, unsafe_allow_html=True)
        
        # Show recent escalations
        recent_escalations = self.escalation_router.get_recent_escalations(5)
        
        if recent_escalations:
            st.subheader("Recent Escalations")
            
            for esc in recent_escalations:
                with st.expander(f"{esc.get('id')}: {esc.get('lead_id')} ({esc.get('level', 'unknown').upper()})"):
                    st.write(f"**Status:** {esc.get('status', 'Unknown')}")
                    st.write(f"**Created:** {esc.get('created_at', 'Unknown')}")
                    
                    if 'scheduled_for' in esc:
                        st.write(f"**Scheduled for:** {esc.get('scheduled_for')}")
                    
                    if 'executed_at' in esc:
                        st.write(f"**Executed:** {esc.get('executed_at')}")
                    
                    st.write(f"**Message:** {esc.get('message', 'No message')}")
                    
                    if 'recipients' in esc:
                        st.write("**Recipients:**")
                        for recipient in esc.get('recipients', []):
                            st.write(f"- {recipient.get('name', 'Unknown')}: {recipient.get('email', 'No email')}")
                    
                    if 'results' in esc and esc['results']:
                        st.write("**Results:**")
                        for channel, result in esc.get('results', {}).items():
                            status = result.get('status', 'unknown')
                            status_color = "green" if status == "success" else "red"
                            st.markdown(f"- {channel}: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
        
        # Show stats
        stats = self.escalation_router.get_escalation_stats()
        
        if stats:
            st.subheader("Escalation Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Escalations", stats.get('total_escalations', 0))
                
            with col2:
                success_rate = 0
                if stats.get('total_escalations', 0) > 0:
                    success_rate = (stats.get('successful', 0) / stats.get('total_escalations', 0)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
                
            with col3:
                st.metric("Critical Escalations", stats.get('by_level', {}).get('critical', 0))
                
            # Level distribution
            level_data = stats.get('by_level', {})
            st.write("**Escalation Levels:**")
            for level, count in level_data.items():
                if count > 0:
                    st.write(f"- {level.capitalize()}: {count}")
                    
            # Channel distribution
            channel_data = stats.get('by_channel', {})
            st.write("**Channels Used:**")
            for channel, count in channel_data.items():
                if count > 0:
                    st.write(f"- {channel.capitalize()}: {count}")