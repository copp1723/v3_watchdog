"""
Module for generating automated briefings from top insights.
Supports output as Slack messages, email content, or dashboard cards.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import datetime
import pandas as pd

class DigestBot:
    """Generates automated briefings from a collection of insights."""
    
    def __init__(self, insights: List[Dict[str, Any]], max_insights: int = 5):
        """Initialize the DigestBot with a list of insights."""
        self.insights = insights
        self.max_insights = max_insights
        self.digest_title = f"Morning Briefing - {datetime.datetime.now().strftime('%Y-%m-%d')}"

    def filter_top_insights(self) -> List[Dict[str, Any]]:
        """Filter and prioritize the top insights based on risk_flag and timestamp."""
        if not self.insights:
            return []
        
        # Sort insights by risk_flag (True first) and timestamp (newest first)
        sorted_insights = sorted(
            self.insights,
            key=lambda x: (-x.get('risk_flag', False), x.get('timestamp', '1970-01-01T00:00:00Z')),
            reverse=True
        )
        
        # Return the top N insights
        return sorted_insights[:self.max_insights]

    def format_slack_message(self, insights: List[Dict[str, Any]]) -> str:
        """Format the top insights as a Slack message."""
        if not insights:
            return f"*{self.digest_title}*\n\nNo insights available for this briefing."
        
        message = [f"*{self.digest_title}*\n"]
        message.append(f"Here are the top {len(insights)} insights for today:\n")
        
        for i, insight in enumerate(insights, 1):
            summary = insight.get('summary', 'No summary available.')
            recommendation = insight.get('recommendation', 'No recommendation provided.')
            risk_flag = insight.get('risk_flag', False)
            
            message.append(f"*{i}. {'🚨 ' if risk_flag else ''}Insight:*")
            message.append(f"  - {summary[:150]}{'...' if len(summary) > 150 else ''}")
            message.append(f"  - *Recommendation:* {recommendation[:100]}{'...' if len(recommendation) > 100 else ''}\n")
        
        message.append("Check the dashboard for full details and follow-up questions.")
        return "\n".join(message)
    
    def format_email_content(self, insights: List[Dict[str, Any]]) -> str:
        """
        Format the top insights as email content (HTML).
        
        Args:
            insights: List of top insights to include
            
        Returns:
            Formatted HTML email content string
        """
        if not insights:
            return f"""
<html>
<body>
    <h2>{self.digest_title}</h2>
    <p>No insights available for this briefing.</p>
</body>
</html>
"""
        
        content = f"""
<html>
<body>
    <h2>{self.digest_title}</h2>
    <p>Here are the top {len(insights)} insights for today:</p>
    <ol>
"""
        
        for i, insight in enumerate(insights):
            summary = insight.get('summary', 'No summary available.')
            recommendation = insight.get('recommendation', 'No recommendation provided.')
            risk_flag = insight.get('risk_flag', False)
            
            content += f"""
    <li>
        <strong>{'🚨 ' if risk_flag else ''}Insight:</strong>
        <p>{summary[:200]}{'...' if len(summary) > 200 else ''}</p>
        <p><em>Recommendation:</em> {recommendation[:150]}{'...' if len(recommendation) > 150 else ''}</p>
    </li>
"""
        
        content += """
    </ol>
    <p>Check the dashboard for full details and follow-up questions.</p>
</body>
</html>
"""
        return content
    
    def format_dashboard_card(self, insights: List[Dict[str, Any]]) -> str:
        """
        Format the top insights as a dashboard card (markdown).
        
        Args:
            insights: List of top insights to include
            
        Returns:
            Formatted markdown string for dashboard card
        """
        if not insights:
            return f"### {self.digest_title}\n\nNo insights available for this briefing."
        
        card = f"### {self.digest_title}\n\n"
        card += f"**Top {len(insights)} Insights for Today:**\n\n"
        
        for i, insight in enumerate(insights, 1):
            summary = insight.get('summary', 'No summary available.')
            recommendation = insight.get('recommendation', 'No recommendation provided.')
            risk_flag = insight.get('risk_flag', False)
            
            card += f"{i}. **{'🚨 ' if risk_flag else ''}Insight:** {summary[:100]}{'...' if len(summary) > 100 else ''}\n"
            card += f"   - *Recommendation:* {recommendation[:75]}{'...' if len(recommendation) > 75 else ''}\n\n"
        
        card += "*See full details in the Insights app.*"
        return card
    
    def generate_digest(self, output_format: str = 'slack') -> str:
        """
        Generate a digest of top insights in the specified format.
        
        Args:
            output_format: Format of the digest ('slack', 'email', or 'dashboard')
            
        Returns:
            Formatted digest content as a string
        """
        # Filter to top insights
        top_insights = self.filter_top_insights()
        
        # Format based on requested output
        if output_format.lower() == 'slack':
            return self.format_slack_message(top_insights)
        elif output_format.lower() == 'email':
            return self.format_email_content(top_insights)
        elif output_format.lower() == 'dashboard':
            return self.format_dashboard_card(top_insights)
        else:
            raise ValueError(f"Unsupported output format: {output_format}. Use 'slack', 'email', or 'dashboard'.")
    
    def render_digest_preview(self, output_format: str = 'slack') -> None:
        """
        Render a preview of the digest in Streamlit for review.
        
        Args:
            output_format: Format of the digest to preview ('slack', 'email', or 'dashboard')
        """
        digest_content = self.generate_digest(output_format)
        
        st.markdown(f"### Digest Preview ({output_format.title()} Format)")
        with st.expander("View Digest Content", expanded=True):
            if output_format.lower() == 'email':
                st.markdown("*Note: Email format is HTML. Preview may not render perfectly in markdown.*")
            st.markdown(digest_content, unsafe_allow_html=(output_format.lower() == 'email'))
        
        st.markdown("#### Export Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export as Slack Message"):
                st.session_state['exported_digest'] = self.generate_digest('slack')
                st.info("Exported as Slack message. Copy from the text area below.")
        with col2:
            if st.button("Export as Email Content"):
                st.session_state['exported_digest'] = self.generate_digest('email')
                st.info("Exported as Email content. Copy from the text area below.")
        with col3:
            if st.button("Export as Dashboard Card"):
                st.session_state['exported_digest'] = self.generate_digest('dashboard')
                st.info("Exported as Dashboard card. Copy from the text area below.")
        
        if 'exported_digest' in st.session_state:
            st.text_area("Exported Digest Content", st.session_state['exported_digest'], height=200)

def run_nightly_digest(history: List[Dict[str, Any]], output_format: str = 'slack', max_insights: int = 5) -> str:
    """
    Simulate a nightly job to generate a digest from recent insights.
    
    Args:
        history: List of conversation history entries (from st.session_state['conversation_history'])
        output_format: Format of the digest ('slack', 'email', or 'dashboard')
        max_insights: Maximum number of insights to include
    
    Returns:
        Formatted digest content
    """
    # Extract insights from conversation history
    insights = [entry['response'] for entry in history if 'response' in entry]
    
    # Filter by recent insights (e.g., last 24 hours)
    # For simplicity, we're taking all insights since this is a mock
    bot = DigestBot(insights, max_insights=max_insights)
    return bot.generate_digest(output_format)

def render_digest_bot_ui(history: List[Dict[str, Any]]):
    """
    Render a UI in Streamlit for generating and previewing digests.
    
    Args:
        history: List of conversation history entries to summarize
    """
    st.markdown("# 📋 Insight Digest Bot")
    st.markdown("Generate automated briefings from your top insights for Slack, email, or dashboards.")
    
    # Extract insights from history
    insights = [entry['response'] for entry in history if 'response' in entry]
    
    if not insights:
        st.warning("No insights available to summarize. Generate some insights first.")
        return
    
    # UI for configuration
    with st.container():
        st.markdown("### Configuration")
        max_insights = st.slider("Max Insights to Include", min_value=1, max_value=10, value=5)
        output_format = st.selectbox("Output Format", ["Slack", "Email", "Dashboard"], index=0)
        
        bot = DigestBot(insights, max_insights=max_insights)
        
        if st.button("Generate Digest Preview"):
            st.session_state['digest_bot'] = bot
            st.session_state['output_format'] = output_format.lower()
            st.experimental_rerun()
    
    # Render preview if available
    if 'digest_bot' in st.session_state and 'output_format' in st.session_state:
        st.session_state['digest_bot'].render_digest_preview(st.session_state['output_format'])