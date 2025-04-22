"""
Debug Panel UI for Watchdog AI insights.
Provides detailed execution tracing and debugging information.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from watchdog_ai.ui.utils.status_formatter import StatusType, format_status_text

# Constants for execution limits
TIME_LIMIT = 30  # seconds
MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB

class InsightDebugPanel:
    """UI component for debugging insight generation."""
    
    def __init__(self):
        """Initialize the debug panel."""
        if 'debug_history' not in st.session_state:
            st.session_state.debug_history = []
    
    def add_trace(self, trace_data: Dict[str, Any]) -> None:
        """
        Add a trace to the debug history.
        
        Args:
            trace_data: Trace information to add
        """
        trace_data['timestamp'] = datetime.now().isoformat()
        st.session_state.debug_history.append(trace_data)
    
    def render_query_flow(self, original_query: str, rephrased_query: str,
                         steps: List[Dict[str, Any]]) -> None:
        """
        Render the query processing flow.
        
        Args:
            original_query: Original user query
            rephrased_query: Rephrased query for LLM
            steps: List of processing steps
        """
        st.subheader("Query Processing Flow")
        
        # Show queries
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Original Query", original_query, disabled=True)
        with col2:
            st.text_area("Rephrased Query", rephrased_query, disabled=True)
        
        # Show processing steps
        with st.expander("View Processing Steps", expanded=False):
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}: {step['name']}**")
                st.markdown(f"_{step['description']}_")
                if 'output' in step:
                    st.code(json.dumps(step['output'], indent=2), language='json')
                st.markdown("---")
    
    def render_code_preview(self, code: str, analysis_result: Dict[str, Any]) -> None:
        """
        Render the generated code with analysis results.
        
        Args:
            code: Generated Python code
            analysis_result: Code analysis results
        """
        st.subheader("Generated Code")
        
        # Show code with syntax highlighting
        st.code(code, language='python')
        
        # Show analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Security Score",
                f"{100 if analysis_result['is_safe'] else 0}%",
                delta="Safe" if analysis_result['is_safe'] else "Issues Found",
                delta_color="normal" if analysis_result['is_safe'] else "inverse"
            )
        
        with col2:
            st.metric(
                "Imported Modules",
                len(analysis_result['imported_modules']),
                delta=None
            )
        
        if analysis_result['issues']:
            warning_text = f"{format_status_text(StatusType.WARNING)} Security Issues Found"
            st.markdown(warning_text, unsafe_allow_html=True)
            for issue in analysis_result['issues']:
                st.markdown(f"- {issue}")
    
    def render_execution_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Render execution performance metrics.
        
        Args:
            metrics: Execution metrics data
        """
        st.subheader("Execution Metrics")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Execution Time",
                f"{metrics['execution_time_ms']:.2f}ms",
                delta=f"{TIME_LIMIT * 1000 - metrics['execution_time_ms']:.0f}ms remaining",
                delta_color="normal"
            )
        
        with col2:
            memory_mb = metrics['memory_used'] / 1024 / 1024
            memory_limit_mb = MEMORY_LIMIT / 1024 / 1024
            st.metric(
                "Memory Usage",
                f"{memory_mb:.1f}MB",
                delta=f"{memory_limit_mb - memory_mb:.1f}MB free",
                delta_color="normal"
            )
        
        with col3:
            cache_status = "Hit" if metrics.get('cache_hit', False) else "Miss"
            status_type = StatusType.SUCCESS if metrics.get('cache_hit', False) else StatusType.INFO
            formatted_status = format_status_text(status_type, custom_text=cache_status, include_brackets=False)
            
            st.metric(
                "Cache Status",
                cache_status,
                delta=None
            )
        
        # Show execution timeline
        if 'timeline' in metrics:
            fig = go.Figure()
            
            for event in metrics['timeline']:
                fig.add_trace(go.Scatter(
                    x=[event['start'], event['end']],
                    y=[event['phase']],
                    mode='lines',
                    name=event['phase'],
                    text=f"{event['duration_ms']:.1f}ms"
                ))
            
            fig.update_layout(
                title="Execution Timeline",
                xaxis_title="Time (ms)",
                yaxis_title="Phase",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_error_details(self, error: Dict[str, Any]) -> None:
        """
        Render error details with suggestions.
        
        Args:
            error: Error information
        """
        error_text = f"{format_status_text(StatusType.ERROR, custom_text='ERROR')} Error Details"
        st.markdown(error_text, unsafe_allow_html=True)
        
        st.error(
            f"**{error['error_type']}**: {error['error_message']}"
        )
        
        with st.expander("View Traceback", expanded=False):
            st.code(error['traceback'], language='python')
        
        if 'suggestions' in error:
            st.subheader("Suggestions")
            for suggestion in error['suggestions']:
                st.markdown(f"- {suggestion}")
    
    def render_trace_history(self) -> None:
        """Render the execution trace history."""
        st.subheader("Trace History")
        
        if not st.session_state.debug_history:
            st.info("No trace history available")
            return
        
        # Create a DataFrame from history
        df = pd.DataFrame(st.session_state.debug_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Show as interactive table
        st.dataframe(
            df.sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Add option to clear history
        if st.button("Clear History"):
            st.session_state.debug_history = []
            st.rerun()
    
    def render_debug_panel(self, insight_execution: Dict[str, Any]) -> None:
        """
        Render the complete debug panel.
        
        Args:
            insight_execution: Complete insight execution data
        """
        st.title("Insight Debug Panel")
        
        # Add trace to history
        self.add_trace(insight_execution)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Query Flow",
            "Code Analysis",
            "Execution Metrics",
            "History"
        ])
        
        with tab1:
            self.render_query_flow(
                insight_execution['original_query'],
                insight_execution['rephrased_query'],
                insight_execution['processing_steps']
            )
        
        with tab2:
            self.render_code_preview(
                insight_execution['generated_code'],
                insight_execution['code_analysis']
            )
        
        with tab3:
            self.render_execution_metrics(insight_execution['metrics'])
            
            if 'error' in insight_execution:
                self.render_error_details(insight_execution['error'])
        
        with tab4:
            self.render_trace_history()