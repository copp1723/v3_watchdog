"""
Debug Dashboard for insight generation monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from ..utils.metrics_logger import metrics_logger
from watchdog_ai.insights.traceability import TraceabilityEngine

class DebugDashboard:
    """Debug dashboard for monitoring insight generation."""
    
    def __init__(self):
        """Initialize the debug dashboard."""
        self.trace_engine = TraceabilityEngine()
        
        if 'selected_trace' not in st.session_state:
            st.session_state.selected_trace = None
    
    def render_dashboard(self):
        """Render the debug dashboard."""
        st.title("Insight Debug Dashboard")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "Execution Metrics",
            "Trace Analysis",
            "Cache Statistics"
        ])
        
        with tab1:
            self._render_execution_metrics()
        
        with tab2:
            self._render_trace_analysis()
        
        with tab3:
            self._render_cache_statistics()
    
    def _render_execution_metrics(self):
        """Render execution metrics section."""
        st.subheader("Execution Metrics")
        
        # Get recent metrics
        recent_metrics = metrics_logger.get_recent_metrics(limit=100)
        if not recent_metrics:
            st.info("No metrics data available")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([m["metrics"] for m in recent_metrics])
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Avg Response Time",
                f"{df['execution_time_ms'].mean():.0f}ms"
            )
        
        with col2:
            st.metric(
                "Cache Hit Rate",
                f"{(df['cache_hit'].mean() * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{(df['status'].value_counts().get('success', 0) / len(df) * 100):.1f}%"
            )
        
        # Response time trend
        st.subheader("Response Time Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df['timestamp']),
            y=df['execution_time_ms'],
            mode='lines+markers',
            name='Response Time'
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Response Time (ms)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        errors = df[df['status'] == 'error']
        if not errors.empty:
            st.subheader("Error Distribution")
            error_counts = errors['error_code'].value_counts()
            fig = go.Figure(data=[
                go.Bar(x=error_counts.index, y=error_counts.values)
            ])
            fig.update_layout(
                xaxis_title="Error Type",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_trace_analysis(self):
        """Render trace analysis section."""
        st.subheader("Trace Analysis")
        
        # Trace selector
        trace_id = st.text_input(
            "Enter Trace ID",
            value=st.session_state.selected_trace or ""
        )
        
        if trace_id:
            st.session_state.selected_trace = trace_id
            
            # Get trace metrics
            trace_metrics = metrics_logger.get_metrics_by_trace(trace_id)
            
            if not trace_metrics:
                st.warning("No metrics found for this trace ID")
                return
            
            # Show trace timeline
            st.subheader("Trace Timeline")
            
            # Get trace data from TraceabilityEngine
            trace_data = self.trace_engine.get_trace(trace_id)
            if trace_data:
                # Show trace details
                st.write("Trace Details:")
                st.json({
                    "query": trace_data.query,
                    "start_time": trace_data.start_time,
                    "end_time": trace_data.end_time,
                    "version": trace_data.version,
                    "step_count": len(trace_data.steps)
                })
                
                # Show steps
                for i, step in enumerate(trace_data.steps):
                    with st.expander(
                        f"Step {i+1}: {step.step_type}",
                        expanded=i == 0
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Input:")
                            st.json(step.input_data)
                            st.write("Metrics:")
                            st.json(step.metrics)
                        
                        with col2:
                            st.write("Output:")
                            st.json(step.output_data)
                
                # Show final result
                if trace_data.final_result:
                    with st.expander("Final Result", expanded=True):
                        st.json(trace_data.final_result)
                
                # Show trace visualization
                st.subheader("Trace Visualization")
                self._render_trace_visualization(trace_data)
            
            # Show metrics timeline
            st.subheader("Metrics Timeline")
            for i, entry in enumerate(trace_metrics):
                metrics = entry["metrics"]
                with st.expander(
                    f"Step {i+1}: {entry['query'][:50]}...",
                    expanded=i == 0
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Metrics:")
                        st.json({
                            "execution_time": f"{metrics['execution_time_ms']:.0f}ms",
                            "memory_used": f"{metrics['memory_mb']:.1f}MB",
                            "llm_tokens": metrics['llm_tokens_used'],
                            "cache_hit": metrics['cache_hit']
                        })
                    
                    with col2:
                        st.write("Result:")
                        st.json(entry["result"])
    
    def _render_cache_statistics(self):
        """Render cache statistics section."""
        st.subheader("Cache Statistics")
        
        # Get recent metrics for cache analysis
        recent_metrics = metrics_logger.get_recent_metrics(limit=1000)
        if not recent_metrics:
            st.info("No cache statistics available")
            return
        
        df = pd.DataFrame([m["metrics"] for m in recent_metrics])
        
        # Cache hit rate over time
        st.subheader("Cache Hit Rate Over Time")
        
        # Group by hour and calculate hit rate
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        hit_rate = df.groupby('hour')['cache_hit'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hit_rate.index,
            y=hit_rate.values * 100,
            mode='lines+markers',
            name='Cache Hit Rate'
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cache Hit Rate (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cache efficiency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            cache_hits = df['cache_hit'].sum()
            total_queries = len(df)
            tokens_saved = df[df['cache_hit']]['llm_tokens_used'].sum()
            
            st.metric("Cache Hit Rate", f"{(cache_hits/total_queries*100):.1f}%")
            st.metric("Tokens Saved", f"{tokens_saved:,}")
        
        with col2:
            avg_hit_latency = df[df['cache_hit']]['execution_time_ms'].mean()
            avg_miss_latency = df[~df['cache_hit']]['execution_time_ms'].mean()
            
            st.metric("Avg Cache Hit Latency", f"{avg_hit_latency:.0f}ms")
            st.metric("Avg Cache Miss Latency", f"{avg_miss_latency:.0f}ms")
    
    def _render_trace_visualization(self, trace_data: Any):
        """Render visualization of a trace."""
        # Create a timeline visualization
        events = []
        
        # Convert trace data to events
        if hasattr(trace_data, 'steps'):
            start_time = datetime.fromisoformat(trace_data.start_time)
            
            for step in trace_data.steps:
                step_time = datetime.fromisoformat(step.timestamp)
                events.append({
                    'Task': step.step_type,
                    'Start': start_time,
                    'Finish': step_time,
                    'Description': f"{step.step_type} ({step.metrics.get('duration_ms', 0)}ms)"
                })
                start_time = step_time
        else:
            # Handle old trace data format
            start_time = datetime.fromisoformat(trace_data['start_time'])
            for step in trace_data['steps']:
                step_time = datetime.fromisoformat(step['timestamp'])
                events.append({
                    'Task': 'Processing',
                    'Start': start_time,
                    'Finish': step_time,
                    'Description': step['description']
                })
                start_time = step_time
        
        df = pd.DataFrame(events)
        
        fig = go.Figure()
        
        for idx, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Finish'] - row['Start']],
                y=[row['Task']],
                orientation='h',
                name=row['Description'],
                showlegend=True
            ))
        
        fig.update_layout(
            title="Trace Timeline",
            xaxis_title="Duration",
            yaxis_title="",
            height=200,
            barmode='stack',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)