"""
Traceability engine for tracking insight generation steps and metrics.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class TraceabilityEngine:
    """Engine for tracking and analyzing insight generation traces."""
    
    def __init__(self):
        """Initialize the traceability engine."""
        self._traces = {}  # In-memory storage for traces
        self._metrics = {}  # Associated metrics for each trace
        
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace data for a given trace ID.
        
        Args:
            trace_id: The ID of the trace to retrieve
            
        Returns:
            Dict containing trace data if found, None otherwise
        """
        trace = self._traces.get(trace_id)
        metrics = self._metrics.get(trace_id)
        
        if not trace:
            return None
            
        trace_data = {
            'trace_id': trace_id,
            'start_time': trace['start_time'].isoformat(),
            'steps': [
                {
                    'timestamp': step['timestamp'].isoformat(),
                    'description': step['description'],
                    'input_data': step.get('input_data'),
                    'output_data': step.get('output_data'),
                    'metrics': step.get('metrics', {})
                }
                for step in trace['steps']
            ]
        }
        
        if metrics:
            trace_data['metrics'] = metrics
            
        return trace_data
    
    def start_trace(self, trace_id: str, query: Optional[str] = None) -> None:
        """
        Start a new trace.
        
        Args:
            trace_id: Unique identifier for the trace
            query: Optional query associated with this trace
        """
        self._traces[trace_id] = {
            'start_time': datetime.now(),
            'steps': [],
            'query': query
        }
        
    def add_step(self, trace_id: str, description: str, 
                input_data: Optional[Dict] = None, 
                output_data: Optional[Dict] = None,
                metrics: Optional[Dict] = None) -> None:
        """
        Add a step to an existing trace.
        
        Args:
            trace_id: ID of the trace to add step to
            description: Description of the step
            input_data: Optional input data for this step
            output_data: Optional output data from this step
            metrics: Optional metrics for this step
        """
        if trace_id not in self._traces:
            return
            
        step = {
            'timestamp': datetime.now(),
            'description': description
        }
        
        if input_data:
            step['input_data'] = input_data
        if output_data:
            step['output_data'] = output_data
        if metrics:
            step['metrics'] = metrics
            
        self._traces[trace_id]['steps'].append(step)
        
    def add_metrics(self, trace_id: str, metrics: Dict[str, Any]) -> None:
        """
        Add metrics data for a trace.
        
        Args:
            trace_id: ID of the trace to add metrics to
            metrics: Dictionary of metrics data
        """
        if trace_id not in self._traces:
            return
            
        self._metrics[trace_id] = metrics
        
    def get_metrics_by_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics data for a specific trace.
        
        Args:
            trace_id: ID of the trace to get metrics for
            
        Returns:
            Dict containing metrics data if found, None otherwise
        """
        return self._metrics.get(trace_id)
        
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace data dictionaries
        """
        sorted_traces = sorted(
            self._traces.items(),
            key=lambda x: x[1]['start_time'],
            reverse=True
        )[:limit]
        
        return [
            self.get_trace(trace_id)
            for trace_id, _ in sorted_traces
        ] 