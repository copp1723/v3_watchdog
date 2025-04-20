"""
Traceability system for Watchdog AI insights.
Tracks and versions insight generation steps.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import uuid
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TraceStep:
    """Represents a single step in insight generation."""
    step_id: str
    step_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: str
    metrics: Dict[str, Any]

@dataclass
class InsightTrace:
    """Complete trace of insight generation."""
    trace_id: str
    query: str
    steps: List[TraceStep]
    final_result: Dict[str, Any]
    start_time: str
    end_time: str
    version: str
    metadata: Dict[str, Any]

class TraceabilityEngine:
    """Manages insight generation tracing and versioning."""
    
    def __init__(self, trace_dir: str = "traces"):
        """
        Initialize the traceability engine.
        
        Args:
            trace_dir: Directory to store trace files
        """
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        
        # Active trace being recorded
        self.active_trace: Optional[InsightTrace] = None
        
        # Load existing traces
        self.traces: Dict[str, InsightTrace] = self._load_traces()
    
    def _load_traces(self) -> Dict[str, InsightTrace]:
        """Load existing traces from disk."""
        traces = {}
        
        try:
            for filename in os.listdir(self.trace_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(self.trace_dir, filename)) as f:
                        data = json.load(f)
                        trace = InsightTrace(**data)
                        traces[trace.trace_id] = trace
        except Exception as e:
            logger.error(f"Error loading traces: {e}")
        
        return traces
    
    def start_trace(self, query: str, metadata: Dict[str, Any]) -> str:
        """
        Start a new trace.
        
        Args:
            query: Query being processed
            metadata: Additional trace metadata
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        
        self.active_trace = InsightTrace(
            trace_id=trace_id,
            query=query,
            steps=[],
            final_result={},
            start_time=datetime.now().isoformat(),
            end_time="",
            version="1.0.0",
            metadata=metadata
        )
        
        return trace_id
    
    def add_step(self, step_type: str, input_data: Dict[str, Any],
                 output_data: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Add a step to the active trace.
        
        Args:
            step_type: Type of step
            input_data: Step input data
            output_data: Step output data
            metrics: Step execution metrics
        """
        if not self.active_trace:
            logger.error("No active trace to add step to")
            return
        
        step = TraceStep(
            step_id=str(uuid.uuid4()),
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )
        
        self.active_trace.steps.append(step)
    
    def end_trace(self, final_result: Dict[str, Any]) -> None:
        """
        End the active trace.
        
        Args:
            final_result: Final insight result
        """
        if not self.active_trace:
            logger.error("No active trace to end")
            return
        
        self.active_trace.final_result = final_result
        self.active_trace.end_time = datetime.now().isoformat()
        
        # Save trace
        self._save_trace(self.active_trace)
        
        # Add to loaded traces
        self.traces[self.active_trace.trace_id] = self.active_trace
        
        # Clear active trace
        self.active_trace = None
    
    def _save_trace(self, trace: InsightTrace) -> None:
        """Save a trace to disk."""
        try:
            filename = f"trace_{trace.trace_id}.json"
            filepath = os.path.join(self.trace_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(trace.__dict__, f, indent=2)
                
            logger.info(f"Saved trace to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving trace: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[InsightTrace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)
    
    def list_traces(self) -> List[Dict[str, Any]]:
        """Get list of all traces."""
        return [
            {
                "trace_id": t.trace_id,
                "query": t.query,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "version": t.version,
                "step_count": len(t.steps)
            }
            for t in self.traces.values()
        ]
    
    def save_version(self, trace_id: str, version: str) -> None:
        """
        Save a specific version of a trace.
        
        Args:
            trace_id: Trace ID to version
            version: Version string
        """
        trace = self.traces.get(trace_id)
        if not trace:
            logger.error(f"Trace {trace_id} not found")
            return
        
        # Create versioned copy
        versioned_trace = InsightTrace(
            trace_id=f"{trace_id}_v{version}",
            query=trace.query,
            steps=trace.steps.copy(),
            final_result=trace.final_result.copy(),
            start_time=trace.start_time,
            end_time=trace.end_time,
            version=version,
            metadata=trace.metadata.copy()
        )
        
        # Save versioned trace
        self._save_trace(versioned_trace)
        
        # Add to loaded traces
        self.traces[versioned_trace.trace_id] = versioned_trace
    
    def compare_versions(self, trace_id: str, version1: str,
                        version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a trace.
        
        Args:
            trace_id: Base trace ID
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        trace1 = self.traces.get(f"{trace_id}_v{version1}")
        trace2 = self.traces.get(f"{trace_id}_v{version2}")
        
        if not trace1 or not trace2:
            return {"error": "One or both versions not found"}
        
        # Compare steps
        step_comparison = []
        for s1, s2 in zip(trace1.steps, trace2.steps):
            if s1.step_type == s2.step_type:
                step_comparison.append({
                    "step_type": s1.step_type,
                    "input_changed": s1.input_data != s2.input_data,
                    "output_changed": s1.output_data != s2.output_data,
                    "metrics_v1": s1.metrics,
                    "metrics_v2": s2.metrics
                })
        
        return {
            "trace_id": trace_id,
            "version1": version1,
            "version2": version2,
            "steps_compared": len(step_comparison),
            "step_changes": step_comparison,
            "result_changed": trace1.final_result != trace2.final_result
        }