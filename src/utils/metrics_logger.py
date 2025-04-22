"""
Metrics logging system for query execution auditing.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import psutil
from logging.handlers import RotatingFileHandler

@dataclass
class QueryMetrics:
    """Metrics collected for a query execution."""
    query_id: str
    timestamp: str
    execution_time_ms: float
    memory_mb: float
    llm_tokens_used: int
    cache_hit: bool
    status: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    trace_id: Optional[str] = None
    data_quality: Optional[Dict[str, Any]] = None  # Added data quality metrics
    nan_percentage: Optional[float] = None  # Added NaN tracking
    excluded_rows: Optional[int] = None  # Added excluded rows tracking

class MetricsLogger:
    """Logger for query execution metrics."""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure JSON logger
        self.logger = logging.getLogger("metrics")
        self.logger.setLevel(logging.INFO)
        
        # Add rotating file handler
        handler = RotatingFileHandler(
            os.path.join(log_dir, "query_metrics.json"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_query(
        self,
        query_id: str,
        query: str,
        result: Dict[str, Any],
        start_time: float,
        llm_tokens: int,
        cache_hit: bool,
        trace_id: Optional[str] = None,
        error: Optional[Exception] = None,
        data_quality: Optional[Dict[str, Any]] = None,
        nan_percentage: Optional[float] = None,
        excluded_rows: Optional[int] = None
    ) -> None:
        """
        Log metrics for a query execution.
        
        Args:
            query_id: Unique query identifier
            query: The query text
            result: Query result
            start_time: Query start timestamp
            llm_tokens: Number of LLM tokens used
            cache_hit: Whether result was from cache
            trace_id: Optional trace ID for linking related operations
            error: Optional exception if query failed
            data_quality: Optional data quality metrics
            nan_percentage: Optional percentage of NaN values
            excluded_rows: Optional count of excluded rows
        """
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Create metrics object
        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time,
            memory_mb=memory_mb,
            llm_tokens_used=llm_tokens,
            cache_hit=cache_hit,
            status="error" if error else "success",
            error_code=error.__class__.__name__ if error else None,
            error_message=str(error) if error else None,
            trace_id=trace_id,
            data_quality=data_quality,
            nan_percentage=nan_percentage,
            excluded_rows=excluded_rows
        )
        
        # Log metrics as JSON
        self.logger.info(json.dumps({
            "query": query,
            "metrics": asdict(metrics),
            "result": result
        }))
        
        return metrics

    def get_metrics_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for a trace ID."""
        metrics = []
        log_file = os.path.join(self.log_dir, "query_metrics.json")
        
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry["metrics"]["trace_id"] == trace_id:
                            metrics.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return metrics

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent metric entries."""
        metrics = []
        log_file = os.path.join(self.log_dir, "query_metrics.json")
        
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return sorted(
            metrics,
            key=lambda x: x["metrics"]["timestamp"],
            reverse=True
        )[:limit]

    def get_data_quality_trends(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metric_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get data quality trends over time.
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            metric_type: Optional metric type filter
            
        Returns:
            Dictionary of data quality metrics over time
        """
        trends = {
            "nan_percentage": [],
            "excluded_rows": [],
            "quality_scores": []
        }
        
        log_file = os.path.join(self.log_dir, "query_metrics.json")
        
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        metrics = entry["metrics"]
                        
                        # Apply date filters if provided
                        if start_date and metrics["timestamp"] < start_date:
                            continue
                        if end_date and metrics["timestamp"] > end_date:
                            continue
                            
                        # Add metrics to trends
                        if metrics.get("nan_percentage") is not None:
                            trends["nan_percentage"].append({
                                "timestamp": metrics["timestamp"],
                                "value": metrics["nan_percentage"]
                            })
                            
                        if metrics.get("excluded_rows") is not None:
                            trends["excluded_rows"].append({
                                "timestamp": metrics["timestamp"],
                                "value": metrics["excluded_rows"]
                            })
                            
                        if metrics.get("data_quality"):
                            trends["quality_scores"].append({
                                "timestamp": metrics["timestamp"],
                                "value": metrics["data_quality"].get("score", 0)
                            })
                            
                    except json.JSONDecodeError:
                        continue
        
        return trends

# Global metrics logger instance
metrics_logger = MetricsLogger()