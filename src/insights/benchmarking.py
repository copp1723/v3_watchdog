"""
Cross-dealership benchmarking engine for metric comparison and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetric:
    """Represents a benchmark metric with comparison data."""
    name: str
    value: float
    percentile: float
    benchmark_avg: float
    benchmark_median: float
    benchmark_p75: float
    benchmark_p90: float
    trend: Optional[float] = None
    anomaly_score: Optional[float] = None

@dataclass
class BenchmarkResult:
    """Contains benchmark analysis results."""
    metrics: List[BenchmarkMetric]
    timestamp: str
    dealership_id: str
    comparison_period: str
    comparison_group: str
    sample_size: int

class BenchmarkEngine:
    """
    Engine for comparing metrics across dealerships and detecting anomalies.
    """
    
    def __init__(self):
        """Initialize the benchmark engine."""
        self.metrics_cache: Dict[str, pd.DataFrame] = {}
        
    def _get_comparison_data(self, metric: str, dealership_ids: Optional[List[str]] = None,
                           start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get comparison data for benchmarking.
        
        Args:
            metric: Name of the metric to compare
            dealership_ids: Optional list of dealership IDs to include
            start_date: Optional start date for comparison
            end_date: Optional end date for comparison
            
        Returns:
            DataFrame with comparison data
        """
        # In a real implementation, this would query a database
        # For now, we'll use cached data if available
        if metric in self.metrics_cache:
            df = self.metrics_cache[metric].copy()
        else:
            # Create sample data for testing
            df = pd.DataFrame({
                'dealership_id': ['d1', 'd2', 'd3', 'd4', 'd5'],
                'value': [100, 150, 90, 200, 175],
                'date': pd.date_range(start='2024-01-01', periods=5)
            })
            self.metrics_cache[metric] = df
            
        # Apply filters
        if dealership_ids:
            df = df[df['dealership_id'].isin(dealership_ids)]
            
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
        return df
    
    def calculate_benchmarks(self, dealership_id: str, metrics: List[str],
                           comparison_group: Optional[List[str]] = None,
                           period: str = "30d") -> BenchmarkResult:
        """
        Calculate benchmarks for specified metrics.
        
        Args:
            dealership_id: ID of the dealership to benchmark
            metrics: List of metrics to benchmark
            comparison_group: Optional list of dealership IDs to compare against
            period: Time period for comparison (e.g., "30d", "90d")
            
        Returns:
            BenchmarkResult with analysis
        """
        # Calculate date range
        end_date = datetime.now()
        days = int(period.rstrip('d'))
        start_date = end_date - timedelta(days=days)
        
        benchmark_metrics = []
        for metric in metrics:
            # Get comparison data
            df = self._get_comparison_data(
                metric,
                dealership_ids=comparison_group,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            if df.empty:
                logger.warning(f"No comparison data for metric: {metric}")
                continue
            
            # Calculate statistics
            stats = df['value'].describe(percentiles=[0.75, 0.9])
            
            # Get dealership's value
            dealership_value = float(
                df[df['dealership_id'] == dealership_id]['value'].mean()
            )
            
            # Calculate percentile
            percentile = float(
                (df['value'] <= dealership_value).mean() * 100
            )
            
            # Calculate trend
            if len(df) >= 2:
                values = df[df['dealership_id'] == dealership_id]['value']
                trend = float((values.iloc[-1] / values.iloc[0] - 1) * 100)
            else:
                trend = None
            
            # Calculate anomaly score
            mean = stats['mean']
            std = stats['std']
            if std > 0:
                anomaly_score = float(abs(dealership_value - mean) / std)
            else:
                anomaly_score = None
            
            benchmark_metrics.append(BenchmarkMetric(
                name=metric,
                value=dealership_value,
                percentile=percentile,
                benchmark_avg=float(stats['mean']),
                benchmark_median=float(stats['50%']),
                benchmark_p75=float(stats['75%']),
                benchmark_p90=float(stats['90%']),
                trend=trend,
                anomaly_score=anomaly_score
            ))
        
        return BenchmarkResult(
            metrics=benchmark_metrics,
            timestamp=datetime.now().isoformat(),
            dealership_id=dealership_id,
            comparison_period=period,
            comparison_group="custom" if comparison_group else "all",
            sample_size=len(df['dealership_id'].unique())
        )
    
    def get_improvement_targets(self, benchmark_result: BenchmarkResult,
                              target_percentile: float = 75) -> Dict[str, float]:
        """
        Calculate improvement targets for metrics.
        
        Args:
            benchmark_result: Benchmark analysis result
            target_percentile: Percentile to target (default 75th)
            
        Returns:
            Dictionary of metric names to target values
        """
        targets = {}
        for metric in benchmark_result.metrics:
            if metric.percentile < target_percentile:
                if target_percentile == 75:
                    target_value = metric.benchmark_p75
                elif target_percentile == 90:
                    target_value = metric.benchmark_p90
                else:
                    # For other percentiles, estimate using average and median
                    target_value = (metric.benchmark_avg + metric.benchmark_median) / 2
                
                improvement_needed = target_value - metric.value
                targets[metric.name] = improvement_needed
                
        return targets
    
    def detect_anomalies(self, benchmark_result: BenchmarkResult,
                        threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalous metrics.
        
        Args:
            benchmark_result: Benchmark analysis result
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        for metric in benchmark_result.metrics:
            if metric.anomaly_score and metric.anomaly_score > threshold:
                direction = "high" if metric.value > metric.benchmark_avg else "low"
                anomalies.append({
                    "metric": metric.name,
                    "value": metric.value,
                    "score": metric.anomaly_score,
                    "direction": direction,
                    "description": (
                        f"{metric.name} is abnormally {direction} "
                        f"({metric.anomaly_score:.1f} standard deviations from mean)"
                    )
                })
                
        return anomalies