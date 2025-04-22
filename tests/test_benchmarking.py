"""
Tests for the benchmarking engine.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.insights.benchmarking import BenchmarkEngine, BenchmarkMetric, BenchmarkResult

class TestBenchmarkEngine(unittest.TestCase):
    """Test the benchmark engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BenchmarkEngine()
        
        # Create sample comparison data
        dates = pd.date_range(start='2024-01-01', periods=30)
        dealers = ['d1', 'd2', 'd3', 'd4', 'd5']
        
        data = []
        for date in dates:
            for dealer in dealers:
                # Generate some realistic-looking metrics
                closing_rate = np.random.normal(30, 5)  # Mean 30%, std 5%
                gross_profit = np.random.normal(2000, 500)  # Mean $2000, std $500
                
                data.append({
                    'dealership_id': dealer,
                    'date': date,
                    'closing_rate': closing_rate,
                    'avg_gross_profit': gross_profit
                })
        
        self.test_data = pd.DataFrame(data)
        
        # Cache test data
        self.engine.metrics_cache['closing_rate'] = self.test_data[
            ['dealership_id', 'date', 'closing_rate']
        ].rename(columns={'closing_rate': 'value'})
        
        self.engine.metrics_cache['avg_gross_profit'] = self.test_data[
            ['dealership_id', 'date', 'avg_gross_profit']
        ].rename(columns={'avg_gross_profit': 'value'})
    
    def test_calculate_benchmarks(self):
        """Test benchmark calculation."""
        results = self.engine.calculate_benchmarks(
            dealership_id='d1',
            metrics=['closing_rate', 'avg_gross_profit'],
            period='30d'
        )
        
        self.assertEqual(results.dealership_id, 'd1')
        self.assertEqual(len(results.metrics), 2)
        self.assertEqual(results.sample_size, 5)
        
        # Verify metric calculations
        for metric in results.metrics:
            self.assertIsInstance(metric, BenchmarkMetric)
            self.assertGreaterEqual(metric.percentile, 0)
            self.assertLessEqual(metric.percentile, 100)
            self.assertIsNotNone(metric.trend)
            self.assertIsNotNone(metric.anomaly_score)
    
    def test_get_improvement_targets(self):
        """Test improvement target calculation."""
        # Create a benchmark result with known values
        metrics = [
            BenchmarkMetric(
                name='closing_rate',
                value=25.0,  # Below 75th percentile
                percentile=50.0,
                benchmark_avg=30.0,
                benchmark_median=29.0,
                benchmark_p75=35.0,
                benchmark_p90=40.0
            ),
            BenchmarkMetric(
                name='avg_gross_profit',
                value=2500.0,  # Above 75th percentile
                percentile=80.0,
                benchmark_avg=2000.0,
                benchmark_median=1900.0,
                benchmark_p75=2400.0,
                benchmark_p90=2800.0
            )
        ]
        
        result = BenchmarkResult(
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            dealership_id='d1',
            comparison_period='30d',
            comparison_group='all',
            sample_size=5
        )
        
        targets = self.engine.get_improvement_targets(result)
        
        # Only closing_rate should have an improvement target
        self.assertEqual(len(targets), 1)
        self.assertIn('closing_rate', targets)
        self.assertEqual(targets['closing_rate'], 10.0)  # 35.0 - 25.0
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Create a benchmark result with known anomalies
        metrics = [
            BenchmarkMetric(
                name='closing_rate',
                value=50.0,  # Significantly high
                percentile=99.0,
                benchmark_avg=30.0,
                benchmark_median=29.0,
                benchmark_p75=35.0,
                benchmark_p90=40.0,
                anomaly_score=3.0  # 3 standard deviations
            ),
            BenchmarkMetric(
                name='avg_gross_profit',
                value=500.0,  # Significantly low
                percentile=1.0,
                benchmark_avg=2000.0,
                benchmark_median=1900.0,
                benchmark_p75=2400.0,
                benchmark_p90=2800.0,
                anomaly_score=2.5  # 2.5 standard deviations
            )
        ]
        
        result = BenchmarkResult(
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            dealership_id='d1',
            comparison_period='30d',
            comparison_group='all',
            sample_size=5
        )
        
        anomalies = self.engine.detect_anomalies(result, threshold=2.0)
        
        self.assertEqual(len(anomalies), 2)
        
        # Verify anomaly details
        closing_rate_anomaly = next(
            a for a in anomalies if a['metric'] == 'closing_rate'
        )
        self.assertEqual(closing_rate_anomaly['direction'], 'high')
        self.assertEqual(closing_rate_anomaly['score'], 3.0)
        
        gross_profit_anomaly = next(
            a for a in anomalies if a['metric'] == 'avg_gross_profit'
        )
        self.assertEqual(gross_profit_anomaly['direction'], 'low')
        self.assertEqual(gross_profit_anomaly['score'], 2.5)
    
    def test_comparison_group_filtering(self):
        """Test filtering by comparison group."""
        results = self.engine.calculate_benchmarks(
            dealership_id='d1',
            metrics=['closing_rate'],
            comparison_group=['d1', 'd2', 'd3'],
            period='30d'
        )
        
        self.assertEqual(results.comparison_group, 'custom')
        self.assertEqual(results.sample_size, 3)
    
    def test_period_filtering(self):
        """Test filtering by time period."""
        # Test different periods
        periods = ['30d', '90d', '180d', '365d']
        
        for period in periods:
            results = self.engine.calculate_benchmarks(
                dealership_id='d1',
                metrics=['closing_rate'],
                period=period
            )
            
            self.assertEqual(results.comparison_period, period)

if __name__ == '__main__':
    unittest.main()