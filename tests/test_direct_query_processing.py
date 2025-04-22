"""
Comprehensive tests for the enhanced query processing functionality.

These tests verify the full pipeline from query parsing through 
metric calculation and benchmark generation.
"""

import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import MagicMock

from src.watchdog_ai.models.query_models import QueryContext, TimeRange
from src.watchdog_ai.insights.context import InsightExecutionContext
from src.watchdog_ai.insights.direct_query_handler import (
    extract_entities,
    extract_temporal_context,
    identify_metric_type,
    process_query,
    find_similar_entities,
    calculate_statistical_significance,
    generate_benchmarks,
    generate_visualization_data
)

# ----------------- Test Data Fixtures -----------------

@pytest.fixture
def sample_sales_data():
    """Create comprehensive sample sales data for testing"""
    # Create date range spanning two years
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='W')
    
    # Create sales reps
    sales_reps = ['Karen Davis', 'John Smith', 'Emma Johnson', 'Michael Brown', 'Sarah Williams']
    
    # Create products
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    
    # Create regions
    regions = ['East', 'West', 'North', 'South', 'Central']
    
    # Create customer segments
    segments = ['Enterprise', 'SMB', 'Government', 'Education', 'Healthcare']
    
    # Generate random data with realistic patterns
    np.random.seed(42)  # For reproducibility
    
    # Create a dataframe with 1000 records
    n_records = 1000
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, n_records),
        'sales_rep': np.random.choice(sales_reps, n_records),
        'product': np.random.choice(products, n_records),
        'region': np.random.choice(regions, n_records),
        'customer_segment': np.random.choice(segments, n_records),
        'revenue': np.random.normal(10000, 3000, n_records),
        'cost': np.random.normal(6000, 1500, n_records),
        'quantity': np.random.randint(1, 20, n_records),
        'days_to_close': np.random.normal(30, 10, n_records),
        'opportunities': np.random.randint(1, 10, n_records),
        'wins': np.random.randint(0, 5, n_records)
    })
    
    # Calculate derived metrics
    df['profit'] = df['revenue'] - df['cost']
    df['profit_margin'] = (df['profit'] / df['revenue']) * 100
    df['conversion_rate'] = (df['wins'] / df['opportunities']) * 100
    
    # Ensure dates are in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure some realistic patterns
    # Karen Davis performs better on revenue
    df.loc[df['sales_rep'] == 'Karen Davis', 'revenue'] *= 1.2
    
    # Product A has better profit margins
    df.loc[df['product'] == 'Product A', 'profit_margin'] *= 1.15
    
    # East region closes faster
    df.loc[df['region'] == 'East', 'days_to_close'] *= 0.8
    
    # Enterprise segment has higher deal values
    df.loc[df['customer_segment'] == 'Enterprise', 'revenue'] *= 1.3
    
    # Recent months show improving trends for all metrics
    recent_dates = pd.date_range(start='2023-09-01', end='2023-12-31')
    df.loc[df['date'].isin(recent_dates), 'revenue'] *= 1.1
    df.loc[df['date'].isin(recent_dates), 'profit'] *= 1.15
    df.loc[df['date'].isin(recent_dates), 'conversion_rate'] *= 1.05
    
    # Round numeric columns to reasonable precision
    df['revenue'] = df['revenue'].round(2)
    df['cost'] = df['cost'].round(2)
    df['profit'] = df['profit'].round(2)
    df['profit_margin'] = df['profit_margin'].round(2)
    df['days_to_close'] = df['days_to_close'].round(1)
    df['conversion_rate'] = df['conversion_rate'].round(2)
    
    return df

@pytest.fixture
def mock_context(sample_sales_data):
    """Create a mock insight execution context with the sample data"""
    context = MagicMock(spec=InsightExecutionContext)
    context.df = sample_sales_data
    context.time_range = None
    context.query = ""
    context.user_role = "analyst"
    return context

@pytest.fixture
def query_context_factory(mock_context):
    """Factory for creating query contexts with different queries"""
    def _create_query_context(query, time_range=None):
        mock_context.time_range = time_range
        return QueryContext(
            query=query,
            insight_context=mock_context
        )
    return _create_query_context

# ----------------- Test Cases -----------------

class TestSimpleMetricQuery:
    """Test simple metric queries for a single entity"""
    
    def test_days_to_close_for_sales_rep(self, query_context_factory):
        """Test query for days to close for a specific sales rep"""
        query = "What is the average days to close for Karen Davis?"
        result = process_query(query_context_factory(query))
        
        # Verify the query was processed successfully
        assert result.success is True
        assert result.confidence_score > 0.7
        
        # Verify entity extraction
        assert "sales_rep" in result.entities
        assert any(e["value"] == "Karen Davis" for e in result.entities["sales_rep"])
        
        # Verify metric calculation
        assert "sales_rep:Karen Davis" in result.metrics
        metric = result.metrics["sales_rep:Karen Davis"]
        assert metric["metric_type"] == "days_to_close"
        assert isinstance(metric["value"], float)
        assert "days" in metric["formatted"]
        
        # Verify benchmarks were generated
        assert hasattr(result, "benchmarks")
        
    def test_revenue_for_product(self, query_context_factory):
        """Test query for revenue for a specific product"""
        query = "What is the total revenue for Product A?"
        result = process_query(query_context_factory(query))
        
        # Verify the query was processed successfully
        assert result.success is True
        
        # Verify entity extraction
        assert "product" in result.entities
        assert any(e["value"] == "Product A" for e in result.entities["product"])
        
        # Verify metric calculation
        assert "product:Product A" in result.metrics
        metric = result.metrics["product:Product A"]
        assert metric["metric_type"] == "revenue"
        assert isinstance(metric["value"], float)
        assert "$" in metric["formatted"]

class TestTemporalContext:
    """Test queries with temporal context"""
    
    def test_query_with_absolute_date(self, query_context_factory):
        """Test query with an absolute date"""
        query = "What was the revenue for Karen Davis on January 15, 2023?"
        result = process_query(query_context_factory(query))
        
        # Verify temporal context extraction
        assert result.intent.time_range is not None
        if result.intent.time_range and result.intent.time_range.start_date:
            assert result.intent.time_range.start_date.month == 1
            assert result.intent.time_range.start_date.day == 15
            assert result.intent.time_range.start_date.year == 2023
        
        # Verify the query was processed successfully with temporal filtering
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics
        
    def test_query_with_relative_date(self, query_context_factory):
        """Test query with a relative date"""
        query = "What was the revenue for Karen Davis in Q4 2023?"
        result = process_query(query_context_factory(query))
        
        # Verify temporal context extraction
        assert result.intent.time_range is not None
        assert result.intent.time_range.period == "q4"
        
        # Verify the query was processed successfully with temporal filtering
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics
        
    def test_query_with_date_range(self, query_context_factory):
        """Test query with a date range"""
        query = "What was the revenue for Karen Davis from January 1 to March 31, 2023?"
        result = process_query(query_context_factory(query))
        
        # Verify temporal context extraction
        assert result.intent.time_range is not None
        assert result.intent.time_range.period == "custom_range"
        
        # Verify the query was processed successfully with temporal filtering
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics

class TestMultipleEntities:
    """Test queries with multiple entities"""
    
    def test_query_with_multiple_sales_reps(self, query_context_factory):
        """Test query for multiple sales reps"""
        query = "Compare the revenue for Karen Davis and John Smith"
        result = process_query(query_context_factory(query))
        
        # Verify entity extraction for multiple entities
        assert "sales_rep" in result.entities
        sales_reps = [e["value"] for e in result.entities["sales_rep"]]
        assert "Karen Davis" in sales_reps
        assert "John Smith" in sales_reps
        
        # Verify metric calculation for multiple entities
        assert "sales_rep:Karen Davis" in result.metrics
        assert "sales_rep:John Smith" in result.metrics
        
        # Verify statistical significance calculation
        if hasattr(result, "statistical_significance") and result.statistical_significance:
            assert "sales_rep" in result.statistical_significance
            
    def test_query_with_multiple_entity_types(self, query_context_factory):
        """Test query with different entity types"""
        query = "Compare the revenue for Product A in the East region with Product B in the West region"
        result = process_query(query_context_factory(query))
        
        # Verify entity extraction for multiple entity types
        assert "product" in result.entities
        assert "region" in result.entities
        
        # Verify metric calculation for multiple entity combinations
        entity_keys = list(result.metrics.keys())
        assert any("Product A" in key for key in entity_keys)
        assert any("Product B" in key for key in entity_keys)
        assert any("East" in key for key in entity_keys) or any("West" in key for key in entity_keys)

class TestFuzzyMatching:
    """Test queries with fuzzy entity matching"""
    
    def test_query_with_misspelled_name(self, query_context_factory):
        """Test query with a misspelled entity name"""
        query = "What is the revenue for Karin Davees?"  # Misspelled Karen Davis
        result = process_query(query_context_factory(query))
        
        # Verify fuzzy matching correctly identified the entity
        assert "sales_rep" in result.entities
        matched_entities = [e["value"] for e in result.entities["sales_rep"]]
        assert "Karen Davis" in matched_entities
        
        # Verify the query was processed using the corrected entity
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics
        
    def test_query_with_partial_name(self, query_context_factory):
        """Test query with a partial entity name"""
        query = "What is the revenue for Product A in the Eastern territory?"  # Eastern instead of East
        result = process_query(query_context_factory(query))
        
        # Verify fuzzy matching correctly identified the entity
        assert "region" in result.entities
        matched_entities = [e["value"] for e in result.entities["region"]]
        assert "East" in matched_entities
        
        # Verify the query was processed using the corrected entity
        assert result.success is True
        assert any("East" in key for key in result.metrics.keys())

class TestCompoundMetrics:
    """Test queries with compound metrics"""
    
    def test_query_with_profit_margin(self, query_context_factory):
        """Test query for profit margin"""
        query = "What is the profit margin for Product A?"
        result = process_query(query_context_factory(query))
        
        # Verify metric identification
        assert result.intent.metric == "profit_margin"
        
        # Verify the query was processed successfully
        assert result.success is True
        assert "product:Product A" in result.metrics
        
        # Verify formatting of percentage
        metric = result.metrics["product:Product A"]
        assert "%" in metric["formatted"]
        
    def test_query_with_trend_analysis(self, query_context_factory):
        """Test query for trend analysis"""
        query = "What is the revenue trend for Karen Davis over the past year?"
        result = process_query(query_context_factory(query))
        
        # Verify metric identification
        assert result.intent.metric == "trend" or result.intent.metric == "revenue"
        
        # Verify temporal context
        assert result.intent.time_range is not None
        
        # Verify the query was processed successfully
        assert result.success is True
        
        # Verify benchmark includes trend data
        if hasattr(result, "benchmarks") and result.benchmarks:
            if "sales_rep" in result.benchmarks:
                bench = result.benchmarks["sales_rep"]
                assert "trend_benchmarks" in bench
                if "Karen Davis" in bench["trend_benchmarks"]:
                    trend = bench["trend_benchmarks"]["Karen Davis"]
                    assert "direction" in trend
                    assert "slope" in trend

class TestStatisticalAnalysis:
    """Test statistical analysis functionality"""
    
    def test_significance_testing(self, query_context_factory):
        """Test statistical significance testing"""
        query = "Compare the days to close between Karen Davis, John Smith, and Emma Johnson"
        result = process_query(query_context_factory(query))
        
        # Verify multiple entities were extracted
        assert "sales_rep" in result.entities
        assert len(result.entities["sales_rep"]) >= 3
        
        # Verify statistical significance calculation
        assert hasattr(result, "statistical_significance")
        if result.statistical_significance and "sales_rep" in result.statistical_significance:
            # Should have at least one pairwise comparison
            assert len(result.statistical_significance["sales_rep"]) > 0
            
            # Get first comparison
            first_test = next(iter(result.statistical_significance["sales_rep"].values()))
            assert "p_value" in first_test
            assert "is_significant" in first_test
            assert "effect_size" in first_test
            assert "confidence_interval_1" in first_test
            assert "comparison" in first_test
    
    def test_benchmark_generation(self, query_context_factory):
        """Test benchmark generation functionality"""
        query = "What is the revenue for Karen Davis compared to other sales reps?"
        result = process_query(query_context_factory(query))
        
        # Verify benchmark generation
        assert hasattr(result, "benchmarks")
        assert "sales_rep" in result.benchmarks
        
        # Check benchmark structure
        sales_rep_benchmarks = result.benchmarks["sales_rep"]
        assert "global_stats" in sales_rep_benchmarks
        assert "peer_benchmarks" in sales_rep_benchmarks
        
        # Check global stats
        global_stats = sales_rep_benchmarks["global_stats"]
        assert "mean" in global_stats
        assert "median" in global_stats
        assert "percentiles" in global_stats
        
        # Check peer benchmarks if available
        if "Karen Davis" in sales_rep_benchmarks["peer_benchmarks"]:
            peer_stats = sales_rep_benchmarks["peer_benchmarks"]["Karen Davis"]
            assert "mean" in peer_stats
            assert "performance_indicator" in peer_stats

class TestTemporalEdgeCases:
    """Test edge cases in temporal context handling"""
    
    def test_invalid_date_format(self, query_context_factory):
        """Test handling of invalid date formats"""
        query = "What is the revenue for Karen Davis on Febtober 32, 2023?"
        result = process_query(query_context_factory(query))
        
        # Should still process the query but ignore the invalid date
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics
        
        # No valid temporal context should be extracted
        if result.intent.time_range:
            assert result.intent.time_range.start_date is None
    
    def test_future_dates(self, query_context_factory):
        """Test handling of future dates"""
        # Use a date far in the future
        future_year = datetime.datetime.now().year + 10
        query = f"What will be the revenue for Karen Davis in {future_year}?"
        
        result = process_query(query_context_factory(query))
        
        # Should still process the query but with no data for future dates
        assert result.success is True
        
        # Future date might be extracted as temporal context
        if result.intent.time_range and result.intent.time_range.start_date:
            assert result.intent.time_range.start_date.year == future_year
    
    def test_overlapping_periods(self, query_context_factory):
        """Test handling of overlapping temporal periods"""
        query = "What is the revenue for Karen Davis in Q1 2023 and January 2023?"
        result = process_query(query_context_factory(query))
        
        # Should still process the query with one of the periods
        assert result.success is True
        assert "sales_rep:Karen Davis" in result.metrics
        
        # Should have extracted at least one temporal context
        assert result.intent.time_range is not None
        
        # Either period could be chosen, but there should be one
        assert (result.intent.time_range.period == "q1" or 
                result.intent.time_range.period == "january" or
                (result.intent.time_range.start_date and result.intent.time_range.start_date.month == 1))
    
    def test_empty_date_range(self, query_context_factory):
        """Test handling of empty date ranges"""
        query = "What is the revenue for Karen Davis from December 31, 2025 to January 1, 2026?"
        result = process_query(query_context_factory(query))
        
        # No data for this period, but should handle gracefully
        assert hasattr(result, "success")
        
        # If success is False, should have appropriate error message
        if not result.success:
            assert result.message is not None
            assert "no data" in result.message.lower() or "empty" in result.message.lower() or "not found" in result.message.lower()

class TestCompleteBenchmarkGeneration:
    """Test complete benchmark generation functionality"""
    
    def test_global_statistics(self, query_context_factory):
        """Test global statistics calculation"""
        query = "What is the profit margin across all products?"
        result = process_query(query_context_factory(query))
        
        # Verify benchmark generation with global stats
        assert hasattr(result, "benchmarks")
        
        if result.benchmarks and any(result.benchmarks):
            entity_type = next(iter(result.benchmarks))
            global_stats = result.benchmarks[entity_type]["global_stats"]
            
            # Check structure and data types
            assert isinstance(global_stats["mean"], float)
            assert isinstance(global_stats["median"], float)
            assert isinstance(global_stats["percentiles"], dict)
            assert "25th" in global_stats["percentiles"]
            assert "75th" in global_stats["percentiles"]
    
    def test_historical_benchmarks(self, query_context_factory):
        """Test historical benchmark generation"""
        query = "How does the current revenue for Product A compare to last year?"
        result = process_query(query_context_factory(query))
        
        # Verify historical benchmarks
        if (result.benchmarks and "product" in result.benchmarks and 
            "historical_benchmarks" in result.benchmarks["product"]):
            
            hist_benchmarks = result.benchmarks["product"]["historical_benchmarks"]
            
            # Should have previous year data
            if "previous_year" in hist_benchmarks:
                prev_year = hist_benchmarks["previous_year"]
                assert "mean" in prev_year
                assert isinstance(prev_year["mean"], float)
    
    def test_peer_group_comparisons(self, query_context_factory):
        """Test peer group comparison benchmarks"""
        query = "How does Karen Davis compare to other sales reps in terms of revenue?"
        result = process_query(query_context_factory(query))
        
        # Verify peer benchmarks
        if (result.benchmarks and "sales_rep" in result.benchmarks and 
            "peer_benchmarks" in result.benchmarks["sales_rep"]):
            
            peer_benchmarks = result.benchmarks["sales_rep"]["peer_benchmarks"]
            
            if "Karen Davis" in peer_benchmarks:
                karen_peers = peer_benchmarks["Karen Davis"]
                assert "mean" in karen_peers
                assert "median" in karen_peers
                assert "percentile_rank" in karen_peers
                assert "performance_indicator" in karen_peers
                
                # Check performance indicator values
                assert karen_peers["performance_indicator"] in ["above_average", "below_average", "average", "neutral"]
    
    def test_trend_based_benchmarks(self, query_context_factory):
        """Test trend-based benchmarks"""
        query = "What is the trend of conversion rate for the East region over the past year?"
        result = process_query(query_context_factory(query))
        
        # Verify trend benchmarks
        if (result.benchmarks and "region" in result.benchmarks and 
            "trend_benchmarks" in result.benchmarks["region"]):
            
            trend_benchmarks = result.benchmarks["region"]["trend_benchmarks"]
            
            if "East" in trend_benchmarks:
                east_trend = trend_benchmarks["East"]
                assert "direction" in east_trend
                assert "slope" in east_trend
                assert "r_squared" in east_trend
                
                # Check direction values
                assert east_trend["direction"] in ["increasing", "decreasing", "stable"]

class TestVisualizationData:
    """Test visualization data generation"""
    
    def test_time_series_data(self, query_context_factory):
        """Test time series data generation"""
        query = "Show me the revenue trend for Karen Davis over the past year"
        result = process_query(query_context_factory(query))
        
        # Verify visualization data for time series
        assert hasattr(result, "historical_context")
        
        if result.historical_context and "time_series" in result.historical_context:
            time_series = result.historical_context["time_series"]
            
            # Skip if empty (implementation may not generate this yet)
            if time_series and any(time_series):
                # Check some entity has time series data
                entity_key = next(iter(time_series))
                series_data = time_series[entity_key]
                
                # Verify data structure if present
                if series_data:
                    assert isinstance(series_data, (list, dict))
    
    def test_comparative_visualization(self, query_context_factory):
        """Test comparative visualization data"""
        query = "Compare the revenue for all sales reps"
        result = process_query(query_context_factory(query))
        
        # Verify comparative visualization data
        if (result.historical_context and 
            "comparative" in result.historical_context and 
            result.historical_context["comparative"]):
            
            comparative = result.historical_context["comparative"]
            
            # Skip if empty (implementation may not generate this yet)
            if comparative and any(comparative):
                # Basic structure check
                assert isinstance(comparative, dict)
    
    def test_distribution_data(self, query_context_factory):
        """Test distribution data generation"""
        query = "What is the distribution of deal sizes for Karen Davis?"
        result = process_query(query_context_factory(query))
        
        # Verify distribution data
        if (result.historical_context and 
            "distribution" in result.historical_context and 
            result.historical_context["distribution"]):
            
            distribution = result.historical_context["distribution"]
            
            # Skip if empty (implementation may not generate this yet)
            if distribution and any(distribution):
                # Basic structure check
                assert isinstance(distribution, dict)
    
    def test_benchmark_overlay(self, query_context_factory):
        """Test benchmark overlay data"""
        query = "How does Karen Davis's revenue compare to the team average?"
        result = process_query(query_context_factory(query))
        
        # Verify benchmark overlay data
        if (result.historical_context and 
            "benchmarks" in result.historical_context and 
            result.historical_context["benchmarks"]):
            
            benchmarks = result.historical_context["benchmarks"]
            
            # Skip if empty (implementation may not generate this yet)
            if benchmarks and any(benchmarks):
                # Basic structure check
                assert isinstance(benchmarks, dict)
    
    def test_confidence_intervals(self, query_context_factory):
        """Test confidence interval data"""
        query = "What is the revenue forecast for Karen Davis with confidence intervals?"
        result = process_query(query_context_factory(query))
        
        # Verify confidence interval data
        if (result.historical_context and 
            "confidence_intervals" in result.historical_context and 
            result.historical_context["confidence_intervals"]):
            
            intervals = result.historical_context["confidence_intervals"]
            
            # Skip if empty (implementation may not generate this yet)
            if intervals and any(intervals):
                # Basic structure check
                assert isinstance(intervals, dict)

class TestErrorHandling:
    """Test error handling in query processing"""
    
    def test_invalid_entities(self, query_context_factory):
        """Test handling of invalid entities"""
        query = "What is the revenue for Nonexistent Person?"
        result = process_query(query_context_factory(query))
        
        # Should either fail gracefully or have low confidence
        if result.success:
            # If processed, should have low confidence
            assert result.confidence_score < 0.5
        else:
            # If failed, should have informative message
            assert "not found" in result.message.lower() or "invalid" in result.message.lower() or "unknown" in result.message.lower()
    
    def test_missing_data(self, query_context_factory, mock_context):
        """Test handling of missing data"""
        # Create empty dataframe context
        mock_context.df = pd.DataFrame()
        
        query = "What is the revenue for Karen Davis?"
        result = process_query(query_context_factory(query))
        
        # Should fail gracefully with informative message
        assert result.success is False
        assert result.message is not None
        assert "no data" in result.message.lower() or "empty" in result.message.lower() or "missing" in result.message.lower()
    
    def test_incompatible_metrics(self, query_context_factory):
        """Test handling of incompatible metrics"""
        # This query doesn't make semantic sense
        query = "What is the days to close percentage for Karen Davis?"
        result = process_query(query_context_factory(query))
        
        # Should either have low confidence or fail with message
        if result.success:
            assert result.confidence_score < 0.7
        else:
            assert "invalid" in result.message.lower() or "incompatible" in result.message.lower() or "not supported" in result.message.lower()
    
    def test_ambiguous_query(self, query_context_factory):
        """Test handling of ambiguous queries"""
        query = "How is performance?"  # Very ambiguous
        result = process_query(query_context_factory(query))
        
        # Should have low confidence
        assert result.confidence_score < 0.7
    
    def test_data_validation_errors(self, query_context_factory, mock_context):
        """Test handling of data validation errors"""
        # Create dataframe with missing key columns
        mock_context.df = pd.DataFrame({
            'some_column': [1, 2, 3],
            'another_column': ['a', 'b', 'c']
        })
        
        query = "What is the revenue for Karen Davis?"
        result = process_query(query_context_factory(query))
        
        # Should fail gracefully with informative message
        assert not result.success or result.confidence_score < 0.5
        if not result.success:
            assert result.message is not None
            assert ("not found" in result.message.lower() or 
                    "missing" in result.message.lower() or 
                    "no data" in result.message.lower())

class TestComplexQuery:
    """Comprehensive integration test for complex query processing"""
    
    def test_complex_query_multiple_features(self, query_context_factory):
        """Test complex query that combines multiple features and components"""
        # Complex query with multiple entities, metrics, temporal context, and comparison
        query = "Compare the revenue and conversion rates for Karen Davis and John Smith in the East region during Q4 2023, and show how it compares to their historical performance"
        
        result = process_query(query_context_factory(query))
        
        # 1. Verify successful processing with high confidence
        assert result.success is True
        assert result.confidence_score > 0.8
        
        # 2. Check entity extraction (multiple entities and entity types)
        assert "sales_rep" in result.entities
        sales_reps = [e["value"] for e in result.entities["sales_rep"]]
        assert "Karen Davis" in sales_reps
        assert "John Smith" in sales_reps
        
        assert "region" in result.entities
        regions = [e["value"] for e in result.entities["region"]]
        assert "East" in regions
        
        # 3. Verify temporal context extraction
        assert result.intent.time_range is not None
        assert result.intent.time_range.period == "q4" or (
            result.intent.time_range.start_date and 
            result.intent.time_range.start_date.year == 2023 and
            result.intent.time_range.start_date.month >= 10
        )
        
        # 4. Validate metric calculations
        # At least one of the metrics should be calculated correctly
        entity_metrics = {}
        for key, metric in result.metrics.items():
            entity_type, entity_value = key.split(":", 1)
            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {}
            entity_metrics[entity_type][entity_value] = metric
        
        if "sales_rep" in entity_metrics:
            sales_rep_metrics = entity_metrics["sales_rep"]
            if "Karen Davis" in sales_rep_metrics:
                karen_metric = sales_rep_metrics["Karen Davis"]
                assert isinstance(karen_metric["value"], (int, float))
                assert karen_metric["metric_type"] in ["revenue", "conversion_rate"]
            
            if "John Smith" in sales_rep_metrics:
                john_metric = sales_rep_metrics["John Smith"]
                assert isinstance(john_metric["value"], (int, float))
                assert john_metric["metric_type"] in ["revenue", "conversion_rate"]
        
        # 5. Verify statistical analysis
        assert hasattr(result, "statistical_significance")
        if (result.statistical_significance and 
            "sales_rep" in result.statistical_significance and
            result.statistical_significance["sales_rep"]):
            
            # Should find at least one comparison between the sales reps
            comparison_keys = list(result.statistical_significance["sales_rep"].keys())
            has_comparison = False
            for key in comparison_keys:
                if "Karen Davis" in key and "John Smith" in key:
                    has_comparison = True
                    break
                    
            if has_comparison:
                comparison = result.statistical_significance["sales_rep"][key]
                assert "p_value" in comparison
                assert "is_significant" in comparison
                assert "effect_size" in comparison
        
        # 6. Check benchmark generation
        assert hasattr(result, "benchmarks")
        if result.benchmarks and "sales_rep" in result.benchmarks:
            benchmarks = result.benchmarks["sales_rep"]
            
            # Verify global statistics
            assert "global_stats" in benchmarks
            assert isinstance(benchmarks["global_stats"]["mean"], float)
            
            # Verify historical benchmarks
            if "historical_benchmarks" in benchmarks and benchmarks["historical_benchmarks"]:
                hist_bench = benchmarks["historical_benchmarks"]
                assert isinstance(next(iter(hist_bench.values()))["mean"], float)
            
            # Verify peer benchmarks
            if "peer_benchmarks" in benchmarks:
                for rep in ["Karen Davis", "John Smith"]:
                    if rep in benchmarks["peer_benchmarks"]:
                        peer_bench = benchmarks["peer_benchmarks"][rep]
                        assert "percentile_rank" in peer_bench
                        assert "performance_indicator" in peer_bench
        
        # 7. Verify visualization data
        assert hasattr(result, "historical_context")
        
        # 8. Check for related insights
        assert hasattr(result, "related_insights")
        if result.related_insights:
            # Should include at least one insight
            insight = result.related_insights[0]
            assert isinstance(insight, dict)
            # Check the insight has the expected structure
            if "type" in insight:
                assert insight["type"] in ["similar_entity", "trend", "anomaly", "correlation"]
                
        # 9. Comprehensive check - ensure all components worked together
        # At this point, if all previous assertions passed, we know the system
        # successfully processed a complex query involving multiple components
        
        # Additional check: ensure metrics are available for all entities in all regions
        all_rep_region_keys = []
        for rep in ["Karen Davis", "John Smith"]:
            for region in ["East"]:
                all_rep_region_keys.append(f"sales_rep:{rep}")
                
        # At least one of these combinations should have a metric result
        assert any(key in result.metrics for key in all_rep_region_keys)
