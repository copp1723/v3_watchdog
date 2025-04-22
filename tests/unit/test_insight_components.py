"""
Unit tests for the insight pipeline components.

Tests for the fallback renderer, precision scoring, chart renderer, and data validation components.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import altair as alt

from src.watchdog_ai.insights.fallback_renderer import FallbackRenderer, FallbackContext, FallbackReason
from src.watchdog_ai.insights.precision_scoring import PrecisionScoringEngine
from src.watchdog_ai.insights.chart_renderer import ChartRenderer, ChartError
from src.watchdog_ai.insights.data_validation import check_data_quality
from src.watchdog_ai.insights.contracts import InsightContract, InsightContractEnforcer, create_default_contract
from src.watchdog_ai.utils.dataframe_utils import DataFrameUtils

# Sample data for testing
SAMPLE_DF = pd.DataFrame({
    'lead_source': ['NeoIdentity', 'AutoTrader', 'GM Financial', 'Website'],
    'listing_price': [55000, 53500, 54000, 49000],
    'sold_price': [53500, 52000, 52300, 47800],
    'profit': [3200, 3100, 3000, 2200],
    'expense': [1400, 900, 1100, 850],
    'sales_rep_name': ['John Doe', 'Mike Wilson', 'Jane Smith', 'John Doe'],
    'vehicle_year': [2022, 2023, 2022, 2018],
    'vehicle_make': ['GMC', 'GMC', 'GMC', 'Honda'],
    'vehicle_model': ['Sierra', 'Yukon', 'Yukon', 'Accord'],
    'days_to_close': [18, 14, 15, 30]
})

# Sample breakdown data for chart testing
SAMPLE_BREAKDOWN = [
    {"label": "GMC", "value": 52600.0},
    {"label": "Honda", "value": 47800.0},
    {"label": "Toyota", "value": 45200.0},
    {"label": "Ford", "value": 42100.0}
]

class TestFallbackRenderer(unittest.TestCase):
    """Test the fallback renderer component."""
    
    def setUp(self):
        """Set up test environment."""
        self.renderer = FallbackRenderer()
    
    def test_low_precision_fallback(self):
        """Test fallback for low precision queries."""
        context = FallbackContext(
            reason=FallbackReason.LOW_PRECISION,
            details={"precision_score": 0.2},
            original_query="Show me sales"
        )
        
        result = self.renderer.render_fallback(context)
        
        # Check result structure
        self.assertEqual(result["type"], "fallback")
        self.assertIn("message", result)
        self.assertIn("suggestion", result)
        self.assertIn("examples", result)
        self.assertGreater(len(result["examples"]), 0)
        self.assertEqual(result["reason"], "low_precision")
        self.assertEqual(result["original_query"], "Show me sales")
        
        # Check precision score is included
        self.assertEqual(result["precision_score"], 0.2)
    
    def test_missing_columns_fallback(self):
        """Test fallback for missing columns."""
        context = FallbackContext(
            reason=FallbackReason.MISSING_COLUMNS,
            details={
                "available_columns": ["profit", "sales_rep_name", "vehicle_make"]
            },
            original_query="Show me customer lifetime value"
        )
        
        result = self.renderer.render_fallback(context)
        
        # Check result structure
        self.assertEqual(result["reason"], "missing_columns")
        self.assertIn("examples", result)
        
        # Check available columns are included in examples
        self.assertEqual(len(result["examples"]), 3)  # 3 columns
        self.assertTrue(any("profit" in example for example in result["examples"]))
    
    def test_data_quality_fallback(self):
        """Test fallback for data quality issues."""
        context = FallbackContext(
            reason=FallbackReason.DATA_QUALITY,
            details={
                "issues": ["High percentage of missing values", "Invalid date format"],
                "data_quality_metrics": {"valid_percentage": 45.0}
            },
            original_query="Analyze sales trends"
        )
        
        result = self.renderer.render_fallback(context)
        
        # Check result structure
        self.assertEqual(result["reason"], "data_quality")
        self.assertIn("data_quality_metrics", result)
        self.assertEqual(result["data_quality_metrics"]["valid_percentage"], 45.0)
    
    def test_did_you_mean(self):
        """Test 'did you mean' suggestions."""
        suggestions = [
            "Show me average profit by sales rep",
            "Show me total profit by vehicle make", 
            "Show me profit trends over time"
        ]
        
        result = self.renderer.render_did_you_mean("Show me profit", suggestions)
        
        # Check result structure
        self.assertEqual(result["type"], "did_you_mean")
        self.assertEqual(result["original_query"], "Show me profit")
        self.assertEqual(result["suggestions"], suggestions)
    
    def test_error_rendering(self):
        """Test error message rendering."""
        error = ValueError("Invalid calculation")
        
        result = self.renderer.render_error(error, "Calculate profit margin")
        
        # Check result structure
        self.assertEqual(result["type"], "error")
        self.assertIn("message", result)
        self.assertIn("suggestion", result)
        self.assertEqual(result["original_query"], "Calculate profit margin")
        self.assertEqual(result["error"], "Invalid calculation")

class TestPrecisionScoring(unittest.TestCase):
    """Test the precision scoring engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.scorer = PrecisionScoringEngine()
    
    def test_business_pattern_matching(self):
        """Test matching business patterns."""
        # Test top performing pattern
        result = self.scorer.predict_precision(
            "Who is the top performing sales rep?",
            {"columns": SAMPLE_DF.columns.tolist()}
        )
        
        self.assertGreaterEqual(result["score"], 0.8)
        self.assertEqual(result["confidence_level"], "high")
        self.assertTrue(any("business pattern" in reason for reason in result["reasons"]))
        
        # Test highest profit pattern
        result = self.scorer.predict_precision(
            "What is the highest profit vehicle make?",
            {"columns": SAMPLE_DF.columns.tolist()}
        )
        
        self.assertGreaterEqual(result["score"], 0.8)
        self.assertEqual(result["confidence_level"], "high")
    
    def test_component_matching(self):
        """Test matching individual components."""
        # Query with metric, dimension, and time
        result = self.scorer.predict_precision(
            "What was the total profit by vehicle make last month?",
            {"columns": SAMPLE_DF.columns.tolist()}
        )
        
        # Should have high score from components
        self.assertGreaterEqual(result["score"], 0.7)
        self.assertEqual(result["confidence_level"], "high")
        
        # Check specific reasons
        reasons_text = " ".join(result["reasons"])
        self.assertIn("metrics", reasons_text)
        self.assertIn("dimensions", reasons_text)
        self.assertIn("time", reasons_text)
    
    def test_column_matching(self):
        """Test direct column name matching."""
        # Query with exact column names
        result = self.scorer.predict_precision(
            "Compare profit and expense by sales_rep_name",
            {"columns": SAMPLE_DF.columns.tolist()}
        )
        
        # Should have column matches
        self.assertTrue(any("columns" in reason for reason in result["reasons"]))
        self.assertGreaterEqual(result["score"], 0.5)
    
    def test_data_quality_impact(self):
        """Test impact of data quality on precision."""
        # Base query
        base_result = self.scorer.predict_precision(
            "What is the average profit?",
            {"columns": SAMPLE_DF.columns.tolist(), "nan_percentage": 0}
        )
        
        # Same query with poor data quality
        poor_result = self.scorer.predict_precision(
            "What is the average profit?",
            {"columns": SAMPLE_DF.columns.tolist(), "nan_percentage": 50}
        )
        
        # Poor quality should reduce score
        self.assertLess(poor_result["score"], base_result["score"])
        self.assertTrue(any("quality" in reason for reason in poor_result["reasons"]))

class TestChartRenderer(unittest.TestCase):
    """Test the chart renderer component."""
    
    def test_create_bar_chart(self):
        """Test creating a bar chart."""
        chart, metadata = ChartRenderer.create_chart(
            breakdown=SAMPLE_BREAKDOWN,
            df=SAMPLE_DF,
            chart_type="bar",
            title="Average Price by Make"
        )
        
        # Check successful creation
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["chart_type"], "bar")
        self.assertIsInstance(chart, alt.Chart)
        
        # Check chart configuration
        chart_dict = chart.to_dict()
        self.assertEqual(chart_dict["mark"], "bar")
        self.assertEqual(chart_dict["title"], "Average Price by Make")
    
    def test_create_pie_chart(self):
        """Test creating a pie chart."""
        chart, metadata = ChartRenderer.create_chart(
            breakdown=SAMPLE_BREAKDOWN,
            df=SAMPLE_DF,
            chart_type="pie",
            title="Price Distribution by Make"
        )
        
        # Check successful creation
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["chart_type"], "pie")
        self.assertIsInstance(chart, alt.Chart)
        
        # Check chart configuration
        chart_dict = chart.to_dict()
        self.assertEqual(chart_dict["mark"], "arc")
        self.assertEqual(chart_dict["title"], "Price Distribution by Make")
    
    def test_invalid_chart_type(self):
        """Test handling invalid chart type."""
        chart, metadata = ChartRenderer.create_chart(
            breakdown=SAMPLE_BREAKDOWN,
            df=SAMPLE_DF,
            chart_type="invalid_type", 
            title="Test Chart"
        )
        
        # Should default to bar
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["chart_type"], "bar")
    
    def test_invalid_breakdown_data(self):
        """Test handling invalid breakdown data."""
        # Empty breakdown
        chart, metadata = ChartRenderer.create_chart(
            breakdown=[],
            df=SAMPLE_DF,
            chart_type="bar"
        )
        
        # Should fail
        self.assertFalse(metadata["success"])
        self.assertIn("error", metadata)
        self.assertIsNone(chart)
        
        # Invalid breakdown structure
        chart, metadata = ChartRenderer.create_chart(
            breakdown=[{"wrong_key": "value"}],
            df=SAMPLE_DF,
            chart_type="bar"
        )
        
        # Should fail
        self.assertFalse(metadata["success"])
        self.assertIn("error", metadata)
        self.assertIsNone(chart)
    
    def test_validate_chart_data(self):
        """Test chart data validation."""
        # Valid data
        is_valid, issues = ChartRenderer.validate_chart_data(SAMPLE_BREAKDOWN)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid data - missing value
        invalid_data = [{"label": "Test"}]
        is_valid, issues = ChartRenderer.validate_chart_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        
        # Invalid data - empty
        is_valid, issues = ChartRenderer.validate_chart_data([])
        self.assertFalse(is_valid)
        self.assertIn("Empty data list", issues)

class TestDataValidation(unittest.TestCase):
    """Test the data validation component."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        is_valid, validation_info = check_data_quality(SAMPLE_DF)
        
        self.assertTrue(is_valid)
        self.assertEqual(validation_info["is_valid"], True)
        self.assertEqual(len(validation_info["issues"]), 0)
        self.assertIn("metrics", validation_info)
        self.assertIn("valid_data_percentage", validation_info)
        self.assertEqual(validation_info["valid_data_percentage"], 100.0)
    
    def test_empty_data(self):
        """Test validation with empty data."""
        is_valid, validation_info = check_data_quality(pd.DataFrame())
        
        self.assertFalse(is_valid)
        self.assertEqual(validation_info["is_valid"], False)
        self.assertIn("No data provided", validation_info["issues"])
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        is_valid, validation_info = check_data_quality(
            SAMPLE_DF, 
            required_columns=["profit", "missing_column"]
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Missing required columns", validation_info["issues"][0])
    
    def test_data_with_nulls(self):
        """Test validation with null values."""
        # Create DataFrame with nulls
        df_with_nulls = SAMPLE_DF.copy()
        df_with_nulls.loc[0, "profit"] = None
        df_with_nulls.loc[1, "profit"] = None
        df_with_nulls.loc[2, "profit"] = None  # 75% nulls in profit column
        
        is_valid, validation_info = check_data_quality(df_with_nulls, ["profit"])
        
        # Check missing percentages are calculated
        self.assertIn("metrics", validation_info)
        self.assertIn("missing_percentages", validation_info["metrics"])
        self.assertGreaterEqual(validation_info["metrics"]["missing_percentages"]["profit"], 75.0)
    
    def test_all_zeros_column(self):
        """Test validation with all-zeros column."""
        # Create DataFrame with a column of all zeros
        df_zeros = SAMPLE_DF.copy()
        df_zeros["profit"] = 0
        
        is_valid, validation_info = check_data_quality(df_zeros, ["profit"])
        
        # Should flag all-zeros column
        all_zeros_issue = any("contains all zeros" in issue for issue in validation_info["issues"])
        self.assertTrue(all_zeros_issue)
    
    def test_low_variance_column(self):
        """Test validation with low variance column."""
        # Create DataFrame with a column of very similar values
        df_low_var = SAMPLE_DF.copy()
        df_low_var["profit"] = [1.000001, 1.000002, 1.000001, 1.000003]
        
        is_valid, validation_info = check_data_quality(df_low_var, ["profit"])
        
        # Should flag low variance column
        low_var_issue = any("low variance" in issue for issue in validation_info["issues"])
        self.assertTrue(low_var_issue)

class TestContractValidation(unittest.TestCase):
    """Test the insight contract validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.enforcer = InsightContractEnforcer()
        self.contract = create_default_contract("test_insight")
    
    def test_validate_output_success(self):
        """Test successful output validation."""
        # Create valid output
        output = {
            "insight_id": "test_insight_20230101_120000",
            "insight_type": "test_insight",
            "version": "1.0.0",
            "required_columns": ["SaleDate", "TotalGross", "LeadSource"],
            "min_rows": 10,
            "data_types": {
                "SaleDate": "datetime64[ns]",
                "TotalGross": "float64",
                "LeadSource": "object"
            },
            "metrics": {
                "total_records": 100,
                "time_period": "Jan 2023",
                "confidence_score": 0.9,
                "data_quality_score": 0.95
            },
            "summary": "This is a test summary that is long enough to be valid.",
            "key_findings": ["The average profit was $3,000 per sale.", "GMC vehicles had the highest profit margin."],
            "recommendations": ["Consider focusing on GMC inventory", "Review pricing strategy for Honda vehicles"],
            "execution_time_ms": 150.5
        }
        
        result = self.enforcer.validate_output(output, self.contract)
        
        # Check validation success
        self.assertTrue(result["is_valid"])
        self.assertEqual(len(result["errors"]), 0)
    
    def test_validate_output_warnings(self):
        """Test validation with warnings."""
        # Create output with low confidence
        output = {
            "insight_id": "test_insight_20230101_120000",
            "insight_type": "test_insight",
            "version": "1.0.0",
            "required_columns": ["SaleDate", "TotalGross", "LeadSource"],
            "min_rows": 10,
            "data_types": {
                "SaleDate": "datetime64[ns]",
                "TotalGross": "float64",
                "LeadSource": "object"
            },
            "metrics": {
                "total_records": 100,
                "time_period": "Jan 2023",
                "confidence_score": 0.5,  # Low confidence
                "data_quality_score": 0.95
            },
            "summary": "This is a test summary that is long enough to be valid.",
            "key_findings": ["The average profit was $3,000 per sale."],
            "recommendations": ["Consider focusing on GMC inventory"],  # Only one recommendation
            "execution_time_ms": 150.5
        }
        
        result = self.enforcer.validate_output(output, self.contract)
        
        # Should be valid but have warnings
        self.assertTrue(result["is_valid"])
        self.assertGreater(len(result["warnings"]), 0)
        
        # Check specific warnings
        warnings_text = " ".join(result["warnings"])
        self.assertIn("confidence", warnings_text.lower())
        self.assertIn("recommendations", warnings_text.lower())
    
    def test_validate_output_failure(self):
        """Test validation failure."""
        # Create invalid output
        output = {
            "insight_id": "test_insight_20230101_120000",
            "insight_type": "test_insight",
            "version": "1.0.0",
            # Missing required fields
            "summary": "This is a test summary.",
            # Missing many required fields
            "execution_time_ms": 150.5
        }
        
        result = self.enforcer.validate_output(output, self.contract)
        
        # Should fail validation
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["errors"]), 0)

class TestDataFrameUtils(unittest.TestCase):
    """Test the DataFrame utilities."""
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        # Valid DataFrame
        is_valid, issues = DataFrameUtils.validate_dataframe(SAMPLE_DF)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Empty DataFrame
        is_valid, issues = DataFrameUtils.validate_dataframe(pd.DataFrame())
        self.assertFalse(is_valid)
        self.assertIn("DataFrame is empty", issues)
        
        # Missing required columns
        is_valid, issues = DataFrameUtils.validate_dataframe(
            SAMPLE_DF, 
            required_cols=["profit", "non_existent_column"]
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required columns" in issue for issue in issues))
        
        # Mixed data types
        df_mixed = SAMPLE_DF.copy()
        df_mixed["mixed_column"] = ["string", 123, 3.14, True]
        is_valid, issues = DataFrameUtils.validate_dataframe(df_mixed)
        mixed_type_issue = any("mixed data types" in issue for issue in issues)
        self.assertTrue(mixed_type_issue)
    
    def test_get_column_stats(self):
        """Test column statistics calculation."""
        # Numeric column
        stats = DataFrameUtils.get_column_stats(SAMPLE_DF, "profit")
        
        self.assertEqual(stats["column"], "profit")
        self.assertEqual(stats["row_count"], 4)
        self.assertEqual(stats["null_count"], 0)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        
        # String column
        stats = DataFrameUtils.get_column_stats(SAMPLE_DF, "vehicle_make")
        
        self.assertEqual(stats["column"], "vehicle_make")
        self.assertEqual(stats["unique_count"], 2)  # GMC and Honda
        
        # Invalid column
        stats = DataFrameUtils.get_column_stats(SAMPLE_DF, "non_existent")
        self.assertIn("error", stats)
    
    def test_get_breakdown(self):
        """Test getting data breakdowns."""
        # Group by vehicle make, sum profit
        breakdown, metadata = DataFrameUtils.get_breakdown(
            SAMPLE_DF,
            group_by="vehicle_make",
            metric="profit",
            agg_func="sum"
        )
        
        # Check breakdown structure
        self.assertGreater(len(breakdown), 0)
        self.assertIn("category", breakdown[0])
        self.assertIn("value", breakdown[0])
        
        # Check metadata
        self.assertIn("total_groups", metadata)
        self.assertEqual(metadata["total_records"], 4)
        self.assertEqual(metadata["aggregation_function"], "sum")
        
        # Group by multiple columns
        breakdown, metadata = DataFrameUtils.get_breakdown(
            SAMPLE_DF,
            group_by=["vehicle_make", "vehicle_model"],
            metric="profit",
            agg_func="mean"
        )
        
        # Check breakdown structure
        self.assertGreater(len(breakdown), 0)
        self.assertIn("vehicle_make", breakdown[0])
        self.assertIn("vehicle_model", breakdown[0])
        self.assertIn("value", breakdown[0])
        
        # Test normalization
        breakdown, metadata = DataFrameUtils.get_breakdown(
            SAMPLE_DF,
            group_by="vehicle_make",
            metric="profit",
            normalize=True
        )
        
        # Values should be percentages
        total_percent = sum(item["value"] for item in breakdown)
        self.assertAlmostEqual(total_percent, 100.0, places=1)
        
        # Test error handling
        breakdown, metadata = DataFrameUtils.get_breakdown(
            SAMPLE_DF,
            group_by="non_existent",
            metric="profit"
        )
        
        self.assertEqual(len(breakdown), 0)
        self.assertIn("error", metadata)

if __name__ == "__main__":
    unittest.main()