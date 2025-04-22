"""
Tests for sales report renderer components.

This test suite verifies the functionality of the SalesReportRenderer component,
which is responsible for rendering sales insights, charts, metrics, and anomaly detection.
The tests cover both the legacy class-based approach and the newer functional components.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import streamlit as st
from watchdog_ai.ui.components.sales_report_renderer import (
    SalesReportRenderer, 
    render_chart,
    render_insight_block, 
    render_error_state, 
    render_metric_group,
    format_timedelta
)
from watchdog_ai.core.visualization import ChartConfig, ChartType

@pytest.fixture
def renderer():
    """Create a SalesReportRenderer instance for testing."""
    return SalesReportRenderer()

@pytest.fixture
def mock_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'LeadSource': ['NeoIdentity'] * 4 + ['CarGurus'] * 3 + ['Website'] * 5 + 
                     ['Walk-in'] * 3 + ['Phone'] * 3,
        'TotalGross': [2600.0] * 4 + [2200.0] * 3 + [2400.0] * 5 + 
                     [2100.0] * 3 + [2300.0] * 3,
        'SaleDate': [datetime.now() - timedelta(days=x) for x in range(18)]
    })

@pytest.fixture
def sample_anomaly_data():
    """Create sample anomaly data for testing."""
    return {
        "primary_metrics": {
            "trend": "-25%",
            "comparison": "152% above team"
        },
        "performance_breakdown": [
            {
                "category": "Vehicle Type",
                "comparison": "152% above team"
            },
            {
                "category": "Sales Team",
                "comparison": "10% below average"
            }
        ]
    }

@pytest.fixture
def insight_data_complete():
    """Create a complete insight data fixture with all possible fields."""
    return {
        "summary": "Sales Performance Analysis",
        "metrics": {
            "total_records": 150,
            "time_period": "Last 30 days",
            "data_quality_score": 0.95,
            "response_time_metrics": {
                "avg_response_time": timedelta(hours=2),
                "within_1hour": 75.5,
                "within_24hours": 95.0,
                "response_rate": 88.5,
                "response_distribution": pd.DataFrame({
                    "Time Range": ["0-1h", "1-4h", "4-24h", ">24h"],
                    "Count": [45, 25, 15, 5]
                })
            },
            "inventory_age_metrics": {
                "avg_days_on_lot": 45.5,
                "best_performing_age": "30-60 days",
                "total_aged_inventory": 12,
                "avg_profit_by_age": {
                    "0-30 days": 2500,
                    "30-60 days": 3200,
                    "60-90 days": 2800,
                    ">90 days": 1900
                },
                "profit_correlation": -0.35
            }
        },
        "visualization": {
            "type": "time_series",
            "data": {
                "2023-01-01": 150,
                "2023-01-02": 180,
                "2023-01-03": 165
            },
            "title": "Daily Sales Trend",
            "x_axis": "Date",
            "y_axis": "Sales Volume",
            "cumulative": True
        },
        "key_findings": [
            "Sales volume increased by 20% MoM",
            "Response time improved by 15%"
        ],
        "recommendations": [
            "Focus on aged inventory reduction",
            "Optimize response time during peak hours"
        ],
        "confidence": "high"
    }

@pytest.fixture
def visualization_payload():
    """Create a visualization data fixture for testing chart rendering."""
    return {
        "bar": {
            "type": "bar",
            "data": pd.DataFrame({
                "Category": ["A", "B", "C"],
                "Value": [10, 20, 30]
            }),
            "title": "Category Distribution",
            "x_axis": "Category",
            "y_axis": "Value"
        },
        "scatter": {
            "type": "scatter",
            "data": {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10]
            },
            "title": "Correlation Plot",
            "x_axis": "X Values",
            "y_axis": "Y Values",
            "trend_line": True
        },
        "pie": {
            "type": "pie",
            "data": pd.DataFrame({
                "Category": ["A", "B", "C"],
                "Value": [30, 40, 30]
            }),
            "title": "Distribution"
        }
    }

@pytest.fixture
def error_state_payload():
    """Create error state test data."""
    return {
        "simple_error": "Data unavailable",
        "complex_error": {
            "message": "Processing failed",
            "details": "Invalid data format",
            "code": "ERR_001"
        }
    }

@pytest.fixture
def metric_group_payload():
    """Create metric group test data with various edge cases."""
    return {
        "basic": {
            "Total Sales": 1500,
            "Average Price": "$25,000",
            "Conversion Rate": "15%"
        },
        "with_thresholds": {
            "Margin": {"value": 12.5, "delta": -2.1, "threshold": {"below": 10}},
            "Volume": {"value": 250, "delta": 25, "threshold": {"above": 200}},
            "Retention": {"value": 85.2, "delta": 0.5}
        },
        "empty": {},
        "none_values": {
            "Sales": None,
            "Profit": np.nan
        }
    }

@pytest.fixture
def streamlit_mocker(monkeypatch):
    """Create mocks for all Streamlit components used in the renderer."""
    mocks = {
        "markdown": MagicMock(),
        "columns": MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()]),
        "metric": MagicMock(),
        "plotly_chart": MagicMock(),
        "bar_chart": MagicMock(),
        "error": MagicMock(),
        "warning": MagicMock()
    }
    
    for component, mock in mocks.items():
        monkeypatch.setattr(st, component, mock)
    
    return mocks

class TestSalesReportRenderer:
    """Tests for the SalesReportRenderer class."""
    
    def test_detect_anomalies_no_drop(self, renderer):
        """Test anomaly detection with no significant drop."""
        report_data = {
            "primary_metrics": {
                "trend": "-15%"  # Not significant enough
            }
        }
        
        anomalies = renderer._detect_anomalies(report_data)
        
        assert len(anomalies) == 0
        assert "anomalies" not in report_data

    def test_detect_anomalies_margin_deviation(self, renderer):
        """Test anomaly detection for margin deviation."""
        report_data = {
            "performance_breakdown": [
                {
                    "category": "Vehicle Type",
                    "comparison": "152% above team"  # Significant deviation
                }
            ]
        }
        
        anomalies = renderer._detect_anomalies(report_data)
        
        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "margin_deviation"
        assert anomalies[0]["value"] == 152
        
        assert "anomalies" in report_data
        assert len(report_data["anomalies"]) == 1
        assert "152%" in report_data["anomalies"][0]

    def test_nova_act_data_integration(self, renderer, mock_data):
        """Test integration with Nova Act data format."""
        # Verify data structure
        assert len(mock_data) == 18
        assert mock_data['LeadSource'].value_counts()['NeoIdentity'] == 4
        assert mock_data['TotalGross'][mock_data['LeadSource'] == 'NeoIdentity'].mean() == 2600.0
        
        # Test rendering with Nova Act data
        with patch('streamlit.markdown') as mock_markdown:
            renderer.render_insight_block({
                'data': mock_data,
                'source': 'nova_act',
                'metrics': {
                    'total_records': len(mock_data),
                    'time_period': 'Last 18 days'
                }
            })
            
            mock_markdown.assert_any_call("### Lead Source Performance")

    def test_detect_anomalies_edge_cases(self, renderer):
        """Test anomaly detection edge cases."""
        # Test empty data
        assert len(renderer._detect_anomalies({})) == 0
        
        # Test None data
        assert len(renderer._detect_anomalies(None)) == 0
        
        # Test invalid metrics format
        assert len(renderer._detect_anomalies({"primary_metrics": "invalid"})) == 0
        
        # Test missing comparison values
        assert len(renderer._detect_anomalies({
            "primary_metrics": {
                "trend": None,
                "comparison": None
            }
        })) == 0

    def test_anomaly_threshold_configuration(self, renderer):
        """Test anomaly detection with different thresholds."""
        report_data = {
            "primary_metrics": {
                "trend": "-22%",
                "comparison": "148% above team"
            }
        }
        
        # Test with default thresholds (should not detect)
        anomalies = renderer._detect_anomalies(report_data)
        assert len(anomalies) == 0
        
        # Test with custom thresholds (should detect)
        anomalies = renderer._detect_anomalies(
            report_data,
            drop_threshold=20,
            deviation_threshold=145
        )
        assert len(anomalies) == 2

    def test_multiple_anomaly_detection(self, renderer, sample_anomaly_data):
        """Test detection of multiple anomalies in same report."""
        anomalies = renderer._detect_anomalies(sample_anomaly_data)
        
        assert len(anomalies) >= 2
        anomaly_types = [a["type"] for a in anomalies]
        assert "metric_drop" in anomaly_types
        assert "margin_deviation" in anomaly_types
        
        # Verify anomaly rendering content
        mock_nova = Mock()
        with patch('nova_client.push_event', mock_nova):
            renderer._render_anomalies(anomalies)
            
            # Verify proper number of anomaly flags
            assert len(anomalies) == 2
            
            # Verify Nova Act integration
            assert mock_nova.call_count >= 1

    @patch('streamlit.markdown')
    def test_anomaly_rendering(self, mock_markdown, renderer):
        """Test rendering of detected anomalies."""
        anomalies = [
            {
                "type": "metric_drop",
                "value": 25,
                "description": "Significant drop in performance (-25%)"
            },
            {
                "type": "margin_deviation",
                "value": 152,
                "description": "Unusual margin deviation (152% above team)"
            }
        ]
        
        renderer._render_anomalies(anomalies)
        
        assert mock_markdown.call_count >= 2
        mock_markdown.assert_any_call("#### ⚠️ Anomalies Detected")

    def test_anomaly_format_validation(self, renderer):
        """Test validation of anomaly formats."""
        report_data = {
            "primary_metrics": {
                "trend": "-30%",
                "comparison": "200% above team"
            }
        }
        
        anomalies = renderer._detect_anomalies(report_data)
        
        for anomaly in anomalies:
            assert isinstance(anomaly.get("description"), str)
            assert isinstance(anomaly.get("type"), str)
            assert isinstance(anomaly.get("value"), (int, float))
            assert anomaly.get("description", "").strip() != ""

    def test_data_preprocessing(self, renderer, mock_data):
        """Test data preprocessing for anomaly detection."""
        with patch.object(renderer, '_preprocess_data') as mock_preprocess:
            mock_preprocess.return_value = mock_data
            
            renderer.render_insight_block({
                'data': mock_data,
                'source': 'nova_act'
            })
            
            mock_preprocess.assert_called_once()
            
    def test_render_insight_block_full(self, renderer, insight_data_complete, monkeypatch):
        """Test rendering a complete insight block with all components."""
        mock_col = Mock()
        mock_markdown = Mock()
        mock_metric = Mock()
        mock_plotly = Mock()
        
        monkeypatch.setattr(st, "columns", lambda x: [mock_col] * x)
        monkeypatch.setattr(st, "markdown", mock_markdown)
        monkeypatch.setattr(st, "metric", mock_metric)
        monkeypatch.setattr(st, "plotly_chart", mock_plotly)
        
        renderer.render_insight_block(insight_data_complete)
        
        # Verify all components were rendered
        mock_markdown.assert_any_call("### Sales Performance Analysis")
        mock_metric.assert_any_call("Total Records", 150)
        mock_plotly.assert_called()
        
        # Verify findings and recommendations
        mock_markdown.assert_any_call("### Key Findings")
        mock_markdown.assert_any_call("• Sales volume increased by 20% MoM")
        
        # Verify confidence level
        mock_markdown.assert_any_call(
            '<p style=\'color: green\'>Confidence Level: High</p>',
            unsafe_allow_html=True,
            help="Indicates the reliability of the insight based on data quality"
        )
        
    def test_render_insight_block_error(self, renderer, monkeypatch):
        """Test error handling in insight block rendering."""
        mock_warning = Mock()
        monkeypatch.setattr(st, "warning", mock_warning)
        
        # Test with None data
        renderer.render_insight_block(None)
        mock_warning.assert_called_once_with("No insight data available")
        
        # Test with malformed data
        mock_warning.reset_mock()
        with patch.object(renderer, '_detect_anomalies', side_effect=Exception("Processing error")):
            renderer.render_insight_block({"invalid": "data"})
            assert mock_warning.called

    def test_render_error_state_simple(self, renderer, error_state_payload):
        """Test rendering simple error states."""
        with patch('streamlit.error') as mock_error:
            render_error_state(error_state_payload["simple_error"])
            mock_error.assert_called_once_with(
                "⚠️ Data unavailable",
                icon="🚫"
            )

    def test_render_error_state_complex(self, renderer, error_state_payload):
        """Test rendering complex error states."""
        with patch('streamlit.error') as mock_error:
            render_error_state(error_state_payload["complex_error"]["message"])
            mock_error.assert_called_once_with(
                "⚠️ Processing failed",
                icon="🚫"
            )
            
    def test_metric_group_rendering(self, renderer, metric_group_payload):
        """Test rendering metric groups with various configurations."""
        metrics = metric_group_payload["basic"]
        
        with patch('streamlit.columns') as mock_columns:
            mock_cols = [Mock(), Mock(), Mock()]
            mock_columns.return_value = mock_cols
            
            with patch('streamlit.metric') as mock_metric:
                # Test with standard metrics dictionary
                render_metric_group(metrics, "Sales Metrics")
                
                assert mock_columns.called
                assert mock_metric.call_count == len(metrics)
                
                # Verify metric calls
                calls = mock_metric.call_args_list
                assert any(call[0][0] == "Total Sales" for call in calls)
                assert any(call[0][0] == "Average Price" for call in calls)
                assert any(call[0][0] == "Conversion Rate" for call in calls)

    def test_metric_group_with_thresholds(self, renderer, metric_group_payload):
        """Test metric group rendering with thresholds for highlighting."""
        metrics = metric_group_payload["with_thresholds"]
        
        with patch('streamlit.columns') as mock_columns:
            mock_cols = [Mock(), Mock(), Mock()]
            mock_columns.return_value = mock_cols
            
            with patch('streamlit.metric') as mock_metric:
                render_metric_group(metrics, "Performance Metrics")
                
                calls = mock_metric.call_args_list
                # For metrics with thresholds, verify delta colors are applied
                margin_call = next(call for call in calls if call[0][0] == "Margin")
                assert "delta_color" in margin_call[1]
                assert margin_call[1]["delta_color"] == "inverse"  # Below threshold = red
                
                volume_call = next(call for call in calls if call[0][0] == "Volume")
                assert "delta_color" in volume_call[1]
                assert volume_call[1]["delta_color"] == "normal"  # Above threshold = green

    def test_data_preprocessing_edge_cases(self, renderer):
        """Test data preprocessing with edge cases."""
        # Test empty metrics list
        empty_data = {"data": pd.DataFrame(), "metrics": {}}
        assert renderer._preprocess_data(empty_data) is not None
        
        # Test NaN/None values
        nan_data = {"data": pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": [4, 5, None]
        })}
        processed = renderer._preprocess_data(nan_data)
        assert processed is not None
        assert not processed["data"].isna().any().any()
        
        # Test invalid threshold dict
        invalid_thresh = {
            "metrics": {"thresholds": "invalid"}
        }
        with patch('streamlit.warning') as mock_warning:
            processed = renderer._preprocess_data(invalid_thresh)
            assert processed is not None
            assert mock_warning.called

    def test_render_chart_visualization_system(self, renderer, visualization_payload, monkeypatch):
        """Test rendering charts using the new visualization system."""
        mock_viz = Mock()
        mock_warning = Mock()
        monkeypatch.setattr('watchdog_ai.core.visualization.render_chart', mock_viz)
        monkeypatch.setattr(st, 'warning', mock_warning)
        
        # Test bar chart
        render_chart(visualization_payload["bar"], use_legacy=False)
        mock_viz.assert_called_once()
        mock_viz.reset_mock()
        
        # Test scatter with trend line (should use legacy)
        render_chart(visualization_payload["scatter"], use_legacy=False)
        assert not mock_viz.called
        assert mock_warning.call_count == 0  # No warning since it's intentional fallback
        
        # Test pie chart
        render_chart(visualization_payload["pie"], use_legacy=False)
        mock_viz.assert_called_once()
        
        # Test invalid chart type
        invalid_chart = visualization_payload["bar"].copy()
        invalid_chart["type"] = "invalid_type"
        render_chart(invalid_chart, use_legacy=False)
        assert mock_warning.called

    def test_format_timedelta_ranges(self):
        """Test format_timedelta with different time ranges."""
        # Test minutes
        td = timedelta(minutes=45)
        assert format_timedelta(td) == "45 minutes"
        td = timedelta(minutes=1)
        assert format_timedelta(td) == "1 minute"
        
        # Test hours
        td = timedelta(hours=2.5)
        assert format_timedelta(td) == "2.5 hours"
        td = timedelta(hours=1)
        assert format_timedelta(td) == "1.0 hour"
        
        # Test days
        td = timedelta(days=3.5)
        assert format_timedelta(td) == "3.5 days"
        td = timedelta(days=1)
        assert format_timedelta(td) == "1.0 day"

    def test_render_insight_empty_edge_cases(self, renderer, streamlit_mocker):
        """Test rendering insights with empty or edge case data."""
        # Test with empty metrics
        render_insight_block({
            "summary": "Test",
            "metrics": {}
        })
        assert not streamlit_mocker["error"].called
        
        # Test with empty visualizations
        render_insight_block({
            "summary": "Test",
            "metrics": {"total_records": 0},
            "visualization": {}
        })
        assert not streamlit_mocker["error"].called
        
        # Test with malformed metrics
        render_insight_block({
            "summary": "Test",
            "metrics": {"response_time_metrics": None}
        })
        assert not streamlit_mocker["error"].called

    def test_nested_metric_rendering(self, renderer, metric_group_payload, streamlit_mocker):
        """Test rendering of nested metric structures."""
        nested_metrics = {
            "group1": {
                "submetric1": 100,
                "submetric2": 200
            },
            "group2": {
                "submetric3": 300
            }
        }
        
        render_metric_group(nested_metrics, "Nested Metrics")
        assert streamlit_mocker["columns"].called
        assert streamlit_mocker["metric"].call_count == 3

    def test_chart_error_handling(self, renderer, visualization_payload, streamlit_mocker):
        """Test error handling in chart rendering."""
        # Test with missing required fields
        invalid_chart = {
            "type": "bar",
            # Missing data field
        }
        render_chart(invalid_chart)
        assert streamlit_mocker["warning"].called
        
        # Test with invalid data format
        invalid_data_chart = visualization_payload["bar"].copy()
        invalid_data_chart["data"] = "invalid"
        render_chart(invalid_data_chart)
        assert streamlit_mocker["warning"].called or streamlit_mocker["error"].called

    def test_report_renderer_integration(self, renderer, insight_data_complete, streamlit_mocker):
        """Test full integration of report renderer components."""
        # Verify the complete flow works
        renderer.render_insight_block(insight_data_complete)
        
        # Check all major components were called
        assert streamlit_mocker["markdown"].called
        assert streamlit_mocker["metric"].called
        assert streamlit_mocker["columns"].called
        
        # Check visualization was rendered
        assert any(
            "Daily Sales Trend" in str(call) 
            for call in streamlit_mocker["markdown"].call_args_list
        )
        
        # Verify findings section
        findings_calls = [
            call for call in streamlit_mocker["markdown"].call_args_list
            if "Key Findings" in str(call)
        ]
        assert len(findings_calls) > 0
