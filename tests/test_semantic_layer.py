"""
Tests for semantic layer and business guardrails.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.rule_engine import BusinessRuleEngine, RuleValidationResult
from src.utils.adaptive_schema import SchemaProfileManager, SchemaAdjustment

@pytest.fixture
def rule_engine():
    """Create BusinessRuleEngine instance."""
    return BusinessRuleEngine("BusinessRuleRegistry.yaml")

@pytest.fixture
def schema_manager():
    """Create SchemaProfileManager instance."""
    return SchemaProfileManager()

@pytest.fixture
def sample_data():
    """Create sample DataFrame with known quality issues."""
    return pd.DataFrame({
        'total_gross_profit': [1000, -500, 2000, np.nan, 1500],
        'days_in_inventory': [30, 95, 45, 120, 60],
        'be_penetration': [80, 65, 75, np.nan, 85],
        'date': pd.date_range(start='2023-01-01', periods=5)
    })

def test_data_quality_evaluation(rule_engine, sample_data):
    """Test data quality rule evaluation."""
    # Calculate quality metrics
    quality_data = {
        "nan_percentage": 15.0,  # Should trigger warning but not error
        "sample_size": len(sample_data),
        "outlier_percentage": 5.0  # Mock value
    }
    
    result = rule_engine.evaluate_data_quality(quality_data)
    assert isinstance(result, RuleValidationResult)
    assert result.rule_id == "data_quality_missing"
    assert result.severity == "medium"  # 15% missing data should trigger warning

def test_schema_profile_loading(schema_manager):
    """Test schema profile loading and validation."""
    profile = schema_manager.get_profile("general_manager")
    assert profile is not None
    assert profile.role == "general_manager"
    assert len(profile.columns) > 0
    
    # Verify business rules
    gross_profit_col = next(col for col in profile.columns if col.name == "total_gross_profit")
    assert len(gross_profit_col.business_rules) == 1
    assert gross_profit_col.business_rules[0]["type"] == "comparison"
    assert gross_profit_col.business_rules[0]["operator"] == ">="
    assert gross_profit_col.business_rules[0]["threshold"] == 0

def test_schema_adjustments(schema_manager):
    """Test schema adjustment handling."""
    # Create a test adjustment
    adjustment = SchemaAdjustment(
        user_id="test_user",
        column_name="total_gross_profit",
        adjustment_type="alias",
        value="total_gp"
    )
    
    # Add adjustment
    success = schema_manager.add_adjustment(adjustment)
    assert success
    
    # Get adjusted profile
    adjusted_profile = schema_manager.get_adjusted_profile("general_manager", "test_user")
    assert adjusted_profile is not None
    
    # Verify adjustment was applied
    gross_profit_col = next(col for col in adjusted_profile.columns if col.name == "total_gross_profit")
    assert "total_gp" in gross_profit_col.aliases

def test_recommendation_templates():
    """Test recommendation template loading and formatting."""
    import yaml
    
    with open("config/recommendations.yml", 'r') as f:
        templates = yaml.safe_load(f)
    
    # Test data quality template
    missing_data_template = templates["templates"]["data_quality"]["missing_data"]["warning"]
    formatted = missing_data_template.format(column="total_gross_profit", missing_pct=20.0)
    assert "total_gross_profit" in formatted
    assert "20.0%" in formatted

def test_business_rule_evaluation(rule_engine, sample_data):
    """Test business rule evaluation."""
    # Test negative gross profit rule
    result = rule_engine.evaluate_rule(
        "gross_not_negative",
        {"total_gross_profit": -500}
    )
    assert not result.is_valid
    assert result.severity == "high"
    
    # Test days in inventory rule
    result = rule_engine.evaluate_rule(
        "max_days_in_inventory",
        {"days_in_inventory": 95}
    )
    assert not result.is_valid
    assert result.severity == "medium"  # This rule has medium severity