"""
Financial Validation Profile for Watchdog AI.

This module defines financial-specific validation rules and profiles for 
validating financial aspects of dealership data, such as gross profit, 
APR, loan terms, and other financial metrics.
"""

import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple, Optional

from ..base_validator import BaseValidator
from ..validation_profile import ValidationRule, ValidationProfile, apply_validation_rule
from ..registry import register_validator

class FinancialValidator(BaseValidator):
    """
    Validator for financial aspects of dealership data.
    """
    
    def __init__(self, profile: ValidationProfile):
        """
        Initialize the validator with a profile.
        
        Args:
            profile: The validation profile to use
        """
        self.profile = profile
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate the provided DataFrame using financial validation rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (validated DataFrame, validation results)
        """
        result_df = df.copy()
        flag_counts = {}
        
        enabled_rules = self.profile.get_enabled_rules()
        financial_rules = [r for r in enabled_rules if r.category.lower() in ('financial', 'finance')]
        
        for rule in financial_rules:
            result_df, count = apply_validation_rule(result_df, rule)
            flag_counts[rule.id] = count
        
        return result_df, flag_counts
    
    def get_rules(self) -> List[ValidationRule]:
        """
        Get the list of validation rules.
        
        Returns:
            List of validation rules
        """
        return [r for r in self.profile.rules if r.category.lower() in ('financial', 'finance')]
    
    def get_name(self) -> str:
        """
        Get the name of the validator.
        
        Returns:
            Name of the validator
        """
        return "Financial Validator"
    
    def get_description(self) -> str:
        """
        Get the description of the validator.
        
        Returns:
            Description of the validator
        """
        return "Validates financial aspects of dealership data"


def create_financial_rules() -> List[ValidationRule]:
    """
    Create financial validation rules.
    
    Returns:
        List of financial validation rules
    """
    timestamp = datetime.datetime.now().isoformat()
    
    return [
        ValidationRule(
            id="negative_gross",
            name="Negative Gross Profit",
            description="Flags transactions with negative gross profit, which may indicate pricing errors or issues with cost allocation.",
            enabled=True,
            severity="High",
            category="Financial",
            threshold_value=0,
            threshold_unit="$",
            threshold_operator=">=",
            column_mapping={"gross_profit": "Gross_Profit"}
        ),
        ValidationRule(
            id="low_gross",
            name="Low Gross Profit",
            description="Flags transactions with unusually low gross profit, which may indicate pricing issues or missed profit opportunities.",
            enabled=False,
            severity="Medium",
            category="Financial",
            threshold_value=500,
            threshold_unit="$",
            threshold_operator=">=",
            column_mapping={"gross_profit": "Gross_Profit"}
        ),
        ValidationRule(
            id="apr_out_of_range",
            name="APR Out of Range",
            description="Flags finance deals with Annual Percentage Rates (APR) outside a defined reasonable range.",
            enabled=False,
            severity="Medium",
            category="Finance",
            threshold_value=25.0,
            threshold_unit="%",
            threshold_operator="<=",
            column_mapping={"apr": "APR"}
        ),
        ValidationRule(
            id="loan_term_out_of_range",
            name="Loan Term Out of Range",
            description="Flags finance deals with loan terms outside typical ranges (e.g., <12 or >96 months).",
            enabled=False,
            severity="Low",
            category="Finance",
            threshold_value=96.0,
            threshold_unit="months",
            threshold_operator="<=",
            column_mapping={"term": "LoanTerm"}
        ),
        ValidationRule(
            id="missing_lender",
            name="Missing Lender Info",
            description="Flags financed deals where the lender information is missing.",
            enabled=False,
            severity="Medium",
            category="Finance",
            column_mapping={"lender": "LenderName"}
        )
    ]


def create_financial_profile() -> ValidationProfile:
    """
    Create a financial validation profile.
    
    Returns:
        Financial validation profile
    """
    timestamp = datetime.datetime.now().isoformat()
    
    return ValidationProfile(
        id="financial",
        name="Financial Validation Profile",
        description="Validates financial aspects of dealership data",
        created_at=timestamp,
        updated_at=timestamp,
        rules=create_financial_rules(),
        is_default=False
    )


# Register the validator with the registry
register_validator = globals().get('register_validator')
if register_validator:
    register_validator('financial', FinancialValidator)