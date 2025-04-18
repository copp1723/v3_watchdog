"""
Financial Validator for Watchdog AI.

This module implements validation rules for financial data in automotive dealerships.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.validators.base_validator import BaseValidator, BaseRule, ValidationError

class FinancialValidator(BaseValidator):
    """
    Validator for financial data in automotive dealerships.
    
    Validates gross profit, pricing, APR, loan terms, and other financial metrics.
    """
    
    def __init__(self):
        """Initialize the financial validator."""
        super().__init__(
            name="Financial Validator", 
            description="Validates financial data including gross profit, APR, and loan terms"
        )
        self.rules = self._create_financial_rules()
    
    def validate(self, df: pd.DataFrame) -> List[ValidationError]:
        """
        Validate the provided DataFrame for financial issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of ValidationError objects
        """
        print(f"[DEBUG] Starting financial validation") # Log start
        errors = []
        
        # Apply each rule
        for rule in self.rules:
            result_df, count = self.apply_rule(df, rule)
            if count > 0:
                errors.append(self.format_error(f"{rule.name} found {count} issues", rule.id))
        
        # Example rule: Check that 'balance' column exists
        if not self.check_column_exists(df, 'balance'):
            errors.append(self.format_error("Missing 'balance' column", 'balance'))
        
        print(f"[DEBUG] Finished financial validation, found {sum(len(e) for e in errors)} issues")
        return errors
    
    def get_rules(self) -> List[BaseRule]:
        """
        Get the list of financial validation rules.
        
        Returns:
            List of financial validation rules
        """
        return self.rules
    
    def _create_financial_rules(self) -> List[BaseRule]:
        """
        Create financial validation rules.
        
        Returns:
            List of financial validation rules
        """
        return [
            BaseRule(
                id="negative_gross",
                name="Negative Gross Profit",
                description="Identifies deals with negative gross profit, which may indicate errors or special circumstances",
                enabled=True,
                severity="High",
                category="Finance",
                column_mapping={"gross_profit": "Gross_Profit"},
                threshold_value=0,
                threshold_operator="<"
            ),
            BaseRule(
                id="low_gross",
                name="Low Gross Profit",
                description="Identifies deals with unusually low (but not negative) gross profit",
                enabled=True,
                severity="Medium",
                category="Finance",
                column_mapping={"gross_profit": "Gross_Profit"},
                threshold_value=500,
                threshold_operator="<"
            ),
            BaseRule(
                id="high_gross",
                name="Unusually High Gross Profit",
                description="Identifies deals with unusually high gross profit that may indicate errors",
                enabled=True,
                severity="Medium",
                category="Finance",
                column_mapping={"gross_profit": "Gross_Profit"},
                threshold_value=10000,
                threshold_operator=">"
            ),
            BaseRule(
                id="apr_out_of_range",
                name="APR Out of Range",
                description="Identifies deals with APR values outside the typical range",
                enabled=True,
                severity="Medium",
                category="Finance",
                column_mapping={"apr": "APR"},
                threshold_value=25.0,  # Max reasonable APR
                threshold_operator=">"
            ),
            BaseRule(
                id="loan_term_out_of_range",
                name="Loan Term Out of Range",
                description="Identifies deals with loan terms outside the typical range (12-96 months)",
                enabled=True,
                severity="Medium",
                category="Finance",
                column_mapping={"term": "LoanTerm"},
                threshold_value=96.0,  # Max reasonable term in months
                threshold_operator=">"
            ),
            BaseRule(
                id="missing_lender",
                name="Missing Lender Information",
                description="Identifies deals with missing lender information when financing is used",
                enabled=True,
                severity="Low",
                category="Finance",
                column_mapping={"lender": "LenderName"}
            )
        ]
    
    def _apply_rule_implementation(self, df: pd.DataFrame, rule: BaseRule, flag_column: str) -> pd.DataFrame:
        """
        Apply a financial validation rule to a DataFrame.
        
        Args:
            df: The input DataFrame
            rule: The validation rule to apply
            flag_column: The name of the flag column to use
            
        Returns:
            DataFrame with the flag column updated
        """
        # Map standard column names to actual DataFrame columns
        column_mapping = rule.column_mapping
        
        if rule.id == "negative_gross":
            # Get the correct column name for gross profit
            gross_col = column_mapping.get("gross_profit", "Gross_Profit")
            
            # Check if the column exists
            if gross_col not in df.columns:
                # Try to find a column that might contain gross information
                potential_cols = [col for col in df.columns if 'gross' in col.lower()]
                if potential_cols:
                    gross_col = potential_cols[0]
                else:
                    # If no gross column found, return with empty flag column
                    df[flag_column] = False
                    return df
            
            # Ensure the gross column is numeric
            if not pd.api.types.is_numeric_dtype(df[gross_col]):
                # Try to convert to numeric, coercing errors to NaN
                df[gross_col] = pd.to_numeric(df[gross_col], errors='coerce')
            
            # Apply threshold if specified
            if rule.threshold_value is not None and rule.threshold_operator is not None:
                if rule.threshold_operator == ">=":
                    df[flag_column] = df[gross_col] >= rule.threshold_value
                elif rule.threshold_operator == "<=":
                    df[flag_column] = df[gross_col] <= rule.threshold_value
                elif rule.threshold_operator == ">":
                    df[flag_column] = df[gross_col] > rule.threshold_value
                elif rule.threshold_operator == "<":
                    df[flag_column] = df[gross_col] < rule.threshold_value
                elif rule.threshold_operator == "==":
                    df[flag_column] = df[gross_col] == rule.threshold_value
                elif rule.threshold_operator == "!=":
                    df[flag_column] = df[gross_col] != rule.threshold_value
                else:
                    # Default behavior
                    df[flag_column] = df[gross_col] < 0
            else:
                # Default behavior (negative gross)
                df[flag_column] = df[gross_col] < 0
                
        elif rule.id == "low_gross":
            # Get the correct column name for gross profit
            gross_col = column_mapping.get("gross_profit", "Gross_Profit")
            
            # Check if the column exists
            if gross_col not in df.columns:
                # Try to find a column that might contain gross information
                potential_cols = [col for col in df.columns if 'gross' in col.lower()]
                if potential_cols:
                    gross_col = potential_cols[0]
                else:
                    # If no gross column found, return with empty flag column
                    df[flag_column] = False
                    return df
            
            # Ensure the gross column is numeric
            if not pd.api.types.is_numeric_dtype(df[gross_col]):
                # Try to convert to numeric, coercing errors to NaN
                df[gross_col] = pd.to_numeric(df[gross_col], errors='coerce')
            
            # Apply threshold if specified, default to $500
            threshold = rule.threshold_value if rule.threshold_value is not None else 500
            
            # Flag rows with gross profit below threshold (but not negative)
            df[flag_column] = (df[gross_col] < threshold) & (df[gross_col] >= 0)
            
        elif rule.id == "high_gross":
            # Get the correct column name for gross profit
            gross_col = column_mapping.get("gross_profit", "Gross_Profit")
            
            # Check if the column exists
            if gross_col not in df.columns:
                # Try to find a column that might contain gross information
                potential_cols = [col for col in df.columns if 'gross' in col.lower()]
                if potential_cols:
                    gross_col = potential_cols[0]
                else:
                    # If no gross column found, return with empty flag column
                    df[flag_column] = False
                    return df
            
            # Ensure the gross column is numeric
            if not pd.api.types.is_numeric_dtype(df[gross_col]):
                # Try to convert to numeric, coercing errors to NaN
                df[gross_col] = pd.to_numeric(df[gross_col], errors='coerce')
            
            # Apply threshold if specified, default to $10000
            threshold = rule.threshold_value if rule.threshold_value is not None else 10000
            
            # Flag rows with gross profit above threshold
            df[flag_column] = df[gross_col] > threshold
            
        elif rule.id == "apr_out_of_range":
            # Get the correct column name for APR
            apr_col = column_mapping.get("apr", "APR")
            
            # Check if the column exists
            if apr_col in df.columns:
                # Convert APR to numeric, removing % if needed
                df[apr_col] = pd.to_numeric(df[apr_col].astype(str).str.replace('%', '', regex=False), errors='coerce')
                
                # Set thresholds
                lower_threshold = 0.0 # Example lower bound
                upper_threshold = rule.threshold_value if rule.threshold_value is not None else 25.0
                
                # Flag rows outside the range
                df[flag_column] = (df[apr_col] < lower_threshold) | (df[apr_col] > upper_threshold)
            else:
                df[flag_column] = False
                
        elif rule.id == "loan_term_out_of_range":
            # Get the correct column name for loan term
            term_col = column_mapping.get("term", "LoanTerm")
            
            # Check if the column exists
            if term_col in df.columns:
                # Convert term to numeric
                df[term_col] = pd.to_numeric(df[term_col], errors='coerce')
                
                # Set thresholds
                lower_threshold = 12.0 # Example lower bound
                upper_threshold = rule.threshold_value if rule.threshold_value is not None else 96.0
                
                # Flag rows outside the range
                df[flag_column] = (df[term_col] < lower_threshold) | (df[term_col] > upper_threshold)
            else:
                df[flag_column] = False
                
        elif rule.id == "missing_lender":
            # Get the correct column name for lender
            lender_col = column_mapping.get("lender", "LenderName")
            
            # Check if the column exists
            if lender_col in df.columns:
                # Flag rows with missing or empty lender
                df[flag_column] = (
                    df[lender_col].isna() | 
                    (df[lender_col] == '') | 
                    (df[lender_col].astype(str).str.strip() == '')
                )
            else:
                df[flag_column] = False
                
        return df