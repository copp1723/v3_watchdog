"""
Customer Validator for Watchdog AI.

This module implements validation rules for customer data in automotive dealerships.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.validators.base_validator import BaseValidator, BaseRule, ValidationError

class CustomerValidator(BaseValidator):
    """
    Validator for customer data in automotive dealerships.
    
    Validates lead information, customer demographics, and transaction details.
    """
    
    required_columns = ['customer_id']
    
    def __init__(self):
        """Initialize the customer validator."""
        super().__init__(
            name="Customer Validator", 
            description="Validates customer and lead data for automotive dealerships"
        )
        self.rules = self._create_customer_rules()
    
    def validate(self, df: pd.DataFrame) -> List[ValidationError]:
        """
        Validate the provided DataFrame for customer data issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of ValidationError objects
        """
        errors = []
        for col in self.required_columns:
            if col not in df.columns:
                errors.append(self.format_error(f"Missing required column: {col}", col))
        return errors
    
    def get_rules(self) -> List:
        """
        Get the list of customer validation rules.
        
        Returns:
            List of customer validation rules
        """
        return self.rules
    
    def _create_customer_rules(self) -> List[BaseRule]:
        """
        Create customer validation rules.
        
        Returns:
            List of customer validation rules
        """
        return [
            BaseRule(
                id="missing_lead_source",
                name="Missing Lead Source",
                description="Identifies records with missing lead source information, which prevents marketing analysis",
                enabled=True,
                severity="Medium",
                category="Customer",
                column_mapping={"lead_source": "Lead_Source"}
            ),
            BaseRule(
                id="missing_salesperson",
                name="Missing Salesperson",
                description="Identifies records with missing salesperson information",
                enabled=True,
                severity="Medium",
                category="Customer",
                column_mapping={"salesperson": "Salesperson"}
            ),
            BaseRule(
                id="incomplete_sale",
                name="Incomplete Sale Information",
                description="Identifies records with critical missing information about the sale",
                enabled=True,
                severity="High",
                category="Customer",
                column_mapping={
                    "customer_name": "CustomerName",
                    "sale_date": "SaleDate",
                    "salesperson": "Salesperson",
                    "deal_type": "DealType"
                }
            ),
            BaseRule(
                id="invalid_date",
                name="Invalid Sale Date",
                description="Identifies records with invalid or future sale dates",
                enabled=True,
                severity="Medium",
                category="Customer",
                column_mapping={"sale_date": "SaleDate"}
            )
        ]
    
    def _apply_rule_implementation(self, df: pd.DataFrame, rule: BaseRule, flag_column: str) -> pd.DataFrame:
        """
        Apply a customer validation rule to a DataFrame.
        
        Args:
            df: The input DataFrame
            rule: The validation rule to apply
            flag_column: The name of the flag column to use
            
        Returns:
            DataFrame with the flag column updated
        """
        # Map standard column names to actual DataFrame columns
        column_mapping = rule.column_mapping
        
        if rule.id == "missing_lead_source":
            # Get the correct column name for lead source
            lead_source_col = column_mapping.get("lead_source", "Lead_Source")
            
            # Check if the column exists
            if lead_source_col not in df.columns:
                # Try to find a column that might contain lead source information
                potential_cols = [col for col in df.columns if 'lead' in col.lower() or 'source' in col.lower()]
                if potential_cols:
                    lead_source_col = potential_cols[0]
                else:
                    # If no lead source column found, return with empty flag column
                    df[flag_column] = False
                    return df
            
            # Flag rows with missing or empty lead source
            df[flag_column] = (
                df[lead_source_col].isna() | 
                (df[lead_source_col] == '') | 
                (df[lead_source_col].astype(str).str.strip() == '')
            )
            
        elif rule.id == "missing_salesperson":
            # Get the correct column name for salesperson
            salesperson_col = column_mapping.get("salesperson", "Salesperson")
            
            # Check if the column exists
            if salesperson_col not in df.columns:
                potential_cols = [col for col in df.columns if any(term in col.lower() for term in ['salesperson', 'sales_person', 'sales person', 'agent', 'rep'])]
                if potential_cols:
                    salesperson_col = potential_cols[0]
                else:
                    df[flag_column] = False
                    return df
            
            # Flag rows with missing salesperson
            df[flag_column] = (
                df[salesperson_col].isna() | 
                (df[salesperson_col] == '') | 
                (df[salesperson_col].astype(str).str.strip() == '')
            )
            
        elif rule.id == "incomplete_sale":
            # Check for missing values in critical columns
            fields_to_check = {}
            
            for std_col, df_col in column_mapping.items():
                if df_col in df.columns:
                    fields_to_check[std_col] = df_col
            
            if not fields_to_check:
                df[flag_column] = False
                return df
            
            # Flag rows with any critical field missing
            df[flag_column] = False
            
            for std_col, df_col in fields_to_check.items():
                df[flag_column] = df[flag_column] | (
                    df[df_col].isna() | 
                    (df[df_col] == '') | 
                    (df[df_col].astype(str).str.strip() == '')
                )
                
        elif rule.id == "invalid_date":
            # Get the correct column name for sale date
            date_col = column_mapping.get("sale_date", "SaleDate")
            
            # Check if the column exists
            if date_col not in df.columns:
                potential_cols = [col for col in df.columns if 'date' in col.lower()]
                if potential_cols:
                    date_col = potential_cols[0]
                else:
                    df[flag_column] = False
                    return df
            
            # Try to convert to datetime
            date_series = pd.to_datetime(df[date_col], errors='coerce')
            
            # Flag invalid dates (NaT) or future dates
            today = pd.Timestamp(pd.Timestamp.now())
            df[flag_column] = date_series.isna() | (date_series > today)
            
        return df