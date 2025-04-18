"""
Customer Validation Profile for Watchdog AI.

This module defines customer-specific validation rules and profiles for
validating customer aspects of dealership data, such as lead sources,
salesperson information, and other customer-related metrics.
"""

import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple, Optional

from ..base_validator import BaseValidator
from ..validation_profile import ValidationRule, ValidationProfile, apply_validation_rule
from ..registry import register_validator

class CustomerValidator(BaseValidator):
    """
    Validator for customer aspects of dealership data.
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
        Validate the provided DataFrame using customer validation rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (validated DataFrame, validation results)
        """
        result_df = df.copy()
        flag_counts = {}
        
        enabled_rules = self.profile.get_enabled_rules()
        customer_rules = [r for r in enabled_rules if r.category.lower() in ('marketing', 'personnel', 'data quality', 'customer', 'service')]
        
        for rule in customer_rules:
            result_df, count = apply_validation_rule(result_df, rule)
            flag_counts[rule.id] = count
        
        return result_df, flag_counts
    
    def get_rules(self) -> List[ValidationRule]:
        """
        Get the list of validation rules.
        
        Returns:
            List of validation rules
        """
        return [r for r in self.profile.rules if r.category.lower() in ('marketing', 'personnel', 'data quality', 'customer', 'service')]
    
    def get_name(self) -> str:
        """
        Get the name of the validator.
        
        Returns:
            Name of the validator
        """
        return "Customer Validator"
    
    def get_description(self) -> str:
        """
        Get the description of the validator.
        
        Returns:
            Description of the validator
        """
        return "Validates customer aspects of dealership data"


def create_customer_rules() -> List[ValidationRule]:
    """
    Create customer validation rules.
    
    Returns:
        List of customer validation rules
    """
    return [
        ValidationRule(
            id="missing_lead_source",
            name="Missing Lead Source",
            description="Flags records where lead source information is missing, which prevents accurate marketing ROI analysis.",
            enabled=True,
            severity="Medium",
            category="Marketing",
            column_mapping={"lead_source": "Lead_Source"}
        ),
        ValidationRule(
            id="incomplete_sale",
            name="Incomplete Sale Record",
            description="Flags sales records with missing critical information like price, cost, or date.",
            enabled=False,
            severity="Medium",
            category="Data Quality",
            column_mapping={
                "sale_price": "Sale_Price",
                "cost": "Cost",
                "sale_date": "Sale_Date"
            }
        ),
        ValidationRule(
            id="anomalous_price",
            name="Anomalous Sale Price",
            description="Flags sales with prices that deviate significantly from typical values for the model.",
            enabled=False,
            severity="Low",
            category="Data Quality",
            threshold_value=2.0,
            threshold_unit="std",
            threshold_operator=">",
            column_mapping={
                "sale_price": "Sale_Price",
                "model": "Model"
            }
        ),
        ValidationRule(
            id="invalid_date",
            name="Invalid Sale Date",
            description="Flags records with invalid or future sale dates.",
            enabled=False,
            severity="Medium",
            category="Data Quality",
            column_mapping={"sale_date": "Sale_Date"}
        ),
        ValidationRule(
            id="missing_salesperson",
            name="Missing Salesperson",
            description="Flags records where the salesperson information is missing.",
            enabled=False,
            severity="Low",
            category="Personnel",
            column_mapping={"salesperson": "Salesperson"}
        ),
        ValidationRule(
            id="warranty_claim_invalid",
            name="Invalid Warranty Claim",
            description="Flags warranty claims submitted after the likely warranty expiration based on date/mileage.",
            enabled=False,
            severity="Medium",
            category="Service",
            column_mapping={"claim_date": "ClaimDate", "service_date": "ServiceDate", "mileage": "MileageAtService"}
        ),
        ValidationRule(
            id="missing_technician",
            name="Missing Technician ID",
            description="Flags repair orders where the Technician ID is missing.",
            enabled=False,
            severity="Low",
            category="Service",
            column_mapping={"tech_id": "TechnicianID"}
        )
    ]


def create_customer_profile() -> ValidationProfile:
    """
    Create a customer validation profile.
    
    Returns:
        Customer validation profile
    """
    timestamp = datetime.datetime.now().isoformat()
    
    return ValidationProfile(
        id="customer",
        name="Customer Validation Profile",
        description="Validates customer aspects of dealership data",
        created_at=timestamp,
        updated_at=timestamp,
        rules=create_customer_rules(),
        is_default=False
    )


# Register the validator with the registry
register_validator = globals().get('register_validator')
if register_validator:
    register_validator('customer', CustomerValidator)