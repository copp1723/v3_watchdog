"""
Inventory Validation Profile for Watchdog AI.

This module defines inventory-specific validation rules and profiles for
validating inventory aspects of dealership data, such as VIN, stock numbers,
vehicle details, mileage, and other inventory-related metrics.
"""

import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple, Optional

from ..base_validator import BaseValidator
from ..validation_profile import ValidationRule, ValidationProfile, apply_validation_rule
from ..registry import register_validator

class InventoryValidator(BaseValidator):
    """
    Validator for inventory aspects of dealership data.
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
        Validate the provided DataFrame using inventory validation rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (validated DataFrame, validation results)
        """
        result_df = df.copy()
        flag_counts = {}
        
        enabled_rules = self.profile.get_enabled_rules()
        inventory_rules = [r for r in enabled_rules if r.category.lower() in ('inventory', 'vehicle', 'vehicle data')]
        
        for rule in inventory_rules:
            result_df, count = apply_validation_rule(result_df, rule)
            flag_counts[rule.id] = count
        
        return result_df, flag_counts
    
    def get_rules(self) -> List[ValidationRule]:
        """
        Get the list of validation rules.
        
        Returns:
            List of validation rules
        """
        return [r for r in self.profile.rules if r.category.lower() in ('inventory', 'vehicle', 'vehicle data')]
    
    def get_name(self) -> str:
        """
        Get the name of the validator.
        
        Returns:
            Name of the validator
        """
        return "Inventory Validator"
    
    def get_description(self) -> str:
        """
        Get the description of the validator.
        
        Returns:
            Description of the validator
        """
        return "Validates inventory aspects of dealership data"


def create_inventory_rules() -> List[ValidationRule]:
    """
    Create inventory validation rules.
    
    Returns:
        List of inventory validation rules
    """
    return [
        ValidationRule(
            id="duplicate_vin",
            name="Duplicate VIN",
            description="Flags records with duplicate VIN numbers, which may indicate data entry errors or multiple transactions on the same vehicle.",
            enabled=True,
            severity="Medium",
            category="Inventory",
            column_mapping={"vin": "VIN"}
        ),
        ValidationRule(
            id="missing_vin",
            name="Missing/Invalid VIN",
            description="Flags records with missing or improperly formatted VIN numbers, which complicates inventory tracking and reporting.",
            enabled=True,
            severity="High",
            category="Inventory",
            column_mapping={"vin": "VIN"}
        ),
        ValidationRule(
            id="mileage_discrepancy",
            name="Mileage-Year Discrepancy",
            description="Flags vehicles where reported mileage seems inconsistent with the vehicle year (e.g., very high for new, very low for old).",
            enabled=False,
            severity="Low",
            category="Vehicle Data",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"mileage": "Mileage", "year": "VehicleYear"}
        ),
        ValidationRule(
            id="new_used_logic",
            name="New/Used Status Logic",
            description='Flags inconsistencies like "New" vehicles with high mileage or "Used" vehicles with zero/missing mileage.',
            enabled=False,
            severity="Medium",
            category="Vehicle Data",
            threshold_value=100,
            threshold_unit="miles",
            threshold_operator="<=",
            column_mapping={"status": "NewUsedStatus", "mileage": "Mileage"}
        ),
        ValidationRule(
            id="title_issue",
            name="Potential Title Issue",
            description="Flags vehicles with reported title statuses like Salvage, Flood, Lemon, etc.",
            enabled=False,
            severity="High",
            category="Vehicle Data",
            column_mapping={"title_status": "TitleStatus"}
        ),
        ValidationRule(
            id="missing_vehicle_info",
            name="Missing Key Vehicle Info",
            description="Flags records missing essential vehicle identifiers like Make, Model, or Year.",
            enabled=False,
            severity="Medium",
            category="Vehicle Data",
            column_mapping={"make": "VehicleMake", "model": "VehicleModel", "year": "VehicleYear"}
        ),
        ValidationRule(
            id="duplicate_stock_number",
            name="Duplicate Stock Number",
            description="Flags records with the same stock number but different VINs, indicating potential inventory errors.",
            enabled=False,
            severity="Medium",
            category="Inventory",
            column_mapping={"stock_number": "VehicleStockNumber", "vin": "VehicleVIN"}
        )
    ]


def create_inventory_profile() -> ValidationProfile:
    """
    Create an inventory validation profile.
    
    Returns:
        Inventory validation profile
    """
    timestamp = datetime.datetime.now().isoformat()
    
    return ValidationProfile(
        id="inventory",
        name="Inventory Validation Profile",
        description="Validates inventory aspects of dealership data",
        created_at=timestamp,
        updated_at=timestamp,
        rules=create_inventory_rules(),
        is_default=False
    )


# Register the validator with the registry
register_validator = globals().get('register_validator')
if register_validator:
    register_validator('inventory', InventoryValidator)