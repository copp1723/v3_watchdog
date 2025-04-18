"""
Validation Profile for Watchdog AI.

This module defines validation profiles that allow users to toggle
specific validation rules on/off and set custom thresholds for each rule.
Profiles can be saved, loaded, and applied to customize the validation process.
"""

import os
import json
import pandas as pd
import datetime
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from pydantic import BaseModel, Field, validator
import streamlit as st


class ValidationRuleType(str, Enum):
    """Types of validation rules available in the system."""
    NEGATIVE_GROSS = "negative_gross"
    MISSING_LEAD_SOURCE = "missing_lead_source"
    DUPLICATE_VIN = "duplicate_vin"
    MISSING_VIN = "missing_vin"
    LOW_GROSS = "low_gross"
    HIGH_GROSS = "high_gross"
    INCOMPLETE_SALE = "incomplete_sale"
    ANOMALOUS_PRICE = "anomalous_price"
    INVALID_DATE = "invalid_date"
    MISSING_SALESPERSON = "missing_salesperson"
    
    # --- New Automotive Rules ---
    MILEAGE_DISCREPANCY = "mileage_discrepancy"
    NEW_USED_LOGIC = "new_used_logic"
    TITLE_ISSUE = "title_issue"
    MISSING_VEHICLE_INFO = "missing_vehicle_info"
    DUPLICATE_STOCK_NUMBER = "duplicate_stock_number"
    
    # --- New Finance Rules (Placeholders) ---
    APR_OUT_OF_RANGE = "apr_out_of_range"
    LOAN_TERM_OUT_OF_RANGE = "loan_term_out_of_range"
    MISSING_LENDER = "missing_lender"

    # --- New Service Rules (Placeholders) ---
    WARRANTY_CLAIM_INVALID = "warranty_claim_invalid"
    MISSING_TECHNICIAN = "missing_technician"


class ValidationRule(BaseModel):
    """Definition of a validation rule with its parameters."""
    id: str = Field(..., description="Unique identifier for the rule")
    name: str = Field(..., description="Display name for the rule")
    description: str = Field(..., description="Detailed description of what the rule checks")
    enabled: bool = Field(True, description="Whether the rule is enabled")
    severity: str = Field("Medium", description="Severity level (High, Medium, Low)")
    category: str = Field("Data Quality", description="Category for grouping rules")
    
    # Threshold parameters (optional, depends on rule type)
    threshold_value: Optional[float] = Field(None, description="Threshold value for numeric rules")
    threshold_unit: Optional[str] = Field(None, description="Unit for the threshold (e.g., ', '%')")
    threshold_operator: Optional[str] = Field(None, description="Operator for comparison (e.g., '>=', '<=')")
    
    # Column mapping
    column_mapping: Dict[str, str] = Field(default_factory=dict, description="Mapping of standard column names to dataset column names")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ["High", "Medium", "Low"]
        if v not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}")
        return v
    
    @validator('threshold_operator')
    def validate_operator(cls, v):
        if v is not None:
            valid_operators = ["==", "!=", ">", "<", ">=", "<="]
            if v not in valid_operators:
                raise ValueError(f"Operator must be one of {valid_operators}")
        return v


class ValidationProfile(BaseModel):
    """
    A validation profile containing a set of validation rules and their configurations.
    Profiles can be saved, loaded, and applied to customize the validation process.
    """
    id: str = Field(..., description="Unique identifier for the profile")
    name: str = Field(..., description="Display name for the profile")
    description: str = Field("", description="Description of the profile")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    rules: List[ValidationRule] = Field(default_factory=list, description="List of validation rules in this profile")
    is_default: bool = Field(False, description="Whether this is the default profile")
    
    def get_enabled_rules(self) -> List[ValidationRule]:
        """Get only the enabled rules from this profile."""
        return [rule for rule in self.rules if rule.enabled]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[ValidationRule]:
        """Get a rule by its ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationProfile":
        """Create a profile from a dictionary."""
        # Ensure rules are properly instantiated as ValidationRule objects
        if "rules" in data and isinstance(data["rules"], list):
            data["rules"] = [
                rule if isinstance(rule, ValidationRule) else ValidationRule(**rule)
                for rule in data["rules"]
            ]
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """
        Save the profile to a JSON file.
        
        Args:
            directory: Directory to save the profile to
            
        Returns:
            Path to the saved profile file
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return file_path
    
    @classmethod
    def load(cls, file_path: str) -> "ValidationProfile":
        """
        Load a profile from a JSON file.
        
        Args:
            file_path: Path to the profile JSON file
            
        Returns:
            Loaded ValidationProfile object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


def create_default_rules() -> List[ValidationRule]:
    """Create a list of default validation rules."""
    # TODO: Add unit tests for these rules
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
            id="missing_lead_source",
            name="Missing Lead Source",
            description="Flags records where lead source information is missing, which prevents accurate marketing ROI analysis.",
            enabled=True,
            severity="Medium",
            category="Marketing",
            column_mapping={"lead_source": "Lead_Source"}
        ),
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
            category="Financial",
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
        
        # --- New Automotive Rules Definitions ---
        ValidationRule(
            id="mileage_discrepancy",
            name="Mileage-Year Discrepancy",
            description="Flags vehicles where reported mileage seems inconsistent with the vehicle year (e.g., very high for new, very low for old).",
            enabled=False, # Disabled by default
            severity="Low",
            category="Vehicle Data",
            threshold_value=None, # Logic might compare mileage ranges based on year
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"mileage": "Mileage", "year": "VehicleYear"}
        ),
        ValidationRule(
            id="new_used_logic",
            name="New/Used Status Logic",
            description='Flags inconsistencies like "New" vehicles with high mileage or "Used" vehicles with zero/missing mileage.',
            enabled=False, # Disabled by default
            severity="Medium",
            category="Vehicle Data",
            threshold_value=100, # Example threshold for "New" mileage
            threshold_unit="miles",
            threshold_operator="<=",
            column_mapping={"status": "NewUsedStatus", "mileage": "Mileage"}
        ),
        ValidationRule(
            id="title_issue",
            name="Potential Title Issue",
            description="Flags vehicles with reported title statuses like Salvage, Flood, Lemon, etc.",
            enabled=False, # Disabled by default
            severity="High",
            category="Vehicle Data",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"title_status": "TitleStatus"}
        ),
        ValidationRule(
            id="missing_vehicle_info",
            name="Missing Key Vehicle Info",
            description="Flags records missing essential vehicle identifiers like Make, Model, or Year.",
            enabled=False, # Disabled by default
            severity="Medium",
            category="Data Quality",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"make": "VehicleMake", "model": "VehicleModel", "year": "VehicleYear"}
        ),
        ValidationRule(
            id="duplicate_stock_number",
            name="Duplicate Stock Number",
            description="Flags records with the same stock number but different VINs, indicating potential inventory errors.",
            enabled=False, # Disabled by default
            severity="Medium",
            category="Inventory",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"stock_number": "VehicleStockNumber", "vin": "VehicleVIN"}
        ),
        
        # --- New Finance Rule Definitions (Placeholders) ---
        ValidationRule(
            id="apr_out_of_range",
            name="APR Out of Range",
            description="Flags finance deals with Annual Percentage Rates (APR) outside a defined reasonable range.",
            enabled=False, # Disabled by default
            severity="Medium",
            category="Finance",
            threshold_value=25.0, # Example upper threshold
            threshold_unit="%",
            threshold_operator="<=", # Checks if APR <= threshold
            column_mapping={"apr": "APR"}
        ),
        ValidationRule(
            id="loan_term_out_of_range",
            name="Loan Term Out of Range",
            description="Flags finance deals with loan terms outside typical ranges (e.g., <12 or >96 months).",
            enabled=False, # Disabled by default
            severity="Low",
            category="Finance",
            threshold_value=96.0, # Example upper threshold
            threshold_unit="months",
            threshold_operator="<=",
            column_mapping={"term": "LoanTerm"}
        ),
        ValidationRule(
            id="missing_lender",
            name="Missing Lender Info",
            description="Flags financed deals where the lender information is missing.",
            enabled=False, # Disabled by default
            severity="Medium",
            category="Finance",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"lender": "LenderName"}
        ),
        
        # --- New Service Rule Definitions (Placeholders) ---
        ValidationRule(
            id="warranty_claim_invalid",
            name="Invalid Warranty Claim",
            description="Flags warranty claims submitted after the likely warranty expiration based on date/mileage.",
            enabled=False, # Disabled by default
            severity="Medium",
            category="Service",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"claim_date": "ClaimDate", "service_date": "ServiceDate", "mileage": "MileageAtService"}
        ),
        ValidationRule(
            id="missing_technician",
            name="Missing Technician ID",
            description="Flags repair orders where the Technician ID is missing.",
            enabled=False, # Disabled by default
            severity="Low",
            category="Service",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"tech_id": "TechnicianID"}
        )
        # TODO: Add more rules based on docs/validation_ideas.md
    ]


def create_default_profile() -> ValidationProfile:
    """Create the default validation profile."""
    timestamp = datetime.datetime.now().isoformat()
    
    return ValidationProfile(
        id="default",
        name="Default Profile",
        description="Default validation profile with standard rules for dealership data.",
        created_at=timestamp,
        updated_at=timestamp,
        rules=create_default_rules(),
        is_default=True
    )


def create_minimal_profile() -> ValidationProfile:
    """Create a minimal validation profile with only essential rules enabled."""
    timestamp = datetime.datetime.now().isoformat()
    
    rules = create_default_rules()
    
    # Only enable essential rules
    essential_rule_ids = {"negative_gross", "duplicate_vin"}
    for rule in rules:
        rule.enabled = rule.id in essential_rule_ids
    
    return ValidationProfile(
        id="minimal",
        name="Minimal Profile",
        description="Minimal validation profile with only essential rules enabled.",
        created_at=timestamp,
        updated_at=timestamp,
        rules=rules,
        is_default=False
    )


def create_comprehensive_profile() -> ValidationProfile:
    """Create a comprehensive validation profile with all rules enabled."""
    timestamp = datetime.datetime.now().isoformat()
    
    rules = create_default_rules()
    
    # Enable all rules
    for rule in rules:
        rule.enabled = True
    
    return ValidationProfile(
        id="comprehensive",
        name="Comprehensive Profile",
        description="Comprehensive validation profile with all rules enabled.",
        created_at=timestamp,
        updated_at=timestamp,
        rules=rules,
        is_default=False
    )


def get_available_profiles(profiles_dir: str) -> List[ValidationProfile]:
    """
    Get a list of all available validation profiles.
    
    Args:
        profiles_dir: Directory containing profile JSON files
        
    Returns:
        List of ValidationProfile objects
    """
    profiles = []
    
    # Ensure directory exists
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Load profiles from files
    for filename in os.listdir(profiles_dir):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(profiles_dir, filename)
                profile = ValidationProfile.load(file_path)
                profiles.append(profile)
            except Exception as e:
                print(f"Error loading profile {filename}: {str(e)}")
    
    # If no profiles found, add default profiles
    if not profiles:
        default_profile = create_default_profile()
        minimal_profile = create_minimal_profile()
        comprehensive_profile = create_comprehensive_profile()
        
        default_profile.save(profiles_dir)
        minimal_profile.save(profiles_dir)
        comprehensive_profile.save(profiles_dir)
        
        profiles = [default_profile, minimal_profile, comprehensive_profile]
    
    # Sort profiles (default first, then alphabetically)
    profiles.sort(key=lambda p: (0 if p.is_default else 1, p.name))
    
    return profiles


def apply_validation_rule(df: pd.DataFrame, rule: ValidationRule) -> Tuple[pd.DataFrame, int]:
    """
    Apply a validation rule to a DataFrame.
    
    Args:
        df: The input DataFrame
        rule: The validation rule to apply
        
    Returns:
        Tuple of (DataFrame with added flag column, count of flagged rows)
    """
    print(f"[DEBUG] Applying rule: {rule.id} ('{rule.name}')") # Log start
    
    # Failsafe: Check all mapped columns exist
    for logical_name, actual_column in rule.column_mapping.items():
        if actual_column not in df.columns:
            print(f"[WARN] Column '{actual_column}' not found for rule '{rule.id}'. Skipping rule.") # Log skip
            flag_column = f"flag_{rule.id}"
            # Ensure flag column exists even if rule skipped, initialized to False
            if flag_column not in df.columns:
                 df[flag_column] = False
            return df, 0
    
    if not rule.enabled:
        print(f"[DEBUG] Rule '{rule.id}' is disabled. Skipping.") # Log skip
        return df, 0
    
    result_df = df.copy()
    flag_column = f"flag_{rule.id}"
    
    # Initialize flag column to False if it doesn't exist
    if flag_column not in result_df.columns:
        result_df[flag_column] = False
        
    # Map standard column names to actual DataFrame columns
    column_mapping = rule.column_mapping
    
    # Apply rule based on type
    if rule.id == "negative_gross":
        # Get the correct column name for gross profit
        gross_col = column_mapping.get("gross_profit", "Gross_Profit")
        
        # Check if the column exists
        if gross_col not in result_df.columns:
            # Try to find a column that might contain gross information
            potential_cols = [col for col in result_df.columns if 'gross' in col.lower()]
            if potential_cols:
                gross_col = potential_cols[0]
            else:
                # If no gross column found, return with empty flag column
                result_df[flag_column] = False
                return result_df, 0
        
        # Ensure the gross column is numeric
        if not pd.api.types.is_numeric_dtype(result_df[gross_col]):
            # Try to convert to numeric, coercing errors to NaN
            result_df[gross_col] = pd.to_numeric(result_df[gross_col], errors='coerce')
        
        # Apply threshold if specified
        if rule.threshold_value is not None and rule.threshold_operator is not None:
            if rule.threshold_operator == ">=":
                result_df[flag_column] = result_df[gross_col] < rule.threshold_value
            elif rule.threshold_operator == "<=":
                result_df[flag_column] = result_df[gross_col] > rule.threshold_value
            elif rule.threshold_operator == ">":
                result_df[flag_column] = result_df[gross_col] <= rule.threshold_value
            elif rule.threshold_operator == "<":
                result_df[flag_column] = result_df[gross_col] >= rule.threshold_value
            elif rule.threshold_operator == "==":
                result_df[flag_column] = result_df[gross_col] == rule.threshold_value
            elif rule.threshold_operator == "!=":
                result_df[flag_column] = result_df[gross_col] != rule.threshold_value
            else:
                # Default behavior
                result_df[flag_column] = result_df[gross_col] < 0
        else:
            # Default behavior (negative gross)
            result_df[flag_column] = result_df[gross_col] < 0
    
    elif rule.id == "missing_lead_source":
        # Get the correct column name for lead source
        lead_source_col = column_mapping.get("lead_source", "Lead_Source")
        
        # Check if the column exists
        if lead_source_col not in result_df.columns:
            # Try to find a column that might contain lead source information
            potential_cols = [col for col in result_df.columns if 'lead' in col.lower() or 'source' in col.lower()]
            if potential_cols:
                lead_source_col = potential_cols[0]
            else:
                # If no lead source column found, return with empty flag column
                result_df[flag_column] = False
                return result_df, 0
        
        # Flag rows with missing or empty lead source
        result_df[flag_column] = (
            result_df[lead_source_col].isna() | 
            (result_df[lead_source_col] == '') | 
            (result_df[lead_source_col].astype(str).str.strip() == '')
        )
    
    elif rule.id == "duplicate_vin":
        # Get the correct column name for VIN
        vin_col = column_mapping.get("vin", "VIN")
        
        # Check if the column exists
        if vin_col not in result_df.columns:
            # Try to find a column that might contain VIN information
            potential_cols = [col for col in result_df.columns if 'vin' in col.lower()]
            if potential_cols:
                vin_col = potential_cols[0]
            else:
                # If no VIN column found, return with empty flag column
                result_df[flag_column] = False
                return result_df, 0
        
        # Count occurrences of each VIN
        vin_counts = result_df[vin_col].value_counts()
        
        # Flag rows with duplicate VINs
        result_df[flag_column] = result_df[vin_col].map(lambda x: vin_counts.get(x, 0) > 1)
    
    elif rule.id == "missing_vin":
        # Get the correct column name for VIN
        vin_col = column_mapping.get("vin", "VIN")
        
        # Check if the column exists
        if vin_col not in result_df.columns:
            # Try to find a column that might contain VIN information
            potential_cols = [col for col in result_df.columns if 'vin' in col.lower()]
            if potential_cols:
                vin_col = potential_cols[0]
            else:
                # If no VIN column found, return with empty flag column
                result_df[flag_column] = False
                return result_df, 0
        
        # Flag rows with missing or invalid VINs (basic check for length and pattern)
        result_df[flag_column] = (
            result_df[vin_col].isna() | 
            (result_df[vin_col] == '') | 
            (result_df[vin_col].astype(str).str.strip() == '') |
            (~result_df[vin_col].astype(str).str.match(r'^[A-HJ-NPR-Z0-9]{17}'))
        )
    
    elif rule.id == "low_gross":
        # Get the correct column name for gross profit
        gross_col = column_mapping.get("gross_profit", "Gross_Profit")
        
        # Check if the column exists
        if gross_col not in result_df.columns:
            # Try to find a column that might contain gross information
            potential_cols = [col for col in result_df.columns if 'gross' in col.lower()]
            if potential_cols:
                gross_col = potential_cols[0]
            else:
                # If no gross column found, return with empty flag column
                result_df[flag_column] = False
                return result_df, 0
        
        # Ensure the gross column is numeric
        if not pd.api.types.is_numeric_dtype(result_df[gross_col]):
            # Try to convert to numeric, coercing errors to NaN
            result_df[gross_col] = pd.to_numeric(result_df[gross_col], errors='coerce')
        
        # Apply threshold if specified, default to $500
        threshold = rule.threshold_value if rule.threshold_value is not None else 500
        
        # Flag rows with gross profit below threshold (but not negative)
        result_df[flag_column] = (result_df[gross_col] < threshold) & (result_df[gross_col] >= 0)
    
    elif rule.id == "incomplete_sale":
        # Check for missing values in critical columns
        fields_to_check = {}
        
        for std_col, df_col in column_mapping.items():
            if df_col in result_df.columns:
                fields_to_check[std_col] = df_col
        
        if not fields_to_check:
            result_df[flag_column] = False
            return result_df, 0
        
        # Flag rows with any critical field missing
        result_df[flag_column] = False
        
        for std_col, df_col in fields_to_check.items():
            result_df[flag_column] = result_df[flag_column] | (
                result_df[df_col].isna() | 
                (result_df[df_col] == '') | 
                (result_df[df_col].astype(str).str.strip() == '')
            )
    
    elif rule.id == "anomalous_price":
        # Get the correct column names
        price_col = column_mapping.get("sale_price", "Sale_Price")
        model_col = column_mapping.get("model", "Model")
        
        # Check if the columns exist
        if price_col not in result_df.columns or model_col not in result_df.columns:
            result_df[flag_column] = False
            return result_df, 0
        
        # Ensure price column is numeric
        if not pd.api.types.is_numeric_dtype(result_df[price_col]):
            result_df[price_col] = pd.to_numeric(result_df[price_col], errors='coerce')
        
        # Calculate price statistics per model
        def is_price_anomalous(group):
            if len(group) <= 1:
                return pd.Series([False] * len(group))
            
            mean = group[price_col].mean()
            std = group[price_col].std()
            
            # If std is too small, use a percentage of the mean
            if std < mean * 0.01:
                std = mean * 0.1
            
            threshold = rule.threshold_value if rule.threshold_value is not None else 2.0
            return ((group[price_col] - mean).abs() / std) > threshold
        
        # Apply the detection logic grouped by model
        result_df[flag_column] = False
        for model, group in result_df.groupby(model_col):
            mask = result_df[model_col] == model
            result_df.loc[mask, flag_column] = is_price_anomalous(group)
    
    elif rule.id == "invalid_date":
        # Get the correct column name for sale date
        date_col = column_mapping.get("sale_date", "Sale_Date")
        
        # Check if the column exists
        if date_col not in result_df.columns:
            potential_cols = [col for col in result_df.columns if 'date' in col.lower()]
            if potential_cols:
                date_col = potential_cols[0]
            else:
                result_df[flag_column] = False
                return result_df, 0
        
        # Try to convert to datetime
        date_series = pd.to_datetime(result_df[date_col], errors='coerce')
        
        # Flag invalid dates (NaT) or future dates
        today = pd.Timestamp(datetime.datetime.now())
        result_df[flag_column] = date_series.isna() | (date_series > today)
    
    elif rule.id == "missing_salesperson":
        # Get the correct column name for salesperson
        salesperson_col = column_mapping.get("salesperson", "Salesperson")
        
        # Check if the column exists
        if salesperson_col not in result_df.columns:
            potential_cols = [col for col in result_df.columns if any(term in col.lower() for term in ['salesperson', 'sales_person', 'sales person', 'agent', 'rep'])]
            if potential_cols:
                salesperson_col = potential_cols[0]
            else:
                result_df[flag_column] = False
                return result_df, 0
        
        # Flag rows with missing salesperson
        result_df[flag_column] = (
            result_df[salesperson_col].isna() | 
            (result_df[salesperson_col] == '') | 
            (result_df[salesperson_col].astype(str).str.strip() == '')
        )
    
    # --- New Automotive Rule Logic ---
    elif rule.id == "mileage_discrepancy":
        mileage_col = column_mapping.get("mileage", "Mileage")
        year_col = column_mapping.get("year", "VehicleYear")
        if mileage_col in result_df.columns and year_col in result_df.columns:
            result_df[mileage_col] = pd.to_numeric(result_df[mileage_col], errors='coerce')
            result_df[year_col] = pd.to_numeric(result_df[year_col], errors='coerce')
            current_year = datetime.datetime.now().year
            
            # Basic Logic: Flag if mileage > 25k * (current_year - vehicle_year + 1) OR < 100 for cars older than 1 year
            # This is a very basic example, more sophisticated logic could be used
            max_expected_mileage = 25000 * (current_year - result_df[year_col] + 1)
            result_df[flag_column] = (
                (result_df[mileage_col] > max_expected_mileage) |
                ((result_df[mileage_col] < 100) & (result_df[year_col] < current_year)) |
                result_df[mileage_col].isna() |
                result_df[year_col].isna()
            )
        else:
            result_df[flag_column] = False # Mark as false if columns missing
            
    elif rule.id == "new_used_logic":
        status_col = column_mapping.get("status", "NewUsedStatus")
        mileage_col = column_mapping.get("mileage", "Mileage")
        if status_col in result_df.columns and mileage_col in result_df.columns:
            result_df[mileage_col] = pd.to_numeric(result_df[mileage_col], errors='coerce')
            mileage_threshold = rule.threshold_value if rule.threshold_value is not None else 100
            
            is_new = result_df[status_col].astype(str).str.lower() == 'new'
            is_used = result_df[status_col].astype(str).str.lower() == 'used'
            
            result_df[flag_column] = (
                (is_new & (result_df[mileage_col] > mileage_threshold)) | # New with high miles
                (is_used & (result_df[mileage_col].isna() | (result_df[mileage_col] <= 0))) # Used with zero/missing miles
            )
        else:
            result_df[flag_column] = False
            
    elif rule.id == "title_issue":
        title_col = column_mapping.get("title_status", "TitleStatus")
        if title_col in result_df.columns:
            problematic_titles = ['salvage', 'flood', 'lemon', 'rebuilt']
            result_df[flag_column] = result_df[title_col].astype(str).str.lower().isin(problematic_titles)
        else:
            result_df[flag_column] = False
            
    elif rule.id == "missing_vehicle_info":
        make_col = column_mapping.get("make", "VehicleMake")
        model_col = column_mapping.get("model", "VehicleModel")
        year_col = column_mapping.get("year", "VehicleYear")
        result_df[flag_column] = False # Start with False
        for col in [make_col, model_col, year_col]:
             if col in result_df.columns:
                 result_df[flag_column] |= (result_df[col].isna() | (result_df[col] == '') | (result_df[col].astype(str).str.strip() == ''))
             else:
                 result_df[flag_column] = True # Flag if a required mapped column doesn't exist
                 break
                 
    elif rule.id == "duplicate_stock_number":
        stock_col = column_mapping.get("stock_number", "VehicleStockNumber")
        vin_col = column_mapping.get("vin", "VehicleVIN") # Need VIN to differentiate
        if stock_col in result_df.columns and vin_col in result_df.columns:
            # Flag rows where the stock number is duplicated *and* has more than one unique VIN associated
            stock_vin_counts = result_df.groupby(stock_col)[vin_col].nunique()
            duplicated_stocks = stock_vin_counts[stock_vin_counts > 1].index
            result_df[flag_column] = result_df[stock_col].isin(duplicated_stocks)
        else:
             result_df[flag_column] = False

    # --- New Finance Rule Logic ---
    elif rule.id == "apr_out_of_range":
        apr_col = column_mapping.get("apr", "APR")
        if apr_col in result_df.columns:
            result_df[apr_col] = pd.to_numeric(result_df[apr_col].astype(str).str.replace('%', '', regex=False), errors='coerce')
            lower_threshold = 0.0 # Example lower bound
            upper_threshold = rule.threshold_value if rule.threshold_value is not None else 25.0
            result_df[flag_column] = (result_df[apr_col] < lower_threshold) | (result_df[apr_col] > upper_threshold)
        else:
             result_df[flag_column] = False
             
    elif rule.id == "loan_term_out_of_range":
        term_col = column_mapping.get("term", "LoanTerm")
        if term_col in result_df.columns:
             result_df[term_col] = pd.to_numeric(result_df[term_col], errors='coerce')
             lower_threshold = 12.0 # Example lower bound
             upper_threshold = rule.threshold_value if rule.threshold_value is not None else 96.0
             result_df[flag_column] = (result_df[term_col] < lower_threshold) | (result_df[term_col] > upper_threshold)
        else:
             result_df[flag_column] = False
             
    elif rule.id == "missing_lender":
        lender_col = column_mapping.get("lender", "LenderName")
        # Assume financing if APR exists and is > 0? Or needs a 'DealType' column?
        # Simple check for now: flag if lender is missing
        if lender_col in result_df.columns:
            result_df[flag_column] = (result_df[lender_col].isna() | (result_df[lender_col] == '') | (result_df[lender_col].astype(str).str.strip() == ''))
        else:
            result_df[flag_column] = False

    # --- New Service Rule Logic ---
    elif rule.id == "warranty_claim_invalid":
        # Basic placeholder logic - needs refinement based on actual warranty terms data
        claim_date_col = column_mapping.get("claim_date", "ClaimDate")
        service_date_col = column_mapping.get("service_date", "ServiceDate") # Assumes warranty starts at service date?
        mileage_col = column_mapping.get("mileage", "MileageAtService")
        # Example: 3 years / 36k miles warranty from service date
        if claim_date_col in result_df.columns and service_date_col in result_df.columns and mileage_col in result_df.columns:
            result_df[claim_date_col] = pd.to_datetime(result_df[claim_date_col], errors='coerce')
            result_df[service_date_col] = pd.to_datetime(result_df[service_date_col], errors='coerce')
            result_df[mileage_col] = pd.to_numeric(result_df[mileage_col], errors='coerce')
            
            warranty_end_date = result_df[service_date_col] + pd.DateOffset(years=3)
            warranty_mileage_limit = 36000
            
            result_df[flag_column] = (
                (result_df[claim_date_col] > warranty_end_date) |
                (result_df[mileage_col] > warranty_mileage_limit) |
                result_df[claim_date_col].isna() |
                result_df[service_date_col].isna() |
                result_df[mileage_col].isna()
            )
        else:
            result_df[flag_column] = False
            
    elif rule.id == "missing_technician":
        tech_col = column_mapping.get("tech_id", "TechnicianID")
        if tech_col in result_df.columns:
             result_df[flag_column] = (result_df[tech_col].isna() | (result_df[tech_col] == '') | (result_df[tech_col].astype(str).str.strip() == ''))
        else:
            result_df[flag_column] = False
            
            
    # Count flagged rows
    # Ensure flag_column exists before summing
    flagged_count = 0
    if flag_column in result_df.columns:
        flagged_count = int(result_df[flag_column].sum())
        print(f"[DEBUG] Rule '{rule.id}' flagged {flagged_count} rows.") # Log count
    else:
        print(f"[WARN] Flag column '{flag_column}' not found after rule '{rule.id}' execution.")
        
    return result_df, flagged_count


def apply_validation_profile(df: pd.DataFrame, profile: ValidationProfile) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply a validation profile to a DataFrame.
    
    Args:
        df: The input DataFrame
        profile: The validation profile to apply
        
    Returns:
        Tuple of (DataFrame with added flag columns, dictionary with flag counts)
    """
    print(f"[DEBUG] Starting validation profile application for profile: {profile.id}") # Log start
    result_df = df.copy()
    flag_counts = {}
    
    enabled_rules = profile.get_enabled_rules()
    print(f"[DEBUG] Enabled rules found in profile '{profile.id}': {[rule.id for rule in enabled_rules]}") # Log enabled rules
    
    # Apply each enabled rule
    print("[DEBUG] Entering loop to apply enabled rules...") # Log before loop
    for rule in enabled_rules: # Use the variable
        result_df, count = apply_validation_rule(result_df, rule)
        flag_counts[rule.id] = count
    
    print("[DEBUG] Finished applying all enabled rules.") # Log after loop
    return result_df, flag_counts


def render_profile_editor(profiles_dir: str, on_profile_change: Optional[Callable[[ValidationProfile], None]] = None) -> ValidationProfile:
    """
    Render a Streamlit UI for editing validation profiles.
    
    Args:
        profiles_dir: Directory containing profile JSON files
        on_profile_change: Optional callback when profile is changed
        
    Returns:
        The currently selected ValidationProfile
    """
    # Load available profiles
    profiles = get_available_profiles(profiles_dir)
    
    # Initialize session state if not already done
    if 'current_profile' not in st.session_state:
        # Find default profile
        default_profile = next((p for p in profiles if p.is_default), profiles[0] if profiles else None)
        st.session_state.current_profile = default_profile
    
    # Create a container for the profile selection
    profile_container = st.container()
    
    with profile_container:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Profile selector dropdown
            profile_names = [p.name for p in profiles]
            current_profile_name = st.session_state.current_profile.name if st.session_state.current_profile else ""
            selected_profile_name = st.selectbox(
                "Select Validation Profile",
                profile_names,
                index=profile_names.index(current_profile_name) if current_profile_name in profile_names else 0
            )
            
            # Update current profile based on selection
            if selected_profile_name != current_profile_name:
                selected_profile = next((p for p in profiles if p.name == selected_profile_name), None)
                if selected_profile:
                    st.session_state.current_profile = selected_profile
                    if on_profile_change:
                        on_profile_change(selected_profile)
        
        with col2:
            # Button to create a new profile
            if st.button("New Profile"):
                with st.form("new_profile_form"):
                    st.subheader("Create New Profile")
                    profile_name = st.text_input("Profile Name", value="My Custom Profile")
                    profile_desc = st.text_area("Description", value="Custom validation profile")
                    
                    # Select which existing profile to base on
                    base_profile_name = st.selectbox(
                        "Base on existing profile",
                        profile_names,
                        index=profile_names.index(current_profile_name) if current_profile_name in profile_names else 0
                    )
                    
                    # Get the base profile
                    base_profile = next((p for p in profiles if p.name == base_profile_name), None)
                    
                    # Submit button
                    submitted = st.form_submit_button("Create Profile")
                    
                    if submitted and profile_name and base_profile:
                        # Create a new profile based on the selected one
                        timestamp = datetime.datetime.now().isoformat()
                        new_profile = ValidationProfile(
                            id=str(uuid.uuid4()),
                            name=profile_name,
                            description=profile_desc,
                            created_at=timestamp,
                            updated_at=timestamp,
                            rules=[ValidationRule(**rule.dict()) for rule in base_profile.rules],
                            is_default=False
                        )
                        
                        # Save the new profile
                        new_profile.save(profiles_dir)
                        
                        # Update the profile list and current profile
                        profiles.append(new_profile)
                        st.session_state.current_profile = new_profile
                        
                        # Trigger callback if provided
                        if on_profile_change:
                            on_profile_change(new_profile)
                        
                        st.success(f"Created new profile: {profile_name}")
                        st.experimental_rerun()
        
        with col3:
            # Button to delete the current profile
            if not st.session_state.current_profile.is_default and st.button("Delete Profile"):
                if st.session_state.current_profile:
                    # Confirmation dialog
                    delete_confirmed = st.checkbox(
                        f"Confirm deletion of '{st.session_state.current_profile.name}'",
                        value=False,
                        key="delete_confirm"
                    )
                    
                    if delete_confirmed:
                        # Delete the profile file
                        profile_path = os.path.join(profiles_dir, f"{st.session_state.current_profile.id}.json")
                        if os.path.exists(profile_path):
                            os.remove(profile_path)
                        
                        # Remove from profiles list
                        profiles = [p for p in profiles if p.id != st.session_state.current_profile.id]
                        
                        # Reset current profile to default
                        default_profile = next((p for p in profiles if p.is_default), profiles[0] if profiles else None)
                        st.session_state.current_profile = default_profile
                        
                        # Trigger callback if provided
                        if on_profile_change and default_profile:
                            on_profile_change(default_profile)
                        
                        st.success("Profile deleted successfully")
                        st.experimental_rerun()
    
    # Profile editing section
    if st.session_state.current_profile:
        profile = st.session_state.current_profile
        
        # Show profile description
        st.markdown(f"**Description:** {profile.description}")
        
        # Categorize rules
        categories = {}
        for rule in profile.rules:
            if rule.category not in categories:
                categories[rule.category] = []
            categories[rule.category].append(rule)
        
        # Create tabs for each category
        if categories:
            tabs = st.tabs(list(categories.keys()))
            
            for i, (category, rules) in enumerate(categories.items()):
                with tabs[i]:
                    for rule in rules:
                        with st.expander(f"{rule.name} ({rule.severity})", expanded=False):
                            # Rule description
                            st.markdown(f"**{rule.description}**")
                            
                            # Enable/disable toggle
                            enabled = st.checkbox("Enable this rule", value=rule.enabled, key=f"enable_{rule.id}")
                            if enabled != rule.enabled:
                                rule.enabled = enabled
                                profile.updated_at = datetime.datetime.now().isoformat()
                                profile.save(profiles_dir)
                            
                            # Threshold settings (if applicable)
                            if rule.threshold_value is not None:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Threshold value input
                                    unit_display = f" ({rule.threshold_unit})" if rule.threshold_unit else ""
                                    step_value = 0.1 if rule.threshold_unit == "std" or rule.threshold_unit == "%" else 100
                                    threshold = st.number_input(
                                        f"Threshold Value{unit_display}",
                                        value=float(rule.threshold_value),
                                        step=step_value,
                                        key=f"threshold_{rule.id}"
                                    )
                                    
                                    if threshold != rule.threshold_value:
                                        rule.threshold_value = threshold
                                        profile.updated_at = datetime.datetime.now().isoformat()
                                        profile.save(profiles_dir)
                                
                                with col2:
                                    # Operator selection
                                    if rule.threshold_operator:
                                        operators = ["==", "!=", ">", "<", ">=", "<="]
                                        op_index = operators.index(rule.threshold_operator) if rule.threshold_operator in operators else 0
                                        operator = st.selectbox(
                                            "Operator",
                                            operators,
                                            index=op_index,
                                            key=f"operator_{rule.id}"
                                        )
                                        
                                        if operator != rule.threshold_operator:
                                            rule.threshold_operator = operator
                                            profile.updated_at = datetime.datetime.now().isoformat()
                                            profile.save(profiles_dir)
                            
                            # Column mapping settings
                            if st.checkbox("Show column mappings", key=f"show_mappings_{rule.id}"):
                                st.subheader("Column Mappings")
                                st.markdown("Map standard column names to your actual dataset columns:")
                                
                                for std_col, dataset_col in rule.column_mapping.items():
                                    col_mapping = st.text_input(
                                        f"Standard column '{std_col}'",
                                        value=dataset_col,
                                        key=f"col_map_{rule.id}_{std_col}"
                                    )
                                    
                                    if col_mapping != dataset_col:
                                        rule.column_mapping[std_col] = col_mapping
                                        profile.updated_at = datetime.datetime.now().isoformat()
                                        profile.save(profiles_dir)
    
    return st.session_state.current_profile