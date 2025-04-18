"""
Legacy rules definition for Watchdog AI.

This module contains the original rules definition for backward compatibility.
New code should use the modularized rule definitions in the profiles directory.
"""

from typing import List
from .validation_profile import ValidationRule

def create_legacy_rules() -> List[ValidationRule]:
    """Create the original list of default validation rules."""
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
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"title_status": "TitleStatus"}
        ),
        ValidationRule(
            id="missing_vehicle_info",
            name="Missing Key Vehicle Info",
            description="Flags records missing essential vehicle identifiers like Make, Model, or Year.",
            enabled=False, 
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
            enabled=False, 
            severity="Medium",
            category="Inventory",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"stock_number": "VehicleStockNumber", "vin": "VehicleVIN"}
        ),
        
        # --- New Finance Rule Definitions ---
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
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"lender": "LenderName"}
        ),
        
        # --- New Service Rule Definitions ---
        ValidationRule(
            id="warranty_claim_invalid",
            name="Invalid Warranty Claim",
            description="Flags warranty claims submitted after the likely warranty expiration based on date/mileage.",
            enabled=False, 
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
            enabled=False, 
            severity="Low",
            category="Service",
            threshold_value=None,
            threshold_unit=None,
            threshold_operator=None,
            column_mapping={"tech_id": "TechnicianID"}
        )
    ]