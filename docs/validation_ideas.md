# Validation Rule Ideas for Watchdog AI (Automotive Focus)

This document outlines potential new validation rules to enhance the Watchdog AI platform, specifically targeting automotive retail, finance & insurance (F&I), and service department data.

## Core Vehicle Data Rules
- **Mileage Discrepancy:** Flag vehicles where mileage seems inconsistent with the reported year (e.g., very high mileage for a new model year, very low for an old one).
- **Odometer Rollback Suspicion:** Flag if mileage decreases between different records for the same VIN.
- **VIN Format/Validity:** Enhance the existing VIN check for more specific pattern errors or check-digit validation (if feasible).
- **Duplicate Stock Number:** Flag records with the same stock number but different VINs.
- **New vs. Used Logic:**
    - Flag "New" vehicles with significant mileage reported.
    - Flag "Used" vehicles with zero or missing mileage.
- **Title Issues:** Flag records indicating problematic title statuses (e.g., Salvage, Flood, Lemon) based on a 'Title Status' column.
- **Missing Key Vehicle Info:** Flag records missing Make, Model, or Year.

## Financial & F&I Rules
- **APR Out of Range:** Flag deals with Annual Percentage Rates (APR) outside a defined typical range (e.g., below 0% or above 25%).
- **Loan Term Out of Range:** Flag deals with loan terms outside typical ranges (e.g., less than 12 months or more than 96 months).
- **Missing F&I Product Attach Rate:** (More complex) Analyze deals to flag salespeople or deals missing expected F&I product attachments based on averages.
- **Inconsistent Deal Numbers:** Flag if Deal Numbers are non-sequential or have unexpected gaps/formats.
- **Cost vs. Selling Price Inconsistency:** Flag deals where the cost is higher than the selling price (unless it's a specific type of deal like a lease return write-down).
- **Missing Lender Information:** Flag financed deals where the lender name/ID is missing.
- **Down Payment Anomaly:** Flag deals with unusually high or zero down payments for financed purchases.

## Service Department Rules
- **Warranty Claim Validity:** Flag warranty claims submitted after the warranty expiration date or mileage limit.
- **Service Interval Mismatch:** Flag services performed significantly earlier or later than recommended intervals based on mileage/time.
- **Recall Status Not Addressed:** Flag vehicles with open recalls that haven't been addressed during a service visit.
- **Duplicate Repair Order:** Flag identical repair orders submitted for the same VIN on the same day.
- **Missing Technician ID:** Flag repair orders where the technician ID is missing.
- **Labor Hours Anomaly:** Flag repair orders where labor hours seem excessively high or low for the type of service performed.

## Sales & Personnel Rules
- **Split Sale Inconsistency:** Flag deals where `SplitSalesRep` is populated but total commission doesn't seem right, or if a split is indicated but only one rep is listed.
- **Salesperson Performance Anomaly:** (More complex) Flag salespeople with significant deviations from average gross profit, volume, or F&I penetration.
- **Missing Deal Date:** Flag deals without a valid transaction date. 