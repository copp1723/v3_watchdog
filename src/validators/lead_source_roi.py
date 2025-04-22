"""
Lead Source ROI data model and processing utilities.

This module provides functionality for:
1. Normalizing lead source names
2. Processing lead source ROI metrics (cost, volume, revenue)
3. Calculating and tracking ROI performance
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Define standard lead source names
STANDARD_LEAD_SOURCES = {
    "website": ["website", "web", "web lead", "dealer website", "dealerwebsite", "site"],
    "cargurus": ["cargurus", "car gurus", "cg", "car guru", "carguru"],
    "autotrader": ["autotrader", "auto trader", "at", "auto traders"],
    "cars.com": ["cars.com", "carscom", "cars com", "cars", "carsdotcom"],
    "facebook": ["facebook", "fb", "facebook marketplace", "fbmp", "facebook leads"],
    "google": ["google", "google ads", "gads", "google adwords", "google organic", "google paid"],
    "truecar": ["truecar", "true car", "tc", "true cars"],
    "walk-in": ["walk-in", "walkin", "walk in", "walk-ins", "walk ins", "walk-up", "walk up"],
    "referral": ["referral", "refer", "reference", "referred", "referrals"],
    "phone": ["phone", "phone call", "phone leads", "call", "calls", "phone inquiry"],
    "email": ["email", "e-mail", "emails", "e-mails", "email leads"],
    "craigslist": ["craigslist", "craig's list", "craigs list", "cl"],
    "other": ["other", "misc", "miscellaneous", "unknown", "unspecified"]
}

class LeadSourceNormalizer:
    """Normalizes lead source names for consistent analysis."""
    
    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the normalizer with optional custom mappings.
        
        Args:
            custom_mappings: Optional custom lead source mappings
        """
        self.mappings = STANDARD_LEAD_SOURCES.copy()
        
        # Add custom mappings if provided
        if custom_mappings:
            for standard, variations in custom_mappings.items():
                if standard in self.mappings:
                    # Append to existing standard source
                    self.mappings[standard].extend(variations)
                else:
                    # Create new standard source
                    self.mappings[standard] = variations
        
        # Pre-compile normalization lookup
        self._build_lookup_table()
    
    def _build_lookup_table(self) -> None:
        """Build the normalization lookup table."""
        self.lookup = {}
        
        for standard, variations in self.mappings.items():
            # Add the standard name itself
            self.lookup[self._clean_name(standard)] = standard
            
            # Add all variations
            for variation in variations:
                self.lookup[self._clean_name(variation)] = standard
    
    def _clean_name(self, name: str) -> str:
        """
        Clean a source name for comparison.
        
        Args:
            name: Source name to clean
            
        Returns:
            Cleaned name for matching
        """
        if not isinstance(name, str):
            return ""
            
        # Convert to lowercase
        name = name.lower()
        
        # Remove special characters and extra spaces
        name = re.sub(r'[^a-z0-9]', '', name)
        
        return name
    
    def normalize(self, name: str) -> str:
        """
        Normalize a lead source name.
        
        Args:
            name: Lead source name to normalize
            
        Returns:
            Normalized standard name
        """
        if not isinstance(name, str) or not name.strip():
            return "unknown"
            
        cleaned = self._clean_name(name)
        
        # Look up the cleaned name
        if cleaned in self.lookup:
            return self.lookup[cleaned]
            
        # If no match, try fuzzy matching
        return self._fuzzy_match(cleaned)
    
    def _fuzzy_match(self, cleaned_name: str) -> str:
        """
        Attempt fuzzy matching for unrecognized names.
        
        Args:
            cleaned_name: Cleaned name to match
            
        Returns:
            Best matching standard name or "other"
        """
        best_match = "other"
        best_score = 0
        
        for standard in self.mappings.keys():
            # Simple substring matching
            if cleaned_name in self._clean_name(standard) or self._clean_name(standard) in cleaned_name:
                # Length-based scoring to favor closer matches
                score = len(self._clean_name(standard)) / max(len(cleaned_name), 1)
                
                if score > best_score:
                    best_score = score
                    best_match = standard
        
        # Only return matches above a threshold
        if best_score > 0.5:
            return best_match
        
        return "other"
    
    def normalize_df(self, df: pd.DataFrame, source_col: str, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize lead sources in a DataFrame.
        
        Args:
            df: DataFrame to process
            source_col: Column with source names
            target_col: Column to store normalized names (defaults to source_col)
            
        Returns:
            DataFrame with normalized lead sources
        """
        if target_col is None:
            target_col = source_col
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Normalize each value
        result[target_col] = df[source_col].astype(str).apply(self.normalize)
        
        return result


class LeadSourceROI:
    """Processes and calculates ROI metrics for lead sources."""
    
    def __init__(self, 
                 normalizer: Optional[LeadSourceNormalizer] = None,
                 cost_data_path: Optional[str] = None):
        """
        Initialize the ROI calculator.
        
        Args:
            normalizer: Optional lead source normalizer
            cost_data_path: Optional path to stored cost data
        """
        self.normalizer = normalizer or LeadSourceNormalizer()
        self.cost_data_path = cost_data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "lead_source_costs.json"
        )
        
        # Load or initialize cost data
        self._load_cost_data()
    
    def _load_cost_data(self) -> None:
        """Load stored cost data or initialize if not found."""
        try:
            if os.path.exists(self.cost_data_path):
                with open(self.cost_data_path, 'r') as f:
                    self.cost_data = json.load(f)
            else:
                # Initialize with empty structure
                self.cost_data = {
                    "sources": {},
                    "updated_at": datetime.now().isoformat()
                }
                # Save the initialized structure
                self._save_cost_data()
        except Exception as e:
            logger.error(f"Error loading cost data: {e}")
            # Fall back to empty structure
            self.cost_data = {
                "sources": {},
                "updated_at": datetime.now().isoformat()
            }
    
    def _save_cost_data(self) -> None:
        """Save cost data to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cost_data_path), exist_ok=True)
            
            # Update timestamp
            self.cost_data["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.cost_data_path, 'w') as f:
                json.dump(self.cost_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cost data: {e}")
    
    def update_source_cost(self, source: str, monthly_cost: float) -> None:
        """
        Update the cost for a lead source.
        
        Args:
            source: Name of the lead source
            monthly_cost: Monthly cost in dollars
        """
        # Normalize the source name
        source = self.normalizer.normalize(source)
        
        # Update cost data
        if source not in self.cost_data["sources"]:
            self.cost_data["sources"][source] = {
                "history": []
            }
        
        # Add to history
        self.cost_data["sources"][source]["history"].append({
            "cost": monthly_cost,
            "effective_date": datetime.now().isoformat(),
            "recorded_by": "system"  # This could be updated to include user info
        })
        
        # Update current cost
        self.cost_data["sources"][source]["monthly_cost"] = monthly_cost
        
        # Save changes
        self._save_cost_data()
    
    def get_source_cost(self, source: str) -> float:
        """
        Get the current cost for a lead source.
        
        Args:
            source: Name of the lead source
            
        Returns:
            Monthly cost or 0 if not found
        """
        # Normalize the source name
        source = self.normalizer.normalize(source)
        
        # Return cost if found
        if source in self.cost_data["sources"]:
            return self.cost_data["sources"][source].get("monthly_cost", 0)
        
        return 0
    
    def calculate_roi(self, 
                     revenue: float, 
                     cost: float,
                     include_zero_cost: bool = True) -> Optional[float]:
        """
        Calculate ROI for a given revenue and cost.
        
        Args:
            revenue: Total revenue
            cost: Total cost
            include_zero_cost: Whether to return infinity for zero cost
            
        Returns:
            ROI as a ratio ((Revenue - Cost) / Cost) or None if invalid
        """
        if cost == 0:
            if include_zero_cost:
                return float('inf')  # Infinite ROI for zero cost
            else:
                return None  # No ROI calculation for zero cost
        
        if revenue < 0 or cost < 0:
            return None  # Invalid inputs
        
        # Calculate ROI
        return (revenue - cost) / cost
    
    def process_dataframe(self, 
                         df: pd.DataFrame,
                         source_col: str = "LeadSource",
                         revenue_col: str = "TotalGross",
                         normalize: bool = True,
                         weekly: bool = False) -> pd.DataFrame:
        """
        Process a DataFrame to calculate ROI metrics.
        
        Args:
            df: DataFrame with lead source and revenue data
            source_col: Column with lead source names
            revenue_col: Column with revenue data
            normalize: Whether to normalize lead source names
            weekly: Whether to calculate weekly instead of monthly metrics
            
        Returns:
            DataFrame with ROI metrics by lead source
        """
        if df is None or df.empty or source_col not in df.columns:
            return pd.DataFrame()
        
        # Create a working copy
        work_df = df.copy()
        
        # Normalize lead sources if requested
        if normalize:
            work_df = self.normalizer.normalize_df(work_df, source_col)
        
        # Group by lead source
        grouped = work_df.groupby(source_col).agg({
            revenue_col: ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            source_col, 'TotalRevenue', 'AvgRevenue', 'LeadCount'
        ]
        
        # Add cost data
        grouped['MonthlyCost'] = grouped[source_col].apply(self.get_source_cost)
        
        # Adjust for weekly if requested
        if weekly:
            grouped['WeeklyCost'] = grouped['MonthlyCost'] / 4.33  # Average weeks per month
            cost_col = 'WeeklyCost'
        else:
            cost_col = 'MonthlyCost'
        
        # Calculate cost per lead
        grouped['CostPerLead'] = grouped.apply(
            lambda x: x[cost_col] / x['LeadCount'] if x['LeadCount'] > 0 else 0, 
            axis=1
        )
        
        # Calculate ROI
        grouped['ROI'] = grouped.apply(
            lambda x: self.calculate_roi(x['TotalRevenue'], x[cost_col]),
            axis=1
        )
        
        # Format ROI as percentage
        grouped['ROIPercentage'] = grouped['ROI'].apply(
            lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
        )
        
        # Sort by ROI descending
        return grouped.sort_values('ROI', ascending=False)
    
    def get_roi_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of ROI metrics.
        
        Args:
            df: Processed DataFrame from process_dataframe
            
        Returns:
            Dictionary with summary metrics
        """
        if df is None or df.empty:
            return {
                "error": "No data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate summary metrics
        total_cost = df['MonthlyCost'].sum()
        total_revenue = df['TotalRevenue'].sum()
        total_leads = df['LeadCount'].sum()
        
        # Top and bottom performers
        top_sources = df.nlargest(3, 'ROI')
        bottom_sources = df.nsmallest(3, 'ROI')
        
        # Calculate overall ROI
        overall_roi = self.calculate_roi(total_revenue, total_cost)
        
        return {
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "total_leads": total_leads,
            "overall_roi": overall_roi,
            "overall_roi_percentage": f"{overall_roi*100:.1f}%" if overall_roi is not None else "N/A",
            "top_performers": top_sources[[
                'LeadSource', 'LeadCount', 'TotalRevenue', 'ROIPercentage'
            ]].to_dict(orient='records'),
            "bottom_performers": bottom_sources[[
                'LeadSource', 'LeadCount', 'TotalRevenue', 'ROIPercentage'
            ]].to_dict(orient='records'),
            "average_cost_per_lead": total_cost / total_leads if total_leads > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }


# Generate default schema profile for lead source ROI
def create_lead_source_roi_schema():
    """
    Create a schema profile for lead source ROI data.
    
    Returns:
        Dictionary with schema profile
    """
    return {
        "id": "lead_source_roi",
        "name": "Lead Source ROI Schema",
        "description": "Schema for analyzing lead source ROI",
        "role": "marketing_manager",
        "columns": [
            {
                "name": "LeadSource",
                "display_name": "Lead Source",
                "description": "Source of the lead",
                "data_type": "string",
                "visibility": "public",
                "required": True,
                "aliases": ["lead_source", "source", "traffic_source"],
                "business_rules": [
                    {
                        "type": "required",
                        "message": "Lead source is required for ROI analysis",
                        "severity": "high"
                    }
                ]
            },
            {
                "name": "LeadDate",
                "display_name": "Lead Date",
                "description": "Date the lead was received",
                "data_type": "datetime",
                "visibility": "public",
                "required": True,
                "aliases": ["lead_date", "date", "inquiry_date"],
                "business_rules": [
                    {
                        "type": "comparison",
                        "operator": "<=",
                        "threshold": "today",
                        "message": "Lead date cannot be in the future",
                        "severity": "high"
                    }
                ]
            },
            {
                "name": "LeadCount",
                "display_name": "Lead Count",
                "description": "Number of leads",
                "data_type": "integer",
                "visibility": "public",
                "required": False,
                "default": 1,
                "aliases": ["lead_count", "count"],
                "business_rules": [
                    {
                        "type": "comparison",
                        "operator": ">",
                        "threshold": 0,
                        "message": "Lead count must be positive",
                        "severity": "medium"
                    }
                ]
            },
            {
                "name": "LeadCost",
                "display_name": "Lead Cost",
                "description": "Cost of the lead",
                "data_type": "float",
                "visibility": "public",
                "required": False,
                "aliases": ["lead_cost", "cost", "cpl"],
                "business_rules": [
                    {
                        "type": "comparison",
                        "operator": ">=",
                        "threshold": 0,
                        "message": "Lead cost cannot be negative",
                        "severity": "medium"
                    }
                ]
            },
            {
                "name": "Revenue",
                "display_name": "Revenue",
                "description": "Revenue generated",
                "data_type": "float",
                "visibility": "public",
                "required": False,
                "aliases": ["revenue", "gross", "gross_profit", "total_gross"],
                "business_rules": []
            },
            {
                "name": "Closed",
                "display_name": "Closed Deal",
                "description": "Whether the lead resulted in a closed deal",
                "data_type": "boolean",
                "visibility": "public",
                "required": False,
                "aliases": ["closed", "sold", "is_sold", "is_closed"]
            }
        ],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }