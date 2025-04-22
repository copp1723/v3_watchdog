"""
Direct data processing module for common automotive queries.
"""

import pandas as pd
from typing import Dict, Any, Optional

def _normalize_currency(value: str) -> float:
    """Convert currency string to float."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Remove currency symbols and commas, convert to float
        cleaned = str(value).replace('$', '').replace(',', '').strip()
        return float(cleaned)
    except:
        return 0.0

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame by normalizing column names and data."""
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Normalize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Look for lead source column
    lead_source_cols = [col for col in df.columns if 'lead' in col.lower() and 'source' in col.lower()]
    if lead_source_cols:
        df['lead_source'] = df[lead_source_cols[0]]
    
    return df

def clean_lead_source_name(name: str) -> str:
    """Clean and standardize lead source names."""
    if pd.isna(name):
        return "Unknown"
    
    name = str(name).lower().strip()
    
    # Common variations of CarGurus
    if any(variant in name for variant in ['car gurus', 'cargurus', 'car guru']):
        return "CarGurus"
    
    return name.title()

def extract_lead_source_from_prompt(prompt: str) -> Optional[str]:
    """Extract lead source name from prompt."""
    prompt = prompt.lower()
    
    # Look for CarGurus variations
    if any(variant in prompt for variant in ['car gurus', 'cargurus', 'car guru']):
        return "CarGurus"
    
    return None

def process_generic_lead_source_query(df: pd.DataFrame, lead_source: str) -> Dict[str, Any]:
    """Process query about a specific lead source."""
    df = _prepare_dataframe(df)
    
    # Find lead source column
    lead_source_cols = [col for col in df.columns if 'lead' in col.lower() and 'source' in col.lower()]
    if not lead_source_cols:
        return {
            "summary": "Could not find lead source column in data",
            "value_insights": ["No lead source information available"],
            "actionable_flags": ["Add lead source tracking to improve analysis"],
            "confidence": "low"
        }
    
    lead_source_col = lead_source_cols[0]
    
    # Clean lead sources
    df['clean_lead_source'] = df[lead_source_col].apply(clean_lead_source_name)
    
    # Filter for the requested lead source
    source_data = df[df['clean_lead_source'] == lead_source]
    total_deals = len(source_data)
    
    if total_deals == 0:
        return {
            "summary": f"No deals found from {lead_source}",
            "value_insights": [f"No transactions recorded from {lead_source}"],
            "actionable_flags": [f"Verify {lead_source} integration is working correctly"],
            "confidence": "high"
        }
    
    # Calculate metrics
    total_all_deals = len(df)
    source_percentage = (total_deals / total_all_deals) * 100
    
    # Calculate gross if available
    gross_cols = [col for col in df.columns if 'gross' in col.lower()]
    if gross_cols:
        gross_col = gross_cols[0]
        total_gross = source_data[gross_col].apply(_normalize_currency).sum()
        avg_gross = total_gross / total_deals if total_deals > 0 else 0
        
        return {
            "summary": f"{lead_source} generated {total_deals} deals ({source_percentage:.1f}% of total) with ${total_gross:,.2f} total gross",
            "value_insights": [
                f"Total deals: {total_deals}",
                f"Percentage of all deals: {source_percentage:.1f}%",
                f"Total gross: ${total_gross:,.2f}",
                f"Average gross per deal: ${avg_gross:,.2f}"
            ],
            "actionable_flags": [
                f"Monitor {lead_source} performance trends",
                "Compare ROI with other lead sources"
            ],
            "confidence": "high",
            "metrics": {
                "total_deals": total_deals,
                "total_gross": total_gross,
                "avg_gross": avg_gross,
                "source_percentage": source_percentage
            }
        }
    else:
        return {
            "summary": f"{lead_source} generated {total_deals} deals ({source_percentage:.1f}% of total)",
            "value_insights": [
                f"Total deals: {total_deals}",
                f"Percentage of all deals: {source_percentage:.1f}%"
            ],
            "actionable_flags": [
                f"Monitor {lead_source} performance trends",
                "Add gross profit tracking for better ROI analysis"
            ],
            "confidence": "medium",
            "metrics": {
                "total_deals": total_deals,
                "source_percentage": source_percentage
            }
        }

def format_metric_output(metric: float, prefix: str = "$") -> str:
    """Format numeric output with appropriate prefix and commas."""
    if prefix == "$":
        return f"${metric:,.2f}"
    return f"{metric:,.2f}"