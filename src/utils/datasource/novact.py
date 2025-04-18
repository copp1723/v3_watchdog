"""
Nova Act data source connector for automated data fetching.
"""

import pandas as pd
from datetime import date, datetime, timedelta
import logging
import asyncio
from typing import Dict, Any, Optional

from ...nova_act.core import NovaActClient
from ...nova_act.constants import FileFormat
from ..errors import ProcessingError
from ..data_normalization import normalize_columns

logger = logging.getLogger(__name__)

class NovaActConnector:
    """Connector for fetching data from Nova Act systems."""
    
    def __init__(self, client: Optional[NovaActClient] = None):
        """
        Initialize the connector.
        
        Args:
            client: Optional NovaActClient instance. If not provided, creates new one.
        """
        self.client = client or NovaActClient()
        self._initialize_column_mappings()
    
    def _initialize_column_mappings(self):
        """Initialize column name mappings for different report types."""
        self.column_mappings = {
            "sales": {
                "sale_date": "SaleDate",
                "total_gross": "TotalGross",
                "sales_rep": "SalesRepName",
                "lead_source": "LeadSource",
                "vin": "VIN",
                "sale_price": "SalePrice"
            },
            "inventory": {
                "stock_date": "StockDate",
                "days_in_stock": "DaysInStock",
                "vin": "VIN",
                "make": "Make",
                "model": "Model",
                "year": "Year",
                "list_price": "ListPrice"
            }
        }

async def fetch_sales_data(
    dealership_id: str,
    start_date: date,
    end_date: date,
    client: Optional[NovaActClient] = None
) -> pd.DataFrame:
    """
    Fetch sales data for a dealership within a date range.
    
    Args:
        dealership_id: Unique identifier for the dealership
        start_date: Start date for data fetch
        end_date: End date for data fetch
        client: Optional NovaActClient instance
        
    Returns:
        DataFrame containing sales data
    """
    try:
        # Create or use provided client
        client = client or NovaActClient()
        
        # Configure report parameters
        report_config = {
            "report_type": "sales_performance",
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "format": FileFormat.CSV.value,
            "selectors": {
                "username": "#username",
                "password": "#password",
                "submit": "#login-button",
                "report_menu": "#reports-menu",
                "date_range": "#date-range",
                "download": "#download-csv"
            }
        }
        
        # Collect report
        result = await client.collect_report(
            vendor="dealersocket",
            credentials={"dealership_id": dealership_id},
            report_config=report_config
        )
        
        if not result["success"]:
            raise ProcessingError(
                f"Failed to fetch sales data: {result.get('error', 'Unknown error')}"
            )
        
        # Read CSV file
        df = pd.read_csv(result["file_path"])
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Apply column mappings
        column_mappings = {
            "sale_date": "SaleDate",
            "total_gross": "TotalGross",
            "sales_rep": "SalesRepName",
            "lead_source": "LeadSource",
            "vin": "VIN",
            "sale_price": "SalePrice"
        }
        df = df.rename(columns=column_mappings)
        
        # Convert date columns
        df["SaleDate"] = pd.to_datetime(df["SaleDate"])
        
        # Clean numeric columns
        df["TotalGross"] = pd.to_numeric(
            df["TotalGross"].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )
        df["SalePrice"] = pd.to_numeric(
            df["SalePrice"].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching sales data: {str(e)}")
        raise ProcessingError(f"Failed to fetch sales data: {str(e)}")

async def fetch_inventory_data(
    dealership_id: str,
    client: Optional[NovaActClient] = None
) -> pd.DataFrame:
    """
    Fetch current inventory data for a dealership.
    
    Args:
        dealership_id: Unique identifier for the dealership
        client: Optional NovaActClient instance
        
    Returns:
        DataFrame containing inventory data
    """
    try:
        # Create or use provided client
        client = client or NovaActClient()
        
        # Configure report parameters
        report_config = {
            "report_type": "inventory_report",
            "format": FileFormat.CSV.value,
            "selectors": {
                "username": "#username",
                "password": "#password",
                "submit": "#login-button",
                "report_menu": "#reports-menu",
                "download": "#download-csv"
            }
        }
        
        # Collect report
        result = await client.collect_report(
            vendor="dealersocket",
            credentials={"dealership_id": dealership_id},
            report_config=report_config
        )
        
        if not result["success"]:
            raise ProcessingError(
                f"Failed to fetch inventory data: {result.get('error', 'Unknown error')}"
            )
        
        # Read CSV file
        df = pd.read_csv(result["file_path"])
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Apply column mappings
        column_mappings = {
            "stock_date": "StockDate",
            "days_in_stock": "DaysInStock",
            "vin": "VIN",
            "make": "Make",
            "model": "Model",
            "year": "Year",
            "list_price": "ListPrice"
        }
        df = df.rename(columns=column_mappings)
        
        # Convert date columns
        df["StockDate"] = pd.to_datetime(df["StockDate"])
        
        # Clean numeric columns
        df["ListPrice"] = pd.to_numeric(
            df["ListPrice"].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )
        df["DaysInStock"] = pd.to_numeric(df["DaysInStock"], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching inventory data: {str(e)}")
        raise ProcessingError(f"Failed to fetch inventory data: {str(e)}")