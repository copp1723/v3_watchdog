"""
Nova Act data source utilities.
"""

from typing import Dict, Any, Optional
import pandas as pd
from ...nova_act.core import NovaActClient

async def fetch_sales_data(credentials: Dict[str, str]) -> pd.DataFrame:
    """Fetch sales data from Nova Act."""
    client = NovaActClient()
    result = await client.collect_report("dealersocket", "sales", credentials)
    return pd.DataFrame(result["data"])

async def fetch_inventory_data(credentials: Dict[str, str]) -> pd.DataFrame:
    """Fetch inventory data from Nova Act."""
    client = NovaActClient()
    result = await client.collect_report("dealersocket", "inventory", credentials)
    return pd.DataFrame(result["data"])