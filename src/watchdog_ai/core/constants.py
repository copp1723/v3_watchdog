"""
Centralized constants for Watchdog AI.
"""
from typing import List, Dict, Set
from watchdog_ai.ui.utils.status_formatter import StatusType

# Validation Thresholds
CONFIDENCE_THRESHOLD = 0.7
DATA_QUALITY_THRESHOLD = 0.8
NAN_WARNING_THRESHOLD = 0.10  # 10%
NAN_SEVERE_THRESHOLD = 0.20   # 20%
MIN_FINDING_LENGTH = 10
MIN_ROWS_DEFAULT = 10

# Action Verbs for Recommendations
VALID_ACTION_VERBS: Set[str] = {
    'Consider',
    'Review',
    'Implement',
    'Analyze',
    'Monitor'
}

# Default Data Schema
DEFAULT_REQUIRED_COLUMNS: List[str] = [
    'SaleDate',
    'LeadSource',
    'TotalGross',
    'SalesPerson',
    'IsSale'
]

DEFAULT_COLUMN_TYPES: Dict[str, str] = {
    'SaleDate': 'datetime64[ns]',
    'TotalGross': 'float64',
    'LeadSource': 'object',
    'SalesPerson': 'object',
    'IsSale': 'int64'
}

# Column Name Mappings - Keys are standardized column names
COLUMN_MAPPINGS = {
    'SaleDate': ['Sale Date', 'DateOfSale', 'Date', 'Date of Sale', 'Transaction Date'],
    'LeadSource': ['Lead Source', 'Source', 'LeadOrigin', 'Origin', 'Channel', 'Source of Lead'],
    'SalesPerson': ['Sales Person', 'SalesRep', 'Sales Rep', 'Representative', 'Salesperson', 'Agent'],
    'TotalGross': ['Gross Profit', 'GrossProfit', 'Profit', 'Total Profit', 'Gross', 'GrossSale'],
    'IsSale': ['Sold', 'Sale', 'IsClosed', 'Is Sale', 'Is Sold', 'Completed']
}

# UI Constants
DEFAULT_THEME = 'dark'
DEFAULT_TAB = 'Insight Engine'
TABS = ['Insight Engine', 'System Connect', 'Settings']
DEFAULT_PREVIOUS_SALES = 25

# Metric Type Indicators
CURRENCY_INDICATORS = {'price', 'cost', 'profit', 'revenue'}
PERCENTAGE_INDICATORS = {'percent', 'percentage', 'rate'}

# Status Types for Error Messages
# The StatusType enum should be used with these messages via the format_status_text utility
ERROR_STATUS_TYPE = StatusType.ERROR
WARNING_STATUS_TYPE = StatusType.WARNING
INFO_STATUS_TYPE = StatusType.INFO
SUCCESS_STATUS_TYPE = StatusType.SUCCESS

# Error Messages (plain text, use with format_status_text)
ERR_NO_DATA = 'No data available for analysis.'
ERR_COLUMN_NOT_FOUND = 'Could not find required column: {}'
ERR_NO_VALID_DATA = 'No valid data found for {} after removing missing values.'
ERR_PROCESSING = 'An error occurred during analysis: {}'

# Streamlit Settings
PAGE_CONFIG = {
    'page_title': 'Watchdog AI',
    'page_icon': 'WD',  # Replaced emoji with text alternative
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

