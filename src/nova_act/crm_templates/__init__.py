"""
CRM navigation templates package.
"""

from .dealersocket import DEALERSOCKET_TEMPLATE
from .vinsolutions import VINSOLUTIONS_TEMPLATE

# Map vendor names to their templates
TEMPLATES = {
    "DealerSocket": DEALERSOCKET_TEMPLATE,
    "VinSolutions": VINSOLUTIONS_TEMPLATE
}

def get_template(vendor: str) -> dict:
    """
    Get the navigation template for a vendor.
    
    Args:
        vendor: Vendor name
        
    Returns:
        Template dictionary or None if not found
    """
    return TEMPLATES.get(vendor) 