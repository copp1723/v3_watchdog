"""
Validators package for Watchdog AI.

This package contains validation components and utilities for 
ensuring data quality and enforcing business rules on datasets.
It includes base validator classes, domain-specific validators,
and a central validator registry.
"""

# Import classes/functions from submodules to make them available
# directly when importing from the 'validators' package.
from .validation_profile import ValidationProfile, ValidationRule, get_available_profiles
from .validator_service import process_uploaded_file
from .base_validator import BaseValidator, BaseRule, BaseProfile
from .registry import ValidatorRegistry, register_validator, get_validator, get_available_validators

# Import domain-specific profiles
try:
    from .profiles import (
        FinancialValidator, 
        InventoryValidator, 
        CustomerValidator,
        create_financial_rules,
        create_inventory_rules,
        create_customer_rules
    )
except ImportError:
    pass  # Domain-specific validators are optional

# Define __all__ to control wildcard imports
__all__ = [
    'ValidationProfile', 
    'ValidationRule',
    'get_available_profiles', 
    'process_uploaded_file',
    'BaseValidator',
    'BaseRule',
    'BaseProfile', 
    'ValidatorRegistry',
    'register_validator',
    'get_validator',
    'get_available_validators'
]
