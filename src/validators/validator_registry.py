"""
Validator Registry for Watchdog AI.

This module discovers and registers all available validators in the system,
providing a centralized registry for validator management.
"""

import inspect
from typing import Dict, Any, List, Type, Optional

from src.validators.base_validator import BaseValidator
from src.validators.financial_validator import FinancialValidator
from src.validators.customer_validator import CustomerValidator

# Additional validators can be imported here

def get_validator_classes() -> List[Type[BaseValidator]]:
    """
    Get all validator classes that are subclasses of BaseValidator.
    
    Returns:
        List of validator class types
    """
    validators = []
    
    # Add the explicitly imported validators
    validators.append(FinancialValidator)
    validators.append(CustomerValidator)
    
    # Dynamically discover any additional validators
    # This will find all loaded subclasses of BaseValidator
    for subclass in BaseValidator.__subclasses__():
        if subclass not in validators:
            validators.append(subclass)
    
    return validators

def get_validators() -> List[BaseValidator]:
    """
    Get instances of all available validators.
    
    Returns:
        List of validator instances
    """
    validator_classes = get_validator_classes()
    validators = []
    
    for validator_class in validator_classes:
        try:
            # Instantiate the validator
            validator = validator_class()
            validators.append(validator)
        except Exception as e:
            print(f"[ERROR] Failed to instantiate validator {validator_class.__name__}: {str(e)}")
    
    return validators

def get_validator_by_name(name: str) -> Optional[BaseValidator]:
    """
    Get a validator instance by name.
    
    Args:
        name: Name of the validator to get
        
    Returns:
        Validator instance or None if not found
    """
    validators = get_validators()
    
    for validator in validators:
        if validator.get_name() == name:
            return validator
    
    return None

def get_available_validator_names() -> List[str]:
    """
    Get the names of all available validators.
    
    Returns:
        List of validator names
    """
    validators = get_validators()
    return [validator.get_name() for validator in validators]