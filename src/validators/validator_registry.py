"""
Validator Registry for Watchdog AI.

This module discovers and registers all available validators in the system,
providing a centralized registry for validator management.
"""

import inspect
import logging
from typing import Dict, Any, List, Type, Optional, Set

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
    # Use a set to automatically avoid duplicates
    validator_set: Set[Type[BaseValidator]] = set()
    
    # Add the explicitly imported validators
    validator_set.add(FinancialValidator)
    validator_set.add(CustomerValidator)
    
    # Dynamically discover any additional validators
    # This will find all loaded subclasses of BaseValidator
    for subclass in BaseValidator.__subclasses__():
        validator_set.add(subclass)
    
    # Convert back to a list for backward compatibility
    return list(validator_set)

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
            # Instantiate the validator without data (BaseValidator now has optional data parameter)
            validator = validator_class()
            validators.append(validator)
        except Exception as e:
            logging.error(f"Failed to instantiate validator {validator_class.__name__}: {str(e)}")
    
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
        try:
            # Use get_name() method that is now abstractly defined in BaseValidator
            if validator.get_name() == name:
                return validator
        except Exception as e:
            logging.warning(f"Validator {type(validator).__name__} failed to provide name: {str(e)}")
    
    logging.debug(f"No validator found with name: {name}")
    return None

def get_available_validator_names() -> List[str]:
    """
    Get the names of all available validators.
    
    Returns:
        List of validator names
    """
    validators = get_validators()
    validator_names = []
    
    for validator in validators:
        try:
            validator_names.append(validator.get_name())
        except Exception as e:
            logging.warning(f"Validator {type(validator).__name__} failed to provide name: {str(e)}")
            
    return validator_names
