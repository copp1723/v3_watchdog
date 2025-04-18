"""
Validator Registry Module.

This module provides a central registry for all validator types in the system,
allowing dynamic registration, discovery, and instantiation of validators.
"""

from typing import Dict, Type, List, Optional, Any
import logging
from .base_validator import BaseValidator

logger = logging.getLogger(__name__)

class ValidatorRegistry:
    """
    Registry for validator types.
    
    This class maintains a mapping of validator types and provides methods
    to register, retrieve, and instantiate validators.
    """
    
    _registry: Dict[str, Type[BaseValidator]] = {}
    
    @classmethod
    def register(cls, validator_id: str, validator_class: Type[BaseValidator]) -> None:
        """
        Register a validator class with the registry.
        
        Args:
            validator_id: Unique identifier for the validator type
            validator_class: The validator class to register
        """
        if validator_id in cls._registry:
            logger.warning(f"Validator '{validator_id}' already registered. Overwriting.")
        cls._registry[validator_id] = validator_class
        logger.debug(f"Registered validator '{validator_id}'")
    
    @classmethod
    def get_validator_class(cls, validator_id: str) -> Optional[Type[BaseValidator]]:
        """
        Get a validator class by its ID.
        
        Args:
            validator_id: Unique identifier for the validator type
            
        Returns:
            The validator class or None if not found
        """
        return cls._registry.get(validator_id)
    
    @classmethod
    def create_validator(cls, validator_id: str, *args: Any, **kwargs: Any) -> Optional[BaseValidator]:
        """
        Create a validator instance by its ID.
        
        Args:
            validator_id: Unique identifier for the validator type
            *args: Positional arguments to pass to the validator constructor
            **kwargs: Keyword arguments to pass to the validator constructor
            
        Returns:
            A validator instance or None if the validator type is not registered
        """
        validator_class = cls.get_validator_class(validator_id)
        if validator_class:
            return validator_class(*args, **kwargs)
        logger.warning(f"Validator '{validator_id}' not found in registry")
        return None
    
    @classmethod
    def get_available_validators(cls) -> List[str]:
        """
        Get a list of all registered validator IDs.
        
        Returns:
            List of validator IDs
        """
        return list(cls._registry.keys())


# Convenience functions
def register_validator(validator_id: str, validator_class: Type[BaseValidator]) -> None:
    """
    Register a validator class with the registry.
    
    Args:
        validator_id: Unique identifier for the validator type
        validator_class: The validator class to register
    """
    ValidatorRegistry.register(validator_id, validator_class)


def get_validator(validator_id: str, *args: Any, **kwargs: Any) -> Optional[BaseValidator]:
    """
    Create a validator instance by its ID.
    
    Args:
        validator_id: Unique identifier for the validator type
        *args: Positional arguments to pass to the validator constructor
        **kwargs: Keyword arguments to pass to the validator constructor
        
    Returns:
        A validator instance or None if the validator type is not registered
    """
    return ValidatorRegistry.create_validator(validator_id, *args, **kwargs)


def get_available_validators() -> List[str]:
    """
    Get a list of all registered validator IDs.
    
    Returns:
        List of validator IDs
    """
    return ValidatorRegistry.get_available_validators()