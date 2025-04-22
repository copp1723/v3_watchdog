# src/validators/base_validator.py
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
class BaseValidator(ABC):
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
    
    @abstractmethod
    def validate(self):
        """Perform validation on the data."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the validator."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return the description of the validator."""
        pass

class BaseRule:
    """
    Base class for validation rules.
    
    Rules define criteria for validating data and identifying issues.
    """
    
    def __init__(self, 
                 id: str = "", 
                 name: str = "", 
                 description: str = "",
                 enabled: bool = True,
                 severity: str = "Medium",
                 category: str = "General",
                 column_mapping: dict = None,
                 threshold_value: float = None,
                 threshold_operator: str = None):
        """
        Initialize a validation rule.
        
        Args:
            id: Unique identifier for the rule
            name: Human-readable name for the rule
            description: Detailed description of what the rule validates
            enabled: Whether the rule is active
            severity: Severity level of rule violations (e.g., "Low", "Medium", "High")
            category: Category of the rule (e.g., "Finance", "Customer")
            column_mapping: Dictionary mapping canonical column names to DataFrame column names
            threshold_value: Numeric threshold for comparison rules
            threshold_operator: Operator for comparison rules (e.g., "<", ">", "==")
        """
        self.id = id
        self.name = name
        self.description = description
        self.enabled = enabled
        self.severity = severity
        self.category = category
        self.column_mapping = column_mapping or {}
        self.threshold_value = threshold_value
        self.threshold_operator = threshold_operator
    
    def apply(self, value):
        """
        Apply the rule to a value.
        
        Args:
            value: Value to validate
            
        Returns:
            Result of validation (implementation-dependent)
        """
        raise NotImplementedError("Subclasses must implement apply().")

class ValidationError(Exception):
    pass