"""
Base Validator for Watchdog AI.

This module defines the abstract base class for validators and common validation logic.
"""

import os
import json
import pandas as pd
import datetime
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from pydantic import BaseModel, Field, validator

class BaseRule(BaseModel):
    """
    Base class for validation rules with common attributes and methods.
    """
    id: str = Field(..., description="Unique identifier for the rule")
    name: str = Field(..., description="Display name for the rule")
    description: str = Field(..., description="Detailed description of what the rule checks")
    enabled: bool = Field(True, description="Whether the rule is enabled")
    severity: str = Field("Medium", description="Severity level (High, Medium, Low)")
    category: str = Field("Data Quality", description="Category for grouping rules")
    
    # Column mapping
    column_mapping: Dict[str, str] = Field(default_factory=dict, description="Mapping of standard column names to dataset column names")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ["High", "Medium", "Low"]
        if v not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseRule":
        """Create a rule from a dictionary."""
        return cls(**data)


class BaseProfile(BaseModel):
    """
    Base class for validation profiles with common attributes and methods.
    """
    id: str = Field(..., description="Unique identifier for the profile")
    name: str = Field(..., description="Display name for the profile")
    description: str = Field("", description="Description of the profile")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    is_default: bool = Field(False, description="Whether this is the default profile")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseProfile":
        """Create a profile from a dictionary."""
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """
        Save the profile to a JSON file.
        
        Args:
            directory: Directory to save the profile to
            
        Returns:
            Path to the saved profile file
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return file_path
    
    @classmethod
    def load(cls, file_path: str) -> "BaseProfile":
        """
        Load a profile from a JSON file.
        
        Args:
            file_path: Path to the profile JSON file
            
        Returns:
            Loaded profile object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class BaseValidator(ABC):
    """
    Abstract base class for validators defining common validator interface.
    """
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate the provided DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (validated DataFrame, validation results)
        """
        pass
    
    @abstractmethod
    def get_rules(self) -> List[Any]:
        """
        Get the list of validation rules.
        
        Returns:
            List of validation rules
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the validator.
        
        Returns:
            Name of the validator
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get the description of the validator.
        
        Returns:
            Description of the validator
        """
        pass


class BaseValidatorFactory:
    """
    Factory class for creating validators from profiles.
    """
    @staticmethod
    @abstractmethod
    def create_validator(profile_id: str) -> BaseValidator:
        """
        Create a validator from a profile ID.
        
        Args:
            profile_id: ID of the profile to use
            
        Returns:
            Validator instance
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_available_validators() -> List[str]:
        """
        Get the list of available validators.
        
        Returns:
            List of validator IDs
        """
        pass