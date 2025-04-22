"""
Response validation module for LLM engine.
"""

from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validates LLM response structure and content."""
    
    def __init__(self):
        """Initialize the validator with required fields and rules."""
        self.required_fields = ["summary", "value_insights", "actionable_flags", "confidence"]
        self.valid_confidence_levels = ["high", "medium", "low"]
        self.min_summary_length = 10
        self.max_summary_length = 500
        self.min_insights = 1
        self.max_insights = 10
        self.min_flags = 1
        self.max_flags = 5
        
    def validate_response(self, response: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate response structure and content.
        
        Args:
            response: Response dictionary to validate
            
        Returns:
            Tuple containing:
            - is_valid: Boolean indicating if response is valid
            - issues: List of validation issues found
            - validated_response: Cleaned and validated response
        """
        issues = []
        validated = {}
        
        try:
            # Check required fields
            for field in self.required_fields:
                if field not in response:
                    issues.append(f"Missing required field: {field}")
                    
            # Validate summary
            if "summary" in response:
                summary_issues = self._validate_summary(response["summary"])
                issues.extend(summary_issues)
                validated["summary"] = str(response["summary"])
            else:
                validated["summary"] = "No summary available"
                
            # Validate value insights
            insights_issues, validated_insights = self._validate_insights(
                response.get("value_insights", [])
            )
            issues.extend(insights_issues)
            validated["value_insights"] = validated_insights
            
            # Validate actionable flags
            flags_issues, validated_flags = self._validate_flags(
                response.get("actionable_flags", [])
            )
            issues.extend(flags_issues)
            validated["actionable_flags"] = validated_flags
            
            # Validate confidence
            confidence_issues, validated_confidence = self._validate_confidence(
                response.get("confidence", "medium")
            )
            issues.extend(confidence_issues)
            validated["confidence"] = validated_confidence
            
            # Add timestamp if not present
            validated["timestamp"] = response.get("timestamp", datetime.now().isoformat())
            
            # Add any additional fields that passed validation
            additional_fields = self._validate_additional_fields(response)
            validated.update(additional_fields)
            
            return len(issues) == 0, issues, validated
            
        except Exception as e:
            logger.error(f"Error during response validation: {str(e)}")
            issues.append(f"Validation error: {str(e)}")
            return False, issues, self._generate_fallback_response()
    
    def _validate_summary(self, summary: Any) -> List[str]:
        """Validate summary content."""
        issues = []
        
        if not isinstance(summary, str):
            issues.append("Summary must be a string")
            return issues
            
        summary_length = len(summary)
        if summary_length < self.min_summary_length:
            issues.append(
                f"Summary too short ({summary_length} chars). "
                f"Minimum {self.min_summary_length} required."
            )
        elif summary_length > self.max_summary_length:
            issues.append(
                f"Summary too long ({summary_length} chars). "
                f"Maximum {self.max_summary_length} allowed."
            )
            
        return issues
    
    def _validate_insights(self, insights: Any) -> Tuple[List[str], List[str]]:
        """Validate value insights."""
        issues = []
        validated_insights = []
        
        # Convert to list if not already
        if not isinstance(insights, list):
            insights = [str(insights)] if insights else []
        
        # Validate each insight
        for insight in insights:
            if not insight:
                continue
            validated_insights.append(str(insight))
        
        # Check count
        if len(validated_insights) < self.min_insights:
            issues.append(
                f"Too few insights ({len(validated_insights)}). "
                f"Minimum {self.min_insights} required."
            )
        elif len(validated_insights) > self.max_insights:
            issues.append(
                f"Too many insights ({len(validated_insights)}). "
                f"Maximum {self.max_insights} allowed."
            )
            validated_insights = validated_insights[:self.max_insights]
            
        return issues, validated_insights
    
    def _validate_flags(self, flags: Any) -> Tuple[List[str], List[str]]:
        """Validate actionable flags."""
        issues = []
        validated_flags = []
        
        # Convert to list if not already
        if not isinstance(flags, list):
            flags = [str(flags)] if flags else []
        
        # Validate each flag
        for flag in flags:
            if not flag:
                continue
            validated_flags.append(str(flag))
        
        # Check count
        if len(validated_flags) < self.min_flags:
            issues.append(
                f"Too few actionable flags ({len(validated_flags)}). "
                f"Minimum {self.min_flags} required."
            )
        elif len(validated_flags) > self.max_flags:
            issues.append(
                f"Too many actionable flags ({len(validated_flags)}). "
                f"Maximum {self.max_flags} allowed."
            )
            validated_flags = validated_flags[:self.max_flags]
            
        return issues, validated_flags
    
    def _validate_confidence(self, confidence: Any) -> Tuple[List[str], str]:
        """Validate confidence level."""
        issues = []
        validated_confidence = "medium"  # default
        
        if not isinstance(confidence, str):
            confidence = str(confidence)
        
        confidence = confidence.lower()
        if confidence not in self.valid_confidence_levels:
            issues.append(
                f"Invalid confidence level: {confidence}. "
                f"Must be one of {self.valid_confidence_levels}"
            )
        else:
            validated_confidence = confidence
            
        return issues, validated_confidence
    
    def _validate_additional_fields(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean additional fields."""
        additional = {}
        
        for key, value in response.items():
            if key not in self.required_fields and key != "timestamp":
                if isinstance(value, (str, int, float, bool, list, dict)):
                    additional[key] = value
                else:
                    try:
                        # Try to convert to string if not a basic type
                        additional[key] = str(value)
                    except:
                        logger.warning(f"Skipping invalid additional field: {key}")
                        
        return additional
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate a fallback response for validation failures."""
        return {
            "summary": "Error validating response",
            "value_insights": [
                "The system encountered an error while validating the response."
            ],
            "actionable_flags": [
                "Please try regenerating the response"
            ],
            "confidence": "low",
            "timestamp": datetime.now().isoformat()
        }

