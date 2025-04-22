"""
Business Rule Engine for Watchdog AI.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, date

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class RuleValidationResult:
    """Result of a business rule validation."""
    
    def __init__(self, rule_id: str, is_valid: bool, message: str, 
                severity: str = "medium", details: Optional[Dict[str, Any]] = None):
        """Initialize a rule validation result."""
        self.rule_id = rule_id
        self.is_valid = is_valid
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "is_valid": self.is_valid,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleValidationResult':
        """Create from dictionary representation."""
        return cls(
            rule_id=data["rule_id"],
            is_valid=data["is_valid"],
            message=data["message"],
            severity=data.get("severity", "medium"),
            details=data.get("details")
        )

class BusinessRuleEngine:
    """Engine for evaluating business rules against data."""
    
    def __init__(self, rules_file: Optional[str] = None):
        """Initialize the business rule engine with an optional rules file."""
        self.rules = {}
        self.custom_functions = {}
        
        # Register built-in custom functions
        self._register_built_in_functions()
        
        # Load quality thresholds
        self.quality_thresholds = self._load_quality_thresholds()
        
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
    
    def _load_quality_thresholds(self) -> Dict[str, Any]:
        """Load data quality thresholds from config."""
        try:
            with open("config/insight_quality.yml", 'r') as f:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)["thresholds"]
                else:
                    logger.warning("YAML not available, using default thresholds")
        except Exception as e:
            logger.error(f"Error loading quality thresholds: {e}")
        
        # Default thresholds
        return {
            "missing_data": {
                "warning": 10.0,
                "error": 20.0
            },
            "sample_size": {
                "minimum": 30
            },
            "outliers": {
                "max_percent": 5.0
            }
        }
    
    def evaluate_data_quality(self, data: Dict[str, Any]) -> RuleValidationResult:
        """Evaluate data quality rules."""
        # Check missing data
        if "nan_percentage" in data:
            nan_pct = data["nan_percentage"]
            if nan_pct >= self.quality_thresholds["missing_data"]["error"]:
                return RuleValidationResult(
                    rule_id="data_quality_missing",
                    is_valid=False,
                    message=f"High percentage of missing data ({nan_pct:.1f}%)",
                    severity="high",
                    details={"nan_percentage": nan_pct}
                )
            elif nan_pct >= self.quality_thresholds["missing_data"]["warning"]:
                return RuleValidationResult(
                    rule_id="data_quality_missing",
                    is_valid=True,
                    message=f"Warning: {nan_pct:.1f}% missing data",
                    severity="medium",
                    details={"nan_percentage": nan_pct}
                )
        
        # Check sample size
        if "sample_size" in data:
            size = data["sample_size"]
            if size < self.quality_thresholds["sample_size"]["minimum"]:
                return RuleValidationResult(
                    rule_id="data_quality_sample_size",
                    is_valid=False,
                    message=f"Insufficient sample size ({size} < {self.quality_thresholds['sample_size']['minimum']})",
                    severity="high",
                    details={"sample_size": size}
                )
        
        # Check outliers
        if "outlier_percentage" in data:
            outlier_pct = data["outlier_percentage"]
            if outlier_pct > self.quality_thresholds["outliers"]["max_percent"]:
                return RuleValidationResult(
                    rule_id="data_quality_outliers",
                    is_valid=True,
                    message=f"Warning: {outlier_pct:.1f}% outliers detected",
                    severity="medium",
                    details={"outlier_percentage": outlier_pct}
                )
        
        return RuleValidationResult(
            rule_id="data_quality",
            is_valid=True,
            message="Data quality checks passed",
            severity="low",
            details={}
        )

    def _register_built_in_functions(self) -> None:
        """Register built-in custom functions."""
        # Register revenue matching validation
        def validate_revenue(data: Dict[str, Any]) -> Tuple[bool, str]:
            revenue = data.get("revenue", 0)
            units = data.get("units_sold", 0)
            avg_price = data.get("avg_selling_price", 0)
            
            if units <= 0 or avg_price <= 0:
                return True, "Not enough data to validate revenue"
            
            expected_revenue = units * avg_price
            margin = expected_revenue * 0.05  # 5% margin of error
            
            if revenue < (expected_revenue - margin):
                return False, f"Revenue {revenue} is less than expected {expected_revenue-margin}"
            
            return True, "Revenue matches expected value"
            
        self.register_custom_function("validate_revenue", validate_revenue)
    
    def load_rules(self, file_path: str) -> bool:
        """Load rules from a YAML or JSON file."""
        try:
            with open(file_path, 'r') as f:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.yaml' or ext == '.yml':
                    self.rules = yaml.safe_load(f)
                else:  # Assume JSON
                    self.rules = json.load(f)
                    
            logger.info(f"Loaded {len(self.rules)} business rules from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load business rules from {file_path}: {str(e)}")
            return False
    
    def save_rules(self, file_path: str) -> bool:
        """Save rules to a YAML or JSON file."""
        try:
            with open(file_path, 'w') as f:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.yaml' or ext == '.yml':
                    yaml.dump(self.rules, f, default_flow_style=False)
                else:  # Use JSON
                    json.dump(self.rules, f, indent=2)
                    
            logger.info(f"Saved {len(self.rules)} business rules to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save business rules to {file_path}: {str(e)}")
            return False
    
    def add_rule(self, rule_id: str, rule_def: Dict[str, Any]) -> None:
        """Add or update a business rule."""
        self.rules[rule_id] = rule_def
        logger.info(f"Added/updated rule: {rule_id}")
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete a business rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Deleted rule: {rule_id}")
            return True
        
        logger.warning(f"Rule not found for deletion: {rule_id}")
        return False
    
    def get_rules_for_column(self, column_name: str) -> List[Dict[str, Any]]:
        """Get all rules applicable to a specific column."""
        result = []
        for rule_id, rule in self.rules.items():
            # Check direct column match
            if rule.get('column') == column_name:
                rule_copy = rule.copy()
                rule_copy['id'] = rule_id
                result.append(rule_copy)
            
            # Check applies_to list
            applies_to = rule.get('applies_to', [])
            if isinstance(applies_to, list) and column_name in applies_to:
                rule_copy = rule.copy()
                rule_copy['id'] = rule_id
                result.append(rule_copy)
        
        return result
    
    def get_rules_for_role(self, role: str) -> Dict[str, Dict[str, Any]]:
        """Get all rules applicable to a specific executive role."""
        result = {}
        
        for rule_id, rule in self.rules.items():
            # Include rule if:
            # 1. It has no role restrictions, or
            # 2. The specified role is in its allowed roles
            roles = rule.get('roles', [])
            if not roles or role in roles:
                result[rule_id] = rule
        
        return result
    
    def evaluate_rule(self, rule_id: str, data: Dict[str, Any]) -> RuleValidationResult:
        """
        Evaluate a single business rule against data.
        Returns a RuleValidationResult object.
        """
        if rule_id not in self.rules:
            return RuleValidationResult(
                rule_id=rule_id,
                is_valid=False,
                message=f"Rule {rule_id} not found",
                severity="high",
                details={"error": "rule_not_found"}
            )
            
        rule = self.rules[rule_id]
        rule_type = rule.get('type', 'comparison')
        severity = rule.get('severity', 'medium')
        
        try:
            # Check context constraints if present
            context = rule.get('context', {})
            for context_field, expected_value in context.items():
                if context_field in data:
                    actual_value = data[context_field]
                    if actual_value != expected_value:
                        # Rule doesn't apply in this context
                        return RuleValidationResult(
                            rule_id=rule_id,
                            is_valid=True,
                            message=f"Rule {rule_id} skipped (context mismatch)",
                            severity=severity,
                            details={
                                "context_field": context_field,
                                "expected": expected_value,
                                "actual": actual_value
                            }
                        )
            
            if rule_type == 'comparison':
                is_valid, message, details = self._evaluate_comparison_rule(rule, data)
                
            elif rule_type == 'range':
                is_valid, message, details = self._evaluate_range_rule(rule, data)
                
            elif rule_type == 'regexp':
                is_valid, message, details = self._evaluate_regexp_rule(rule, data)
                
            elif rule_type == 'custom':
                is_valid, message, details = self._evaluate_custom_rule(rule, data)
                
            else:
                return RuleValidationResult(
                    rule_id=rule_id,
                    is_valid=False,
                    message=f"Unsupported rule type: {rule_type}",
                    severity=severity,
                    details={"error": "unsupported_rule_type", "type": rule_type}
                )
            
            # Use provided message if validation failed
            if not is_valid and rule.get('message'):
                message = rule['message']
                
            return RuleValidationResult(
                rule_id=rule_id,
                is_valid=is_valid,
                message=message,
                severity=severity,
                details=details
            )
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule_id}: {str(e)}")
            return RuleValidationResult(
                rule_id=rule_id,
                is_valid=False,
                message=f"Error evaluating rule {rule_id}: {str(e)}",
                severity=severity,
                details={"error": "evaluation_error", "exception": str(e)}
            )
    
    def evaluate_all_rules(self, data: Dict[str, Any], role: Optional[str] = None) -> Dict[str, RuleValidationResult]:
        """
        Evaluate all applicable rules against data.
        If role is specified, only rules for that role are evaluated.
        """
        results = {}
        
        # Filter rules by role if specified
        rules_to_evaluate = self.rules
        if role:
            rules_to_evaluate = self.get_rules_for_role(role)
        
        for rule_id in rules_to_evaluate:
            results[rule_id] = self.evaluate_rule(rule_id, data)
        
        return results
    
    def register_custom_function(self, name: str, func: Callable) -> None:
        """Register a custom validation function."""
        self.custom_functions[name] = func
        logger.info(f"Registered custom function: {name}")
    
    def _evaluate_comparison_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate a comparison rule."""
        column = rule.get('column')
        operator = rule.get('operator', '==')
        threshold = rule.get('threshold')
        
        if column not in data:
            return False, f"Column {column} not found in data", {"error": "column_not_found"}
        
        value = data[column]
        details = {"column": column, "value": value, "operator": operator}
        
        # Handle dynamic threshold from another column
        if isinstance(threshold, dict) and 'column' in threshold:
            threshold_column = threshold['column']
            if threshold_column not in data:
                return False, f"Threshold column {threshold_column} not found", {"error": "threshold_column_not_found"}
            
            threshold = data[threshold_column]
            details["threshold_source"] = threshold_column
        
        details["threshold"] = threshold
        
        # Handle special case for TODAY
        if threshold == "TODAY":
            threshold = datetime.now().date()
            details["threshold"] = str(threshold)
        
        # Perform comparison operation
        try:
            result = self._compare_values(value, operator, threshold)
            message = (f"Value {value} {operator} {threshold} is "
                     f"{'valid' if result else 'invalid'}")
            
            return result, message, details
        except Exception as e:
            return False, f"Comparison error: {str(e)}", {"error": "comparison_error", "exception": str(e)}
    
    def _evaluate_range_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate a range rule."""
        column = rule.get('column')
        min_value = rule.get('min_value')
        max_value = rule.get('max_value')
        
        if column not in data:
            return False, f"Column {column} not found in data", {"error": "column_not_found"}
        
        value = data[column]
        details = {"column": column, "value": value, "min_value": min_value, "max_value": max_value}
        
        # Handle special case for TODAY
        if max_value == "TODAY":
            max_value = datetime.now().date()
            details["max_value"] = str(max_value)
        
        # Ensure comparable types
        if min_value is not None:
            if isinstance(value, datetime) and isinstance(min_value, str):
                min_value = datetime.fromisoformat(min_value)
            elif isinstance(value, date) and isinstance(min_value, str):
                min_value = datetime.fromisoformat(min_value).date()
        
        if max_value is not None:
            if isinstance(value, datetime) and isinstance(max_value, str):
                max_value = datetime.fromisoformat(max_value)
            elif isinstance(value, date) and isinstance(max_value, str):
                max_value = datetime.fromisoformat(max_value).date()
        
        # Check range
        in_range = True
        if min_value is not None and value < min_value:
            in_range = False
            message = f"Value {value} is below minimum {min_value}"
        elif max_value is not None and value > max_value:
            in_range = False
            message = f"Value {value} is above maximum {max_value}"
        else:
            message = f"Value {value} is within valid range"
        
        return in_range, message, details
    
    def _evaluate_regexp_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate a regular expression rule."""
        column = rule.get('column')
        pattern = rule.get('pattern')
        
        if column not in data:
            return False, f"Column {column} not found in data", {"error": "column_not_found"}
        
        if not pattern:
            return False, "No pattern specified", {"error": "missing_pattern"}
        
        value = str(data[column])
        details = {"column": column, "value": value, "pattern": pattern}
        
        try:
            matches = bool(re.match(pattern, value))
            message = f"Value '{value}' {'matches' if matches else 'does not match'} pattern"
            return matches, message, details
        except re.error as e:
            return False, f"Regular expression error: {str(e)}", {"error": "regexp_error", "exception": str(e)}
    
    def _evaluate_custom_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate a custom rule."""
        function_name = rule.get('function')
        condition = rule.get('condition')
        
        details = {"rule_type": "custom"}
        
        # Use registered function if specified
        if function_name:
            if function_name not in self.custom_functions:
                return False, f"Custom function {function_name} not found", {"error": "function_not_found"}
            
            try:
                result, message = self.custom_functions[function_name](data)
                details["function"] = function_name
                return result, message, details
            except Exception as e:
                return False, f"Custom function error: {str(e)}", {"error": "function_error", "exception": str(e)}
        
        # Evaluate condition expression
        elif condition:
            # This would require a safe eval implementation
            # For now, just handle a few specific conditions
            if condition == "revenue >= (units_sold * avg_selling_price * 0.95)":
                return self._custom_revenue_match(data)
            
            return False, f"Unsupported custom condition: {condition}", {"error": "unsupported_condition"}
        
        return False, "No custom validation method specified", {"error": "missing_validation_method"}
    
    def _custom_revenue_match(self, data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Custom validation for revenue matching."""
        revenue = data.get("revenue", 0)
        units = data.get("units_sold", 0)
        avg_price = data.get("avg_selling_price", 0)
        
        details = {
            "revenue": revenue,
            "units_sold": units,
            "avg_selling_price": avg_price
        }
        
        if units <= 0 or avg_price <= 0:
            return True, "Not enough data to validate revenue", details
        
        expected_revenue = units * avg_price
        margin = expected_revenue * 0.05  # 5% margin of error
        min_expected = expected_revenue - margin
        
        details["expected_revenue"] = expected_revenue
        details["min_expected"] = min_expected
        
        if revenue < min_expected:
            return False, f"Revenue {revenue} is less than expected {min_expected}", details
        
        return True, "Revenue matches expected value", details
    
    def _compare_values(self, value: Any, operator: str, threshold: Any) -> bool:
        """Compare values using the specified operator."""
        if operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == 'in':
            return value in threshold
        elif operator == 'not in':
            return value not in threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")

# Helper function to create a default rule registry
def create_default_rule_registry() -> Dict[str, Dict[str, Any]]:
    """Create a default business rule registry."""
    return {
        "gross_not_negative": {
            "type": "comparison",
            "column": "total_gross_profit",
            "operator": ">=",
            "threshold": 0,
            "message": "Gross profit should not be negative",
            "severity": "high",
            "applies_to": ["total_gross_profit", "frontend_gross", "backend_gross"]
        },
        "min_gross_per_unit": {
            "type": "comparison",
            "column": "gross_profit_per_unit",
            "operator": ">=",
            "threshold": 500,
            "message": "Gross profit per unit should be at least $500",
            "severity": "medium"
        },
        "min_close_rate": {
            "type": "comparison",
            "column": "close_rate",
            "operator": ">=",
            "threshold": 8,
            "message": "Close rate should be at least 8%",
            "severity": "medium"
        },
        "max_days_in_inventory": {
            "type": "comparison",
            "column": "days_in_inventory",
            "operator": "<=",
            "threshold": 90,
            "message": "Days in inventory should not exceed 90 days",
            "severity": "medium"
        },
        "min_be_penetration": {
            "type": "comparison",
            "column": "be_penetration",
            "operator": ">=",
            "threshold": 70,
            "message": "Back end penetration should be at least 70%",
            "severity": "low"
        },
        "revenue_matching": {
            "type": "custom",
            "condition": "revenue >= (units_sold * avg_selling_price)",
            "message": "Revenue should match or exceed calculated value from units sold",
            "severity": "high"
        },
        # Price validation rule
        "price_below_cost": {
            "type": "comparison",
            "column": "price",
            "operator": ">",
            "threshold": {"column": "cost"},
            "message": "Selling price should be higher than cost",
            "severity": "high"
        },
        # Date range validation
        "sale_date_range": {
            "type": "range",
            "column": "sale_date",
            "min_value": "2020-01-01",
            "max_value": "TODAY",
            "message": "Sale date must be between 2020 and today",
            "severity": "high"
        },
        # String validation
        "valid_vin": {
            "type": "regexp",
            "column": "vin",
            "pattern": "^[A-HJ-NPR-Z0-9]{17}$",
            "message": "VIN must be 17 characters and contain only valid characters",
            "severity": "medium"
        }
    }

# Function to initialize or create rule registry file
def init_rule_registry(file_path: str = "BusinessRuleRegistry.yaml") -> None:
    """Initialize or create a business rule registry file."""
    if os.path.exists(file_path):
        logger.info(f"Rule registry file already exists at {file_path}")
        return
        
    # Create default rules
    rules = create_default_rule_registry()
    
    # Save to file
    try:
        with open(file_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)
            
        logger.info(f"Created default rule registry with {len(rules)} rules at {file_path}")
    except Exception as e:
        logger.error(f"Failed to create rule registry: {str(e)}")

# Main execution for standalone usage
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Create default rule registry
    init_rule_registry()
    
    # Create a rule engine
    engine = BusinessRuleEngine("BusinessRuleRegistry.yaml")
    
    # Test data
    test_data = {
        "total_gross_profit": 5000,
        "frontend_gross": 3000,
        "backend_gross": 2000,
        "gross_profit_per_unit": 600,
        "units_sold": 10,
        "avg_selling_price": 20000,
        "revenue": 200000,
        "sale_date": datetime.now().date(),
        "vin": "1HGCM82633A123456",
        "price": 25000,
        "cost": 20000
    }
    
    # Evaluate all rules
    results = engine.evaluate_all_rules(test_data)
    
    # Display results
    print("\n=== RULE VALIDATION RESULTS ===")
    for rule_id, result in results.items():
        is_valid = "✓" if result.is_valid else "✗"
        print(f"{is_valid} {rule_id}: {result.message} ({result.severity})")
    
    # Test adding a new rule
    engine.add_rule("min_revenue", {
        "type": "comparison",
        "column": "revenue",
        "operator": ">=",
        "threshold": 100000,
        "message": "Revenue should meet minimum target",
        "severity": "medium"
    })
    
    # Save updated rules
    engine.save_rules("BusinessRuleRegistry.yaml")
    
    print("\nRule registry updated. Total rules:", len(engine.rules))