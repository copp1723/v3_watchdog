# Schema Validation and Alias System Implementation Plan

## Phase 1: Core Schema Definition

1. Create DatasetSchema class:
```python
class DatasetSchema:
    def __init__(self):
        self.required_columns = {}
        self.optional_columns = {}
        self.column_types = {}
        self.aliases = {}
```

2. Define schema configuration:
```python
SCHEMA_CONFIG = {
    'required_columns': {
        'gross': {'type': float, 'aliases': ['gross', 'total_gross', 'front_gross']},
        'lead_source': {'type': str, 'aliases': ['lead_source', 'leadsource', 'source']},
        'sale_date': {'type': 'datetime', 'aliases': ['date', 'sale_date', 'transaction_date']}
    },
    'optional_columns': {
        'vin': {'type': str, 'aliases': ['vin', 'vin_number', 'vehicle_id']},
        'sales_rep': {'type': str, 'aliases': ['sales_rep', 'salesperson', 'rep_name']}
    }
}
```

## Phase 2: Validation Implementation

1. Add validation methods:
```python
def validate_column_presence(self, df: pd.DataFrame) -> List[str]:
    """Check for required columns including aliases"""

def validate_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate and attempt to coerce column types"""

def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    """Map alias columns to canonical names"""
```

2. Implement type coercion:
```python
def coerce_column_type(self, series: pd.Series, target_type: Any) -> Tuple[pd.Series, bool]:
    """Attempt to coerce column to target type"""
```

## Phase 3: Error Handling

1. Define custom exceptions:
```python
class SchemaValidationError(Exception):
    def __init__(self, message: str, details: Dict[str, Any]):
        self.message = message
        self.details = details
```

2. Create error collection system:
```python
class ValidationResult:
    def __init__(self):
        self.missing_columns = []
        self.type_errors = {}
        self.coercion_failures = {}
```

## Phase 4: Integration

1. Update existing validators to use new schema system
2. Add caching for validation results
3. Update tests to cover new functionality
4. Document schema configuration format

## Testing Strategy

1. Test core validation:
- Column presence with aliases
- Type validation and coercion
- Error handling

2. Test edge cases:
- Missing required columns
- Invalid data types
- Multiple valid aliases

3. Test integration:
- Full validation pipeline
- Error reporting
- Performance with large datasets

## Migration Plan

1. Create new schema validation module
2. Update existing code gradually
3. Add deprecation warnings for old methods
4. Provide migration guide for users