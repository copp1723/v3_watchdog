# Validation System for Watchdog AI

This directory contains the validation system for Watchdog AI, which provides capabilities for validating data quality and applying data cleaning operations.

## Components

### 1. Validation Profile System

Located in `validation_profile.py`, this system allows users to:
- Define custom validation rules with thresholds and parameters
- Create, load, and save validation profiles
- Apply validation profiles to datasets
- Automatically clean data based on validation results

### 2. Insight Validator

Located in `insight_validator.py`, this component provides:
- Basic validation functions for common data issues
- Summary statistics for validation results
- Markdown report generation for validation results

### 3. Validator Service

Located in `validator_service.py`, this integrates everything into a unified service:
- Process uploaded files through validation
- Manage validation profiles
- Provide UI components for validation
- Streamlined API for application integration

## Usage Examples

### Basic Validation

```python
from watchdog_ai.validators import ValidationProfile, apply_validation_profile

# Get a validation profile
profile = create_default_profile()

# Apply validation to a DataFrame
validated_df, flag_counts = apply_validation_profile(df, profile)

# Get validation summary
from watchdog_ai.validators import summarize_flags
summary = summarize_flags(validated_df)
```

### Using the Validator Service

```python
from watchdog_ai.validators import ValidatorService, process_uploaded_file

# Create a validator service
validator = ValidatorService(profiles_dir="profiles")

# Process an uploaded file
df, summary, validator = process_uploaded_file(uploaded_file, profiles_dir)

# Validate a DataFrame
validated_df, validation_summary = validator.validate_dataframe(df)

# Clean the DataFrame
cleaned_df = validator.auto_clean_dataframe(validated_df)
```

### UI Integration

```python
from watchdog_ai.validators import render_profile_editor, render_data_validation_interface

# Render a profile editor UI
profile = render_profile_editor(profiles_dir)

# Render a complete validation interface
def on_continue(cleaned_df):
    # Do something with the cleaned data
    pass

render_data_validation_interface(df, validator, on_continue)
```

## Development

To extend the validation system:

1. Add new rule types to `ValidationRuleType` in `validation_profile.py`
2. Add rule implementation in `apply_validation_rule` function
3. Update the default rule set in `create_default_rules` function