# Watchdog AI V3

A data validation and insight generation platform designed for automotive dealership data.

## Overview

Watchdog AI analyzes dealership data, identifies data quality issues, and generates business insights. It offers customizable validation profiles, automated data cleaning, and in-depth analytics.

## Features

- **Data Validation System**: Customizable validation profiles with rules for common data issues
- **Interactive UI**: Streamlit-based interface for data uploading, validation, and insight generation
- **Insight Generation**: AI-powered analytics for extracting business insights
- **Data Cleaning**: Automated data cleaning operations based on validation results

## Project Structure

```
watchdog_ai/
├── examples/            # Example datasets and usage examples
├── src/                 # Source code
│   ├── validators/      # Data validation components
│   │   ├── validation_profile.py   # Validation profile system
│   │   ├── insight_validator.py    # Basic validation rules
│   │   └── validator_service.py    # Validator integration service
│   ├── ui/              # User interface components
│   │   └── components/  # Reusable UI components
│   │       ├── data_upload.py      # File upload handling
│   │       └── flag_panel.py       # Data quality display
│   ├── app.py           # Main Streamlit application
│   ├── server.py        # API server
│   ├── insight_flow.py  # Insight generation pipeline
│   └── ...              # Other modules
├── tests/               # Test suite
├── run.sh               # Run script
└── requirements.txt     # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- virtualenv (recommended)

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Use the provided run script:

```
./run.sh
```

Or run directly with Streamlit:

```
streamlit run src/app.py
```

## Validation Profile System

The validation profile system allows you to:

1. **Create custom validation profiles** with specific rules enabled/disabled
2. **Adjust thresholds** for numeric validation rules
3. **Save and load profiles** for reuse across datasets
4. **Auto-clean data** based on validation results

### Example Usage

```python
from watchdog_ai.validators import ValidationProfile, apply_validation_profile

# Create or load a profile
profile = create_default_profile()

# Apply validation to data
validated_df, flag_counts = apply_validation_profile(df, profile)

# Check validation results
if flag_counts["negative_gross"] > 0:
    print(f"Found {flag_counts['negative_gross']} records with negative gross profit")
```

## Insight Generation

After validation and cleaning, the system can generate business insights such as:

- Sales performance metrics
- Inventory analysis
- Marketing channel effectiveness
- Salesperson performance

## License

[Your License] © 2024 Watchdog AI