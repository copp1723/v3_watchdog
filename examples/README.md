# Watchdog AI Examples

This directory contains example scripts demonstrating the usage of the Watchdog AI system.

## Available Examples

### Insight Validator Example

The `insight_validator_example.py` script demonstrates how to use the insight validator module to flag and analyze data quality issues in dealership data. This example:

1. Loads sample dealership data
2. Flags common data quality issues including:
   - Negative gross profit
   - Missing lead sources
   - Duplicate VINs
   - Missing or invalid VINs
3. Generates a summary of detected issues
4. Creates a detailed markdown report with recommendations
5. Produces visualizations of data quality metrics

To run this example:

```bash
cd examples
python insight_validator_example.py
```

This will generate two output files:
- `data_quality_report.md`: A detailed report of data quality issues
- `data_quality_visualizations.png`: Visual representations of the data quality metrics

## Adding New Examples

When adding new examples, please follow these guidelines:

1. Create a descriptive file name (e.g., `feature_name_example.py`)
2. Include a docstring explaining the purpose of the example
3. Add detailed comments explaining key steps
4. Update this README with information about your example
