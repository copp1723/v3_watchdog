# API Documentation

## LLM Column Mapping API

The LLM-driven column mapping API enables semantic mapping of arbitrary dataset column names to a canonical schema using a structured, Jeopardy-style reasoning process. It supports ambiguity resolution through interactive clarifications and helps identify lead source values that appear as column headers.

### Endpoint

```python
from src.llm_engine import LLMEngine

# Initialize the LLM engine
engine = LLMEngine()

# Map columns
result = engine.map_columns_jeopardy(columns)
```

### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | List[str] | A list of column headers from the dataset to be mapped |

Example input:
```python
columns = ["lead_source", "profit", "sold_price", "vehicle_year", "vehicle_make", "vehicle_model", "CarnowCars.com"]
```

### Output

The method returns a structured JSON object with the following sections:

| Field | Type | Description |
|-------|------|-------------|
| `mapping` | Dict | A hierarchical mapping of canonical schema categories to fields with confidence scores |
| `clarifications` | List[Dict] | Questions for the user about ambiguous mappings |
| `unmapped_columns` | List[Dict] | Columns that couldn't be mapped or were identified as lead source values |

#### Response Schema

```json
{
  "mapping": {
    "VehicleInformation": {
      "VIN": {"column": null, "confidence": 0.00},
      "VehicleYear": {"column": "vehicle_year", "confidence": 0.99},
      "VehicleMake": {"column": "vehicle_make", "confidence": 0.99},
      "VehicleModel": {"column": "vehicle_model", "confidence": 0.99}
    },
    "TransactionInformation": {
      "SalePrice": {"column": "sold_price", "confidence": 0.97},
      "TotalGross": {"column": "profit", "confidence": 0.98}
    },
    "SalesProcessInformation": {
      "LeadSource": {"column": "lead_source", "confidence": 1.00}
    }
  },
  "clarifications": [],
  "unmapped_columns": [
    {
      "column": "CarnowCars.com",
      "potential_category": "LeadSource",
      "notes": "This looks like a specific lead-source value, not a header."
    }
  ]
}
```

#### Clarifications Example

When the system is uncertain about a mapping or detects potential ambiguity:

```json
{
  "clarifications": [
    {
      "column": "trade_in_value",
      "question": "Does 'trade_in_value' track dealer profit (TotalGross) or vehicle sale price (SalePrice)?",
      "options": ["TotalGross", "SalePrice"]
    }
  ]
}
```

### Configuration Options

The column mapping behavior can be customized through the following configuration options in `src/utils/config.py`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `MIN_CONFIDENCE_TO_AUTOMAP` | float | 0.7 | Minimum confidence threshold for automatic mapping without user clarification |
| `DROP_UNMAPPED_COLUMNS` | bool | false | Whether to automatically drop columns that can't be mapped to the canonical schema |
| `REDIS_CACHE_ENABLED` | bool | true | Whether to use Redis caching for column mapping results |
| `COLUMN_MAPPING_CACHE_TTL` | int | 86400 | Time-to-live for cached column mappings in seconds (default: 24 hours) |

### Caching

To improve performance and reduce LLM API costs, column mapping results are cached using Redis when enabled. The cache key is generated based on a hash of the sorted column names, ensuring consistent results for the same set of columns regardless of order.

### Error Handling

If an error occurs during column mapping, the system returns a structured error response that maintains the expected output format:

```json
{
  "mapping": {},
  "clarifications": [],
  "unmapped_columns": [
    {"column": "column1", "potential_category": null, "notes": "Error: [error details]"}
  ]
}
```

## Usage in UI Flow

The column mapping API is used during the data upload flow to automatically map user-uploaded data columns to the canonical schema used by Watchdog AI. The process flow is:

1. User uploads a dataset (CSV or Excel)
2. The system extracts column headers
3. Column headers are passed to `map_columns_jeopardy()`
4. If high-confidence mappings are found (≥ 0.7), they are applied automatically
5. If clarifications are needed, the user is prompted to resolve ambiguities
6. Once mappings are finalized, the column names are renamed in the DataFrame
7. If `DROP_UNMAPPED_COLUMNS` is enabled, unmapped columns are removed
8. The mapped data is then available for querying through Chat Analysis