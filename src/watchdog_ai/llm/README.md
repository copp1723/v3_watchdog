# Watchdog AI LLM Engine

## Overview
The LLM Engine provides an enhanced Language Model interface with comprehensive analysis capabilities, pattern detection, and structured response handling. It is designed to work with the OpenAI API while providing additional analysis and validation features.

## Directory Structure
```
llm/
├── __init__.py              # Package interface
├── engine.py               # Main LLM engine implementation
├── analysis/               # Analysis components
│   ├── patterns/          # Pattern detection
│   │   ├── trend_detector.py
│   │   ├── seasonality_analyzer.py
│   │   ├── anomaly_detector.py
│   │   └── correlation_analyzer.py
│   └── metrics/           # Metrics calculation
│       ├── calculator.py
│       ├── confidence.py
│       └── period_analyzer.py
├── parsing/               # Response parsing
│   ├── response_parser.py
│   ├── validator.py
│   └── formatter.py
└── config/               # Configuration management
    ├── api_config.py
    ├── prompt_config.py
    ├── engine_config.py
    └── prompts/          # System prompts
```

## Features
- Advanced pattern detection and analysis
- Comprehensive metric calculation
- Structured response parsing and validation
- Configurable engine settings
- Robust error handling

## Usage

### Basic Usage
```python
from watchdog_ai.llm import LLMEngine

# Initialize engine
engine = LLMEngine()

# Generate insight
response = engine.generate_insight(
    "Analyze sales trends",
    context={'data': sales_data}
)

print(response['summary'])
```

### With Custom Configuration
```python
from watchdog_ai.llm import LLMEngine, EngineSettings

# Create custom settings
settings = EngineSettings(config_path="custom_config.yml")
settings.update_settings(
    analysis={"pattern_confidence_threshold": 0.1},
    validation={"min_insights": 2}
)

# Initialize engine with custom settings
engine = LLMEngine(settings=settings)
```

### Direct Analysis Functions
```python
from watchdog_ai.llm import (
    detect_trends,
    analyze_seasonality,
    calculate_metrics
)

# Pattern analysis
trends = detect_trends(time_series_data)
seasonality = analyze_seasonality(time_series_data)

# Metric analysis
metrics = calculate_metrics(dataframe, query_terms=['sales', 'revenue'])
```

## Configuration

### API Configuration
The engine uses OpenAI's API and can be configured via:
1. Environment variables (OPENAI_API_KEY, etc.)
2. Configuration file (~/.watchdog_ai/llm_config.json)
3. Direct settings updates

### Analysis Settings
- Pattern confidence threshold
- Minimum data points required
- Maximum anomaly percentage
- Correlation threshold
- Seasonality requirements

### Validation Settings
- Summary length limits
- Minimum/maximum insights
- Required response fields
- Confidence levels

## Error Handling
The engine provides comprehensive error handling:
- API connection issues
- Data validation errors
- Analysis failures
- Response parsing problems

Each error response includes:
- Error description
- Detailed message
- Suggested actions
- Error timestamp

## Development

### Adding New Features
1. Create new analysis components in appropriate subdirectories
2. Update engine.py to integrate new functionality
3. Add configuration options if needed
4. Update tests and documentation

### Testing
Run tests with:
```bash
python -m pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Dependencies
- openai
- pandas
- numpy
- scipy
- pyyaml

## License
This module is part of the Watchdog AI project and is subject to its licensing terms.

