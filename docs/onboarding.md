# V3 Watchdog AI Developer Onboarding Guide

Welcome to the V3 Watchdog AI team! This guide will help you get started with the codebase, understand the architecture, and learn how to contribute effectively.

## System Architecture

The V3 Watchdog AI platform consists of several key components that work together to provide insights and analytics for automotive dealerships:

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│                     │      │                     │      │                     │
│  Web UI Layer       │      │  Business Logic     │      │  Data Processing    │
│                     │      │                     │      │                     │
│  - Streamlit UI     │◄────►│  - Insight Engine   │◄────►│  - Data Validation  │
│  - React Components │      │  - Scheduler        │      │  - ETL Pipeline     │
│  - UI Components    │      │  - Notification     │      │  - Forecasting      │
│                     │      │  - Session Mgmt     │      │  - Nova ACT         │
│                     │      │                     │      │                     │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          ▲                            ▲                            ▲
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         Shared Infrastructure                               │
│                                                                             │
│  - Redis (Caching, Session Management, Task Queue)                          │
│  - PostgreSQL (User Data, Configuration, Audit Logs)                        │
│  - S3/Object Storage (Report Storage, Model Storage)                        │
│  - LLM Services (OpenAI, Anthropic)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Web UI Layer**
   - Streamlit-based web interface
   - React components for advanced UI elements
   - Mobile-responsive design

2. **Business Logic Layer**
   - Insight generation engine
   - Report scheduler
   - Notification services
   - Session management
   - User authentication

3. **Data Processing Layer**
   - Data validation and normalization
   - ETL pipeline for DMS integration
   - Time-series forecasting
   - Nova ACT integration

4. **Shared Infrastructure**
   - Redis for caching and task queues
   - PostgreSQL for structured data storage
   - S3-compatible storage for reports and models
   - LLM integration for natural language processing

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized development)
- Git
- Access to the V3 Watchdog AI repository

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourdealership/v3watchdog_ai.git
   cd v3watchdog_ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   python setup_env.py --init-db
   ```

### Running Locally

1. Start the development server:
   ```bash
   ./run.sh
   ```

2. Access the web interface:
   - Open your browser and navigate to `http://localhost:8501`

3. Optional: Run with Docker:
   ```bash
   docker-compose up -d
   ```

### Test Suite Commands

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_validators.py
```

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/XXX` - Feature branches
- `bugfix/XXX` - Bug fix branches
- `release/X.Y.Z` - Release preparation branches

### Pull Request Process

1. Create a feature or bugfix branch from `develop`
2. Implement your changes with tests
3. Ensure all tests pass and linting is clean
4. Submit a PR to `develop`
5. Address code review feedback
6. Once approved, merge to `develop`

## Extending the Platform

### Adding a New Validator

Validators ensure data quality and consistency. To add a new validator:

1. Create a new validator class in `src/validators/`:

```python
from .validation_profile import ValidationRule
from src.validators import ValidatorRegistry

class MyNewValidator:
    """
    A validator for checking [specific condition].
    """
    
    def __init__(self):
        self.rule_id = "my_new_validation_rule"
        self.description = "Checks for [specific condition]"
    
    def validate(self, df, config=None):
        """
        Validates the dataframe against the rule.
        
        Args:
            df: The pandas DataFrame to validate
            config: Optional configuration dictionary
            
        Returns:
            dict: Validation results with keys:
                - 'status': 'pass' or 'fail'
                - 'details': Details about the validation
                - 'flags': Issues found
        """
        # Implement your validation logic here
        is_valid = True  # Your validation check
        
        if is_valid:
            return {
                "status": "pass",
                "details": "All data meets [specific condition]",
                "flags": {}
            }
        else:
            return {
                "status": "fail",
                "details": "Data does not meet [specific condition]",
                "flags": {
                    "issue_locations": [...],  # Where issues were found
                    "issue_count": 0  # Number of issues
                }
            }
```

2. Register your validator in `src/validators/validator_service.py`:

```python
from .my_new_validator import MyNewValidator

def initialize_validators():
    # Add your validator to the registry
    registry = ValidatorRegistry()
    registry.register("my_new_validation_rule", MyNewValidator())
    return registry
```

3. Add tests in `tests/unit/validators/test_my_new_validator.py`:

```python
import pytest
import pandas as pd
from src.validators.my_new_validator import MyNewValidator

def test_my_new_validator_valid_data():
    # Create test data that should pass validation
    df = pd.DataFrame({
        "column1": [1, 2, 3],
        "column2": ["A", "B", "C"]
    })
    
    validator = MyNewValidator()
    result = validator.validate(df)
    
    assert result["status"] == "pass"
    assert "issue_count" not in result["flags"]

def test_my_new_validator_invalid_data():
    # Create test data that should fail validation
    df = pd.DataFrame({
        "column1": [1, None, 3],
        "column2": ["A", "B", None]
    })
    
    validator = MyNewValidator()
    result = validator.validate(df)
    
    assert result["status"] == "fail"
    assert result["flags"]["issue_count"] > 0
```

### Adding a New Scheduler Task

The scheduler system handles recurring tasks. To add a new scheduled task:

1. Create a task class in `src/scheduler/tasks/`:

```python
from src.scheduler.base_scheduler import Task

class MyNewTask(Task):
    """
    A task that performs [specific action].
    """
    
    def __init__(self, parameters=None):
        super().__init__(
            task_id="my_new_task",
            name="My New Task",
            description="Performs [specific action]"
        )
        self.parameters = parameters or {}
    
    def run(self, context=None):
        """
        Execute the task.
        
        Args:
            context: Execution context
            
        Returns:
            dict: Task execution results
        """
        # Implement your task logic
        try:
            # Do something useful
            result = self._process_data()
            
            return {
                "status": "success",
                "result": result,
                "message": "Task completed successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Task failed to complete"
            }
    
    def _process_data(self):
        # Implement processing logic
        return {"processed_items": 42}
```

2. Register your task in `src/scheduler/report_scheduler.py`:

```python
from .tasks.my_new_task import MyNewTask

class ReportScheduler:
    def __init__(self):
        # ... existing code ...
        
        # Register tasks
        self.task_registry = {
            "my_new_task": MyNewTask,
            # ... other tasks ...
        }
```

3. Create a test in `tests/unit/scheduler/test_my_new_task.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.scheduler.tasks.my_new_task import MyNewTask

def test_my_new_task_successful_execution():
    task = MyNewTask(parameters={"param1": "value1"})
    
    # Test successful execution
    with patch.object(task, '_process_data', return_value={"processed_items": 42}):
        result = task.run()
        
        assert result["status"] == "success"
        assert result["result"]["processed_items"] == 42

def test_my_new_task_failure():
    task = MyNewTask()
    
    # Test failure handling
    with patch.object(task, '_process_data', side_effect=Exception("Test error")):
        result = task.run()
        
        assert result["status"] == "error"
        assert "Test error" in result["error"]
```

### Adding a New Insight

Insights provide business intelligence to users. To add a new insight:

1. Create a new insight class in `src/insights/`:

```python
from src.insights import InsightBase

class MyNewInsight(InsightBase):
    """
    An insight that analyzes [specific business aspect].
    """
    
    def __init__(self):
        super().__init__("my_new_insight_type")
    
    def _validate_insight_input(self, df, **kwargs):
        """
        Validate input data requirements.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Empty if valid, contains 'error' key if invalid
        """
        required_columns = ["date", "amount", "category"]
        
        for col in required_columns:
            if col not in df.columns:
                return {
                    "error": f"Missing required column: {col}",
                    "status": "error"
                }
        
        return {}
    
    def compute_insight(self, df, **kwargs):
        """
        Compute the insight from data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Insight data
        """
        validation = self._validate_insight_input(df, **kwargs)
        if "error" in validation:
            return validation
        
        # Implement your insight logic
        # For example, calculate monthly trends
        
        return {
            "insight_type": self.insight_type,
            "summary": "Key finding about [business aspect]",
            "metrics": {
                "key_metric_1": 123.45,
                "key_metric_2": 67.89
            },
            "trends": [
                {"period": "Jan", "value": 100},
                {"period": "Feb", "value": 120}
            ],
            "recommendations": [
                "Consider taking action X",
                "Investigate anomaly Y"
            ]
        }
```

2. Register your insight in `src/insights/insight_generator.py`:

```python
from .my_new_insight import MyNewInsight

class InsightGenerator:
    def __init__(self):
        # ... existing code ...
        
        # Register insights
        self.insights = {
            # ... existing insights ...
            "my_new_insight_type": MyNewInsight()
        }
```

3. Add pattern matching for natural language queries:

```python
def generate_insight(self, prompt, df):
    # ... existing code ...
    
    # Add pattern matching for your insight
    if "analyze my business aspect" in prompt.lower() or "show me trend x" in prompt.lower():
        insight_type = "my_new_insight_type"
    
    # ... existing code ...
```

4. Write tests in `tests/unit/insights/test_my_new_insight.py`:

```python
import pytest
import pandas as pd
from src.insights.my_new_insight import MyNewInsight

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=10),
        "amount": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "category": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B"]
    })

def test_my_new_insight_validation(sample_data):
    """Test input validation."""
    insight = MyNewInsight()
    
    # Valid data
    result = insight._validate_insight_input(sample_data)
    assert "error" not in result
    
    # Invalid data (missing column)
    invalid_df = sample_data.drop(columns=["category"])
    result = insight._validate_insight_input(invalid_df)
    assert "error" in result
    assert "Missing required column" in result["error"]

def test_my_new_insight_computation(sample_data):
    """Test insight computation."""
    insight = MyNewInsight()
    result = insight.compute_insight(sample_data)
    
    assert result["insight_type"] == "my_new_insight_type"
    assert "summary" in result
    assert "metrics" in result
    assert "recommendations" in result
```

## Working with Modules

### UI Components

Custom UI components are in `src/ui/components/`. To create a new UI component:

```python
import streamlit as st

def render_my_component(data, config=None):
    """
    Render a custom UI component.
    
    Args:
        data: The data to display
        config: Optional configuration
    
    Returns:
        None
    """
    st.subheader("My Custom Component")
    
    # Implement your UI logic here
    st.write("Data summary:", data.describe())
    
    # Add interactive elements
    if st.button("Show Details"):
        st.write("Detailed view:", data)
```

### Data Processing

For data processing functions, use the utilities in `src/utils/`:

```python
from src.utils.data_normalization import normalize_column_names
from src.utils.data_io import save_processed_data

def process_my_data(df):
    """
    Process data with custom logic.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: Processed data
    """
    # Normalize column names
    df = normalize_column_names(df)
    
    # Apply your processing logic
    df['processed_value'] = df['raw_value'] * 1.5
    
    # Save intermediate results
    save_processed_data(df, "my_processing_step")
    
    return df
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   ```
   ModuleNotFoundError: No module named 'X'
   ```
   Solution: Run `pip install -r requirements.txt` to install all dependencies.

2. **Configuration errors**
   ```
   KeyError: 'MISSING_ENV_VAR'
   ```
   Solution: Ensure all required environment variables are set in your `.env` file.

3. **Database connection issues**
   ```
   OperationalError: could not connect to server
   ```
   Solution: Check database connection settings and ensure the database server is running.

### Getting Help

- Check the `docs/` directory for more documentation
- Look for related issues in the GitHub repository
- Ask questions in the team Slack channel (#v3-watchdog-dev)
- Consult with the tech lead for complex architecture questions

## Additional Resources

- [Project README](../README.md)
- [Architecture Documentation](./infra.md)
- [Deployment Guide](./deployment.md)
- [API Documentation](./api.md)
- [Testing Strategy](./testing.md)

Welcome aboard and happy coding!