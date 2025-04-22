# Watchdog AI Development Guide

This guide explains the reorganized project structure and provides instructions for setting up a development environment for Watchdog AI.

## Getting Started

This section provides a quick way to get up and running with the Watchdog AI codebase.

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/watchdog-ai.git
cd watchdog-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,test]"  # Install with development and test dependencies

# Set up environment variables
cp config/env/.env.template config/env/.env.development
# Edit .env.development to add your API keys
```

### Running the Application

The simplest way to run the application is using the provided runner script:

```bash
# Using the runner script
python src/run_app.py

# Or using Streamlit directly
streamlit run src/app.py

# With specific environment settings
ENV=production python src/run_app.py
```

### Running Tests

Watchdog AI uses pytest for testing. Here are common test commands:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_specific_file.py

# Run tests with specific marker (e.g., visual regression tests)
pytest -m visual

# Run tests with coverage report
pytest --cov=src tests/

# Run async tests (for Playwright/UI tests)
pytest tests/ui/
```

### Common Troubleshooting

Here are solutions to common issues you might encounter:

1. **Playwright Browser Installation**
   
   If you see errors related to missing browsers:
   ```bash
   python -m playwright install
   ```

2. **Port Conflicts**
   
   If port 8501 is already in use:
   ```bash
   # Run on a different port
   streamlit run src/app.py --server.port 8502
   ```

3. **Environment Variables Not Loading**
   
   Check that your environment file is properly set up:
   ```bash
   # Verify env file location
   ln -s config/env/.env.development .env
   ```

4. **Redis Connection Errors**
   
   If Redis connection errors occur:
   ```bash
   # Start Redis locally
   docker run -d -p 6379:6379 redis
   
   # Or disable Redis in your .env file
   REDIS_CACHE_ENABLED=false
   ```

5. **API Key Issues**
   
   For development without API keys:
   ```bash
   # Set mock mode in .env
   USE_MOCK=true
   ```

### Development Workflow

1. **Create a new feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   # Run relevant tests
   pytest tests/unit/path/to/specific_tests.py
   ```

3. **Run linting before committing**
   ```bash
   # Run formatting
   black src/
   isort src/
   
   # Run linting
   flake8 src/
   ```

4. **Create a pull request** for code review

---

## Project Structure

The Watchdog AI project has been reorganized into a cleaner, more maintainable structure:

```
watchdog_ai/
├── assets/                  # Static assets like images and icons
├── config/                  # All configuration files
│   ├── data/                # Test data files
│   ├── docker/              # Docker-related configurations
│   ├── env/                 # Environment configurations
│   ├── rules/               # Business rule definitions
│   ├── scripts/             # Utility and execution scripts
│   ├── testing/             # Test configuration
│   └── ui/                  # UI configuration
├── docs/                    # Documentation
│   ├── core/                # Core system documentation
│   ├── deployment/          # Deployment guides
│   ├── features/            # Feature documentation
│   ├── integration/         # Integration documentation
│   ├── testing/             # Testing documentation
│   ├── ui/                  # UI documentation
│   └── archive/             # Historical documentation
├── src/                     # Source code
│   ├── app.py               # Main application entry point
│   ├── run_app.py           # Application runner script
│   ├── watchdog_ai/         # Main package
│   │   ├── analysis/        # Data analysis modules
│   │   ├── llm/             # LLM integration modules
│   │   ├── schema/          # Schema processing modules
│   │   ├── testing/         # Testing utilities
│   │   └── ui/              # UI components
│   └── utils/               # Utility modules
├── tests/                   # Test suite
│   ├── integration/         # Integration tests
│   └── unit/                # Unit tests
├── .git/                    # Git repository
├── .github/                 # GitHub workflows and templates
├── pyproject.toml           # Project metadata and build configuration
├── setup.py                 # Package installation script
└── various symlinks         # For convenience (requirements.txt, Dockerfile, etc.)
```

## Configuration

Configuration files are now centralized in the `config/` directory:

### Environment Configuration

Environment settings are managed through `.env` files in `config/env/`:

* `.env.development` - Development environment configuration
* `.env.production` - Production environment configuration
* `.env.template` - Template with all available configuration options

Key environment variables include:

```
# API Keys
OPENAI_API_KEY=your-openai-key
AGENTOPS_API_KEY=your-agentops-key

# Application Settings
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100
USE_MOCK=false

# Redis Configuration
REDIS_CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Feature Flags
DROP_UNMAPPED_COLUMNS=false

# Development Settings
DEBUG=true
TESTING=false
```

### Business Rules

Business rules are defined in YAML files in `config/rules/`. The main file is `BusinessRuleRegistry.yaml`, which contains validation rules and operational constraints.

### Docker Configuration

Docker-related configurations are in `config/docker/`:
* `Dockerfile` - Container definition
* `docker-compose.yml` - Multi-container orchestration

## Development Setup

### Prerequisites

* Python 3.8 or higher
* Git
* Docker (optional, for containerized development)
* Redis (optional, for caching)

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/watchdog-ai.git
   cd watchdog-ai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp config/env/.env.template config/env/.env.development
   # Edit the file to add your API keys and adjust settings
   ln -s config/env/.env.development .env  # Create symlink for convenience
   ```

5. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Running the Application

Run the application using one of the following methods:

1. **Using Streamlit directly**:
   ```bash
   streamlit run src/app.py
   ```

2. **Using the run script**:
   ```bash
   ./run_app.sh
   ```

3. **Using Docker**:
   ```bash
   docker-compose up --build
   ```

The application will be available at http://localhost:8501.

### Directory-Specific Development

#### Working with UI Components

UI components are located in `src/watchdog_ai/ui/`. To add a new page:

1. Create a new module in `src/watchdog_ai/ui/pages/`
2. Register it in the main application flow

#### LLM Integration

LLM-related code is in `src/watchdog_ai/llm/`. When working with LLM features:

1. Use the prompt templates from `prompt_templates/`
2. Follow the established pattern for API calls and error handling
3. Ensure proper caching for expensive operations

#### Schema Processing

Schema validation and mapping is in `src/watchdog_ai/schema/`. This code handles:

1. Dynamic column mapping from various input formats
2. Schema validation and enforcement
3. Business rule validation

## Testing

The test suite is organized into unit and integration tests:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_specific_feature.py
```

## Documentation

When adding new features, please update the relevant documentation in the `docs/` directory:

1. For new features, add documentation to `docs/features/`
2. For API changes, update `docs/core/`
3. For UI changes, update `docs/ui/`

## Feature Documentation

The following documents provide detailed information about specific features of the Watchdog AI application:

1. **User Interface Features**:
   - [Theme Toggle System](../features/theme-toggle.md) - Documentation for the light/dark mode toggle functionality
   - [Mobile UI Enhancements](../mobile_ui_enhancements.md) - Information about mobile-specific UI optimizations

2. **Testing Infrastructure**:
   - [Visual Regression Testing](../features/visual-regression.md) - Guide to the visual regression testing system
   - [Playwright Test Setup](../features/playwright-setup.md) - Documentation for end-to-end testing with Playwright

3. **System Components**:
   - [Fallback Renderer](../fallback_renderer.md) - Documentation for the fallback rendering system
   - [Insights Configuration](../insights_configuration.md) - Guide to configuring the insights system

## Contributing

See `docs/core/CONTRIBUTING.md` for guidelines on contributing to the project, including code style, branch naming conventions, and the pull request process.


## Quick Environment Verification
To verify your development environment setup:
```bash
python -m src.utils.verify_setup
```
This will check your Python version, dependencies, configuration files, and environment variables.
