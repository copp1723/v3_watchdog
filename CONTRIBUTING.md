# Contributing to Watchdog AI

Thank you for your interest in contributing to the Watchdog AI project! This document provides guidelines and instructions for contributing to the codebase.

## Table of Contents

- [CI/CD Pipeline](#cicd-pipeline)
- [Branch Protection Rules](#branch-protection-rules)
- [Pull Request Workflow](#pull-request-workflow)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)

## CI/CD Pipeline

The Watchdog AI project uses a comprehensive CI/CD pipeline to ensure code quality and security. All pull requests must pass through this pipeline before being merged into the `main` branch.

The pipeline consists of the following jobs, which run in sequence:

### 1. Lint

The lint job runs static code analysis tools to ensure code quality and consistency.

- **Tools**: 
  - `flake8`: Checks for PEP 8 compliance and other linting issues
  - `black`: Verifies code formatting
  - `isort`: Checks import ordering

- **Triggered by**: All PRs and pushes to `main`
- **Required for merging**: Yes

```yaml
# Job snippet from CI configuration
lint:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ['3.10']
  steps:
  - uses: actions/checkout@v4
  - name: Run linting
    run: |
      flake8 .
      black --check .
      isort --check-only .
```

### 2. Safety & Bandit (Parallel Jobs)

These security-focused jobs run in parallel after the lint job completes successfully.

#### Safety

The safety job checks for known vulnerabilities in the project dependencies.

- **Tool**: `safety check`
- **Runs after**: Lint job
- **Required for merging**: Yes

```yaml
safety:
  needs: lint
  runs-on: ubuntu-latest
  # Scans all dependencies for known security vulnerabilities
```

#### Bandit

The bandit job performs static security analysis on the Python code.

- **Tool**: `bandit`
- **Runs after**: Lint job
- **Command**: `bandit -r watchdog_ai -ll -ii`
- **Required for merging**: Yes

```yaml
bandit:
  needs: lint
  runs-on: ubuntu-latest
  # Performs static security analysis of the code
```

### 3. Tests

The tests job runs the full test suite with coverage reporting.

- **Tool**: `pytest` with coverage reporting
- **Runs after**: Both Safety and Bandit jobs
- **Coverage threshold**: 90%
- **Required for merging**: Yes

```yaml
tests:
  needs: [safety, bandit]
  runs-on: ubuntu-latest
  # Runs tests with coverage enforcement
  # Coverage results are uploaded to Codecov
```

### 4. Build

The build job creates and tests the Docker image.

- **Runs after**: Tests job
- **Action**: Builds a Docker image to verify build process
- **Required for merging**: Yes

```yaml
build:
  needs: tests
  runs-on: ubuntu-latest
  # Builds the Docker image as final verification
```

## Branch Protection Rules

The `main` branch in the Watchdog AI repository is protected by the following rules:

### Required Status Checks

All of the following status checks must pass before merging a pull request:

| Status Check | Description |
|--------------|-------------|
| `lint` | Code linting and formatting verification |
| `safety` | Dependency vulnerability scanning |
| `bandit` | Static security code analysis |
| `tests` | Unit and integration tests with coverage requirements (90%) |
| `build` | Docker image building verification |

### Additional Protection Rules

- **Require branch to be up to date before merging**: Ensures your branch contains the latest code from `main`
- **Require pull request reviews**: At least one approved review is required
- **Dismiss stale pull request approvals when new commits are pushed**: Reviewers must re-approve after changes
- **Require linear history**: Maintains a clean commit history without merge commits

## Pull Request Workflow

### Branch Naming Conventions

Name your branches according to these prefixes:

- `feature/` - For new features
- `fix/` - For bug fixes
- `docs/` - For documentation changes
- `test/` - For changes to tests
- `refactor/` - For code refactoring
- `chore/` - For routine tasks, dependency updates, etc.

Example: `feature/add-schema-validation`

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for your commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Examples:
- `feat(schema): add support for nested validation rules`
- `fix(ui): resolve theme toggle issue on mobile devices`
- `docs(readme): update installation instructions`
- `test(integration): add missing test case for data ingestion`

### Pull Request Process

1. **Create a branch** from `main` using the naming convention above
2. **Make your changes** and commit following the commit message guidelines
3. **Run tests locally** to verify your changes work as expected
4. **Push your branch** and create a pull request to `main`
5. **Fill in the PR template** with:
   - Summary of changes
   - Link to related issue(s)
   - Screenshots (if UI changes)
   - Checklist of what was tested
6. **Request review** from appropriate team members
7. **Address feedback** from reviewers
8. **Ensure all CI checks pass**
9. Once approved and all checks pass, your PR can be merged

### Code Review Process

- All pull requests require at least one approved review
- Reviewers should focus on:
  - Code quality and readability
  - Test coverage
  - Security implications
  - Performance considerations
  - Documentation completeness
- Address all review comments before merging
- If changes are requested, make the changes and request re-review

## Development Workflow

For detailed instructions on setting up your development environment, running the application locally, and common troubleshooting, please refer to [DEVELOPMENT.md](docs/core/DEVELOPMENT.md).

The document covers:
- Project structure
- Environment setup
- Running the application
- Testing
- Configuration
- Common issues and solutions

## Code Quality Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions
- Use [Black](https://black.readthedocs.io/) formatting
- Sort imports with [isort](https://pycqa.github.io/isort/)
- Maximum line length: 88 characters (Black default)

### Documentation Standards

- Document all public functions, classes, and methods
- Use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Keep README and other documentation up-to-date with code changes

### Testing Requirements

- Maintain minimum 90% test coverage
- Write unit tests for all new functionality
- Include integration tests for complex features
- Use pytest fixtures and parametrization for clean test code

---

Thank you for contributing to Watchdog AI! Your efforts help make this project better for everyone.

