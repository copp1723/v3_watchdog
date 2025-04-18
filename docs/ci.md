# CI Pipeline Documentation

## Purpose

This document describes the Continuous Integration (CI) pipeline for the Watchdog AI project. It aims to help developers understand its components and how to extend it.

## Overview

The CI workflow automates essential checks for every push and pull request to the main branches. It currently includes:

*   **Linting:** Checks code style and formatting (Placeholder - Flake8/Black can be added here).
*   **Unit Tests:** Runs the test suite using `pytest`.
*   **Dependency Audit:** Scans dependencies for known vulnerabilities using `pip-audit`.
*   **End-to-End (E2E) Tests:** Runs tests simulating user interaction with the application (Placeholder - can be added later).

## Status Badges

[![CI Pipeline Status](https://github.com/YOUR_REPO_URL/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_REPO_URL/actions/workflows/ci.yml) <!-- Overall CI Status -->
[![E2E Tests](https://img.shields.io/badge/E2E%20Tests-Placeholder-lightgrey)](https://github.com/YOUR_REPO_URL/actions/workflows/ci.yml) <!-- Replace with actual badge URL/logic -->
[![Security Audit](https://img.shields.io/badge/Security%20Audit-Placeholder-lightgrey)](https://github.com/YOUR_REPO_URL/actions/workflows/ci.yml) <!-- Replace with actual badge URL/logic -->
[![Coverage](https://img.shields.io/badge/Coverage-Placeholder-lightgrey)](https://github.com/YOUR_REPO_URL/actions/workflows/ci.yml) <!-- Replace with actual badge URL/logic -->

*(Replace `YOUR_REPO_URL` with the actual repository URL. You might need specific actions or services to generate dynamic badges for E2E, Security, and Coverage.)*

*   **Passing (Green):** All checks passed successfully.
*   **Failing (Red):** One or more checks failed. Click the badge to see details in the GitHub Actions tab.
*   **Running (Yellow):** The workflow is currently in progress.

## Workflow File Location

The CI workflow is defined in the following file:

*   **Path:** `.github/workflows/ci.yml`

**Workflow Summary:**

3.  **Install Dependencies:** Installs project requirements (`requirements.txt`) and testing tools (`pytest`, `pip-audit`, `coverage`). it's created. Example steps below)*

5.  **Run Unit Tests & Coverage:** Executes `coverage run -m pytest tests/unit`.
6.  **Generate Coverage Report:** Runs `coverage report` (and optionally `coverage xml`).
7.  **Run E2E Tests:** Executes `pytest tests/e2e/test_full_flow.py`.
8.  **Audit Dependencies:** Runs `pip-audit --fail-on high`.esting tools.
4.  **Lint Code:** Runs linters like Flake8 (if configured).
5.  **Run Unit Tests:** Executes `pytest`.
6.  **Audit Dependencies:** Runs `pip-audit`.
7.  **Run E2E Tests:** Executes end-to-end tests (if configured).

*   `pytest` (for running unit and E2E tests)

*   `coverage` (for code coverage measurement)sted in `requirements.txt`. Additionally, it installs development/testing dependencies directly:

*   `pytest` (for running unit tests)
*   `pip-audit` (for security vulnerability scanning)
*   *(Add others like `flake8` if used)*

## Running Locally

You can run the same checks performed by the CI pipeline locally before pushing your changes:

*   **Unit Tests:**
    ```bash
    pytest
    ```
*   **Dependency Audit:**
    ```bash
    pip-audit --fail-on high
    ```
*   **Linting (Example with Flake8):**
    ```bash
    # flake8 src/ tests/ ui/  (Adjust paths as needed)
    ```
*   **E2E Tests (Example):**
    ```bash
    # python run_e2e_tests.py (Adjust command as needed)
    ```

## Extending the Pipeline

To add new checks or steps (e.g., security scans, build steps, notifications):

1.  Edit the `.github/workflows/ci.yml` file.
2.  Add new jobs or new steps within existing jobs.
3.  Ensure new jobs have appropriate dependencies (`needs`) if they rely on previous steps.
4.  Refer to the [GitHub Actions documentation](https://docs.github.com/en/actions) for detailed syntax and available actions.
