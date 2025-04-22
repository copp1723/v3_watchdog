#!/usr/bin/env python3
"""
Verification script for the Watchdog AI project.

This script checks the development environment to ensure that:
1. The correct Python version is installed
2. Required dependencies are available
3. Configuration files exist in the correct locations
4. Required environment variables are set
5. Connections to required services work

Usage:
    python -m src.utils.verify_setup

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import os
import sys
import pkg_resources
import platform
import importlib.util
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import subprocess
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("setup_verifier")

# Project root directory
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
except NameError:
    # For running as a script
    PROJECT_ROOT = Path(os.getcwd()).absolute()

# Minimum Python version
MIN_PYTHON_VERSION = (3, 8)

# Core required packages
CORE_REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "openai",
    "pydantic",
    "pyyaml",
    "plotly",
    "altair",
    "redis",
]

# Required configuration files
REQUIRED_CONFIG_FILES = [
    "config/env/.env.template",
    "config/rules/BusinessRuleRegistry.yaml",
    "config/docker/Dockerfile",
]

# Required environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
]

# Optional environment variables with description
OPTIONAL_ENV_VARS = {
    "REDIS_HOST": "Redis server hostname (default: localhost)",
    "REDIS_PORT": "Redis server port (default: 6379)",
    "REDIS_CACHE_ENABLED": "Whether Redis caching is enabled (default: false)",
    "LOG_LEVEL": "Logging level (default: INFO)",
    "MAX_UPLOAD_SIZE_MB": "Maximum upload size in MB (default: 100)",
    "USE_MOCK": "Whether to use mock APIs (default: false)",
}

def check_python_version() -> bool:
    """
    Check if the running Python version meets the minimum requirements.
    """
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        logger.error(f"❌ Python version {'.'.join(map(str, current_version))} is not supported.")
        logger.error(f"   Minimum required version is {'.'.join(map(str, MIN_PYTHON_VERSION))}.")
        return False
    
    logger.info(f"✅ Python version {platform.python_version()} detected.")
    return True

def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    """
    missing_packages = []
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package in CORE_REQUIRED_PACKAGES:
        package_lower = package.lower()
        if package_lower not in installed_packages:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ Missing required packages:")
        for package in missing_packages:
            logger.error(f"   - {package}")
        logger.error("   Install them with: pip install -r config/env/requirements-core.txt")
        return False
    
    logger.info("✅ All core dependencies are installed.")
    return True

def check_config_files() -> bool:
    """
    Check if all required configuration files exist.
    """
    missing_files = []
    
    for config_file in REQUIRED_CONFIG_FILES:
        file_path = PROJECT_ROOT / config_file
        if not file_path.exists():
            missing_files.append(config_file)
    
    if missing_files:
        logger.error("❌ Missing configuration files:")
        for file in missing_files:
            logger.error(f"   - {file}")
        return False
    
    # Check if .env exists in project root or config/env
    env_file = PROJECT_ROOT / ".env"
    env_dev_file = PROJECT_ROOT / "config/env/.env.development"
    env_prod_file = PROJECT_ROOT / "config/env/.env.production"
    
    if not (env_file.exists() or env_dev_file.exists() or env_prod_file.exists()):
        logger.warning("⚠️ No .env file found. You should create one from the template:")
        logger.warning("   cp config/env/.env.template config/env/.env.development")
        logger.warning("   ln -s config/env/.env.development .env")
    else:
        logger.info("✅ .env file is present.")
    
    logger.info("✅ All required configuration files are present.")
    return True

def check_environment_variables() -> bool:
    """
    Check if all required environment variables are set.
    """
    missing_vars = []
    
    # Load environment variables from .env if python-dotenv is installed
    if importlib.util.find_spec("dotenv"):
        from dotenv import load_dotenv
        load_dotenv()
    
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("   Set them in your .env file or environment.")
        return False
    
    # Check optional variables and report status
    logger.info("ℹ️ Checking optional environment variables:")
    for var, description in OPTIONAL_ENV_VARS.items():
        value = os.environ.get(var)
        if value:
            logger.info(f"   ✅ {var} is set to: {value}")
        else:
            logger.info(f"   ⚠️ {var} is not set. {description}")
    
    if not missing_vars:
        logger.info("✅ All required environment variables are set.")
    
    return len(missing_vars) == 0

def check_redis_connection() -> bool:
    """
    Try to connect to Redis if enabled.
    """
    redis_enabled = os.environ.get("REDIS_CACHE_ENABLED", "false").lower() in ("true", "1", "yes")
    
    if not redis_enabled:
        logger.info("ℹ️ Redis caching is disabled, skipping connection check.")
        return True
    
    # Check if redis package is installed
    if importlib.util.find_spec("redis") is None:
        logger.error("❌ Redis is enabled but redis package is not installed.")
        logger.error("   Install it with: pip install redis")
        return False
    
    import redis
    
    # Get Redis connection details
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    
    # Try to connect
    try:
        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=1)
        r.ping()
        logger.info(f"✅ Successfully connected to Redis at {redis_host}:{redis_port}")
        return True
    except redis.exceptions.ConnectionError:
        logger.error(f"❌ Failed to connect to Redis at {redis_host}:{redis_port}")
        logger.error("   Make sure Redis is running and accessible.")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Redis connection: {str(e)}")
        return False

def check_tools_availability() -> bool:
    """
    Check if external tools required for development are available.
    """
    required_tools = {
        "git": "Git version control",
        "docker": "Docker for containerization (optional)",
        "pytest": "Pytest for running tests",
    }
    
    missing_tools = []
    
    for tool, description in required_tools.items():
        try:
            process = subprocess.run(
                ["which", tool], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False
            )
            if process.returncode != 0:
                missing_tools.append((tool, description))
            else:
                logger.info(f"✅ {tool} is available at: {process.stdout.strip()}")
        except Exception:
            missing_tools.append((tool, description))
    
    if missing_tools:
        logger.warning("⚠️ Some development tools are not available:")
        for tool, description in missing_tools:
            logger.warning(f"   - {tool}: {description}")
        
        if any(tool == "docker" for tool, _ in missing_tools):
            logger.info("   Docker is optional but recommended for containerized development.")
        
        return False
    
    return True

def run_all_checks() -> bool:
    """
    Run all verification checks and return overall status.
    """
    logger.info("Starting Watchdog AI environment verification...")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Run all checks
    python_check = check_python_version()
    dependency_check = check_dependencies()
    config_check = check_config_files()
    env_var_check = check_environment_variables()
    redis_check = check_redis_connection()
    tools_check = check_tools_availability()
    
    # Print summary
    logger.info("\n=== Verification Summary ===")
    logger.info(f"Python version: {'✅ PASS' if python_check else '❌ FAIL'}")
    logger.info(f"Dependencies: {'✅ PASS' if dependency_check else '❌ FAIL'}")
    logger.info(f"Config files: {'✅ PASS' if config_check else '❌ FAIL'}")
    logger.info(f"Environment variables: {'✅ PASS' if env_var_check else '❌ FAIL'}")
    logger.info(f"Redis connection: {'✅ PASS' if redis_check else '❌ FAIL'}")
    logger.info(f"Development tools: {'✅ PASS' if tools_check else '⚠️ WARNING'}")
    
    # Overall status
    all_passed = python_check and dependency_check and config_check and env_var_check and redis_check
    if all_passed:
        logger.info("\n✅ All essential checks PASSED! Environment is ready for development.")
    else:
        logger.warning("\n⚠️ Some checks FAILED. See above for details and fix the issues.")
    
    return all_passed

def create_summary_report() -> Dict[str, Any]:
    """
    Create a machine-readable summary report of the verification.
    """
    # System information
    system_info = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
    }
    
    # Installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Environment variables (excluding sensitive ones)
    env_vars = {}
    sensitive_vars = {"OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"}
    for key, value in os.environ.items():
        if key.startswith(("WATCHDOG_", "REDIS_")) or key in OPTIONAL_ENV_VARS:
            if key in sensitive_vars:
                env_vars[key] = "REDACTED"
            else:
                env_vars[key] = value
    
    return {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "system_info": system_info,
        "installed_packages": installed_packages,
        "environment_variables": env_vars
    }

def main():
    """
    Main entry point for the verification script.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Verify Watchdog AI development environment")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    parser.add_argument("--report", action="store_true", help="Generate a detailed report file")
    args = parser.parse_args()
    
    if args.json:
        # JSON output for programmatic use
        report = create_summary_report()
        print(json.dumps(report, indent=2))
        return 0
    
    # Run verification checks
    all_passed = run_all_checks()
    
    if args.report:
        # Generate a detailed report
        report = create_summary_report()
        report_path = PROJECT_ROOT / "environment_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed environment report saved to {report_path}")
    
    # Return appropriate exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

