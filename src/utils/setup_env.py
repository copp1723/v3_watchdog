#!/usr/bin/env python
# Making this file executable
# chmod +x setup_env.py
"""
Environment Setup Script for v3watchdog_ai

This script helps configure the .env file with proper API keys and settings
to ensure the Watchdog AI application works correctly.
"""

import os
import argparse
import sys
from dotenv import load_dotenv

def setup_env(use_mock=None, api_key=None, provider=None):
    """
    Setup or update the .env file with appropriate settings
    
    Args:
        use_mock (bool): Whether to use mock responses instead of real API calls
        api_key (str): API key for OpenAI or Anthropic
        provider (str): LLM provider (openai or anthropic)
    """
    # Path to .env file
    env_path = ".env"
    
    # Check if .env exists, and if not, check for .env.example
    if not os.path.exists(env_path) and os.path.exists(".env.example"):
        print("No .env file found. Creating from .env.example...")
        with open(".env.example", "r") as example_file:
            with open(env_path, "w") as env_file:
                env_file.write(example_file.read())
    
    # Load existing environment variables from .env
    load_dotenv(env_path)
    
    # Current settings
    current_use_mock = os.getenv("USE_MOCK", "true").lower() in ["true", "1", "yes"]
    current_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    current_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Use provided values or current values
    use_mock = use_mock if use_mock is not None else current_use_mock
    provider = provider.lower() if provider else current_provider
    
    # Read the current .env file
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    
    # Prepare new environment variables
    env_vars = {
        "USE_MOCK": "true" if use_mock else "false",
        f"{provider.upper()}_API_KEY": api_key if api_key else (current_api_key or ""),
        "LLM_PROVIDER": provider
    }
    
    # Update or add variables in the .env file
    updated_vars = set()
    updated_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            updated_lines.append(line)
            continue
        
        key, value = line.split("=", 1) if "=" in line else (line, "")
        key = key.strip()
        
        if key in env_vars:
            updated_line = f"{key}={env_vars[key]}"
            updated_lines.append(updated_line)
            updated_vars.add(key)
        else:
            updated_lines.append(line)
    
    # Add missing variables
    for key, value in env_vars.items():
        if key not in updated_vars:
            updated_lines.append(f"{key}={value}")
    
    # Write updated content back to .env file
    with open(env_path, "w") as f:
        f.write("\n".join(updated_lines) + "\n")
    
    print(f".env file updated successfully:")
    print(f"  USE_MOCK = {env_vars['USE_MOCK']}")
    print(f"  LLM_PROVIDER = {env_vars['LLM_PROVIDER']}")
    print(f"  {provider.upper()}_API_KEY = {'[SET]' if env_vars[f'{provider.upper()}_API_KEY'] else '[NOT SET]'}")

def main():
    parser = argparse.ArgumentParser(description="Setup environment for v3watchdog_ai")
    parser.add_argument("--use-mock", dest="use_mock", action="store_true", help="Use mock responses (no API calls)")
    parser.add_argument("--use-api", dest="use_mock", action="store_false", help="Use real API calls")
    parser.add_argument("--api-key", help="API key for the selected provider")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai", help="LLM provider (default: openai)")
    parser.set_defaults(use_mock=None)
    
    args = parser.parse_args()
    
    setup_env(
        use_mock=args.use_mock,
        api_key=args.api_key,
        provider=args.provider
    )
    
    print("\nEnvironment setup complete!")
    print("To run the enhanced version of the app, use: python src/app_enhanced.py")

if __name__ == "__main__":
    main()
