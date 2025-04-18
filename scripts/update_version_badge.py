#!/usr/bin/env python3
"""
Update version badge in README.md based on VERSION file.

This script reads the current version from the VERSION file and updates
the version badge in the README.md file.

Usage:
    python scripts/update_version_badge.py
"""

import os
import re
import sys
from pathlib import Path

def main():
    # Get the repository root directory
    repo_root = Path(__file__).parent.parent
    
    # Read the version from VERSION file
    version_file = repo_root / "VERSION"
    readme_file = repo_root / "README.md"
    
    if not version_file.exists():
        print(f"Error: VERSION file not found at {version_file}")
        sys.exit(1)
    
    if not readme_file.exists():
        print(f"Error: README.md file not found at {readme_file}")
        sys.exit(1)
    
    # Read the version
    with open(version_file, "r") as f:
        version = f.read().strip()
    
    # Read the README content
    with open(readme_file, "r") as f:
        readme_content = f.read()
    
    # Define the pattern for the version badge
    version_badge_pattern = r'(!\[Version\]\(https://img\.shields\.io/badge/version-)[^-]+(.*?\))'
    
    # Create the replacement with the current version
    # Ensure special characters in version are URL encoded (- becomes --)
    url_encoded_version = version.replace("-", "--")
    replacement = r'\1' + url_encoded_version + r'\2'
    
    # Update the README with the new version badge
    updated_readme = re.sub(version_badge_pattern, replacement, readme_content)
    
    # Write the updated README back to the file
    with open(readme_file, "w") as f:
        f.write(updated_readme)
    
    print(f"Updated version badge in README.md to {version}")

if __name__ == "__main__":
    main()