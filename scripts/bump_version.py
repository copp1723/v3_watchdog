#!/usr/bin/env python3
"""
Bump the version number and update version references across the project.

Usage:
    python scripts/bump_version.py [major|minor|patch|alpha|beta|rc]

Examples:
    python scripts/bump_version.py minor    # Increase minor version: 0.1.0 -> 0.2.0
    python scripts/bump_version.py patch    # Increase patch version: 0.1.0 -> 0.1.1
    python scripts/bump_version.py alpha    # Change release type: 0.1.0 -> 0.1.0-alpha
    python scripts/bump_version.py beta     # Change release type: 0.1.0-alpha -> 0.1.0-beta
    python scripts/bump_version.py rc       # Change release type: 0.1.0-beta -> 0.1.0-rc
    python scripts/bump_version.py release  # Remove prerelease suffix: 0.1.0-rc -> 0.1.0
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def parse_version(version_str):
    """Parse a version string into its components."""
    # Parse version with optional prerelease tag
    # Format: MAJOR.MINOR.PATCH[-PRERELEASE]
    match = re.match(r'(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?', version_str)
    
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease = match.group(4)
    
    return major, minor, patch, prerelease


def format_version(major, minor, patch, prerelease=None):
    """Format version components into a version string."""
    version = f"{major}.{minor}.{patch}"
    if prerelease:
        version += f"-{prerelease}"
    return version


def bump_version(version_str, bump_type):
    """Bump the version according to the specified bump type."""
    major, minor, patch, prerelease = parse_version(version_str)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        prerelease = None
    elif bump_type == "minor":
        minor += 1
        patch = 0
        prerelease = None
    elif bump_type == "patch":
        patch += 1
        prerelease = None
    elif bump_type == "alpha":
        prerelease = "alpha"
    elif bump_type == "beta":
        prerelease = "beta"
    elif bump_type == "rc":
        prerelease = "rc"
    elif bump_type == "release":
        prerelease = None
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return format_version(major, minor, patch, prerelease)


def update_changelog(version, repo_root):
    """Update CHANGELOG.md with the new version."""
    changelog_path = repo_root / "CHANGELOG.md"
    
    if not changelog_path.exists():
        print(f"Warning: CHANGELOG.md not found at {changelog_path}")
        return
    
    with open(changelog_path, "r") as f:
        content = f.read()
    
    # Check if the version is already in the changelog
    if f"## [{version}]" in content:
        print(f"Version {version} already exists in CHANGELOG.md")
        return
    
    # Add new version section below the header and first paragraph
    today = datetime.now().strftime("%Y-%m-%d")
    version_header = f"## [{version}] - {today}\n\n"
    version_content = "### Added\n\n- \n\n### Changed\n\n- \n\n### Fixed\n\n- \n\n"
    
    # Find the position to insert the new version (after the first version or at the end)
    match = re.search(r'## \[\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?\]', content)
    
    if match:
        insert_pos = match.start()
        updated_content = (
            content[:insert_pos] + 
            version_header + 
            version_content + 
            content[insert_pos:]
        )
    else:
        # If no version found, add at the end
        updated_content = content + "\n" + version_header + version_content
    
    with open(changelog_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated CHANGELOG.md with version {version}")


def main():
    # Get the repository root directory
    repo_root = Path(__file__).parent.parent
    
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [major|minor|patch|alpha|beta|rc|release]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    valid_types = ["major", "minor", "patch", "alpha", "beta", "rc", "release"]
    
    if bump_type not in valid_types:
        print(f"Error: Invalid bump type '{bump_type}'. Must be one of {', '.join(valid_types)}")
        sys.exit(1)
    
    # Read the current version
    version_file = repo_root / "VERSION"
    
    if not version_file.exists():
        print(f"Error: VERSION file not found at {version_file}")
        sys.exit(1)
    
    with open(version_file, "r") as f:
        current_version = f.read().strip()
    
    # Bump the version
    try:
        new_version = bump_version(current_version, bump_type)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Update the VERSION file
    with open(version_file, "w") as f:
        f.write(new_version)
    
    print(f"Updated VERSION file: {current_version} -> {new_version}")
    
    # Update the README badge
    update_badge_script = repo_root / "scripts" / "update_version_badge.py"
    if update_badge_script.exists():
        subprocess.run([sys.executable, str(update_badge_script)])
    
    # Update the CHANGELOG
    update_changelog(new_version, repo_root)
    
    print(f"\nVersion bumped from {current_version} to {new_version}")
    print("Remember to update the CHANGELOG.md with details of the changes.")

if __name__ == "__main__":
    main()