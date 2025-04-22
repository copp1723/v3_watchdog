#!/usr/bin/env python3
"""
Generate an inventory matrix of Python modules in the project.

This script recursively scans the project directory for Python files 
and generates a detailed inventory including path, size, import name,
SHA1 hash, and last modified date.

Outputs are saved in both CSV and JSON formats.
"""

import os
import sys
import hashlib
import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
# Directories to scan
SCAN_DIRS = ["src", "tests", "scripts", "examples"]
# Directories to ignore
IGNORE_DIRS = [".git", "__pycache__", ".pytest_cache", ".venv", "venv"]
# Output file paths
OUTPUT_CSV = ROOT_DIR / "docs" / "module_inventory.csv"
OUTPUT_JSON = ROOT_DIR / "docs" / "module_inventory.json"


def calculate_sha1(file_path: Path) -> str:
    """Calculate SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def determine_import_name(file_path: Path) -> str:
    """
    Determine the import name based on package structure.
    
    Example: src/watchdog_ai/config.py -> watchdog_ai.config
    """
    rel_path = file_path.relative_to(ROOT_DIR)
    parts = list(rel_path.parts)
    
    # Handle different module patterns
    if parts[0] == "src":
        parts.pop(0)  # Remove 'src'
    
    # Convert path parts to dotted import path
    module_path = ".".join([p for p in parts])
    # Remove .py extension
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    
    return module_path


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Collect all metadata for a Python file."""
    stat = file_path.stat()
    
    return {
        "path": str(file_path.relative_to(ROOT_DIR)),
        "size_bytes": stat.st_size,
        "size_formatted": f"{stat.st_size:,} bytes",
        "import_name": determine_import_name(file_path),
        "sha1": calculate_sha1(file_path),
        "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_package": (file_path.parent / "__init__.py").exists(),
    }


def find_python_files() -> List[Dict[str, Any]]:
    """Find all Python files and gather their metadata."""
    modules = []
    
    for scan_dir in SCAN_DIRS:
        dir_path = ROOT_DIR / scan_dir
        if not dir_path.exists():
            continue
            
        for root, dirs, files in os.walk(dir_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            root_path = Path(root)
            
            for file in files:
                if file.endswith(".py"):
                    file_path = root_path / file
                    modules.append(get_file_metadata(file_path))
    
    return modules


def export_to_csv(modules: List[Dict[str, Any]], output_path: Path) -> None:
    """Export modules metadata to CSV file."""
    if not modules:
        print("No modules found to export.")
        return
        
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract field names from first module
    fieldnames = list(modules[0].keys())
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modules)
    
    print(f"CSV inventory written to {output_path}")


def export_to_json(modules: List[Dict[str, Any]], output_path: Path) -> None:
    """Export modules metadata to JSON file."""
    if not modules:
        print("No modules found to export.")
        return
        
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as jsonfile:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "module_count": len(modules),
            "modules": modules
        }, jsonfile, indent=2)
    
    print(f"JSON inventory written to {output_path}")


def summarize_modules(modules: List[Dict[str, Any]]) -> Tuple[Dict[str, int], int]:
    """Generate summary statistics about the modules."""
    total_size = sum(module["size_bytes"] for module in modules)
    
    # Count by directory
    dir_counts = {}
    for module in modules:
        path = Path(module["path"])
        top_dir = path.parts[0] if path.parts else "root"
        dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
    
    return dir_counts, total_size


def main():
    """Main entry point for the script."""
    print(f"Scanning Python modules in {ROOT_DIR}...")
    
    # Find and process all Python files
    modules = find_python_files()
    modules.sort(key=lambda x: x["path"])
    
    # Generate summary
    dir_counts, total_size = summarize_modules(modules)
    
    # Print summary to console
    print(f"\nFound {len(modules)} Python modules, total size: {total_size:,} bytes")
    print("\nModules by directory:")
    for dir_name, count in dir_counts.items():
        print(f"  {dir_name}: {count}")
    
    # Export data
    export_to_csv(modules, OUTPUT_CSV)
    export_to_json(modules, OUTPUT_JSON)
    
    print("\nInventory complete!")


if __name__ == "__main__":
    main()

