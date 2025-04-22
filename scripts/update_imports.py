#!/usr/bin/env python3
"""
Script to help update import statements from old module structure to the new watchdog_ai package.

This script searches through Python files and identifies import statements
that reference the old module structure, then suggests replacement imports
using the new package structure.

It can be run in two modes:
1. Analysis mode (default): Scans and reports findings without making changes
2. Update mode: Makes the suggested changes to the files (use with caution)

Usage:
    python update_imports.py [--update] [path1 path2 ...]
    
Arguments:
    --update: Apply the changes (default is to only report)
    path1, path2, ...: Paths to search (default is src/ and tests/)
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Tuple, Set

# Default paths to search
DEFAULT_PATHS = ['src', 'tests']

# Import pattern mapping from old to new
IMPORT_PATTERNS = [
    # Format: (regex pattern for finding, replacement template)
    (r'^from src\.insights(\..+|) import (.+)$', r'from src.watchdog_ai.insights\1 import \2'),
    (r'^import src\.insights(\..+|)$', r'import src.watchdog_ai.insights\1'),
    (r'^from src\.ui(\..+|) import (.+)$', r'from src.watchdog_ai.ui\1 import \2'),
    (r'^import src\.ui(\..+|)$', r'import src.watchdog_ai.ui\1'),
    (r'^from src\.insight_card(_consolidated|) import (.+)$', r'from src.watchdog_ai.insight_card import \2'),
    (r'^import src\.insight_card(_consolidated|)$', r'import src.watchdog_ai.insight_card'),
    (r'^from src\.insight_conversation(_consolidated|) import (.+)$', r'from src.watchdog_ai.insights.insight_conversation import \2'),
    (r'^import src\.insight_conversation(_consolidated|)$', r'import src.watchdog_ai.insights.insight_conversation')
]

def find_python_files(paths: List[str]) -> List[str]:
    """Find all Python files in the specified paths."""
    python_files = []
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist")
            continue
            
        if os.path.isfile(path) and path.endswith('.py'):
            python_files.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
    
    return python_files

def analyze_imports(filepath: str) -> List[Tuple[int, str, str]]:
    """
    Analyze imports in a Python file.
    
    Returns a list of tuples (line_number, original_line, suggested_replacement)
    """
    results = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        for pattern, replacement in IMPORT_PATTERNS:
            if re.match(pattern, stripped_line):
                new_line = re.sub(pattern, replacement, stripped_line)
                if new_line != stripped_line:
                    results.append((i+1, stripped_line, new_line))
    
    return results

def update_file(filepath: str, changes: List[Tuple[int, str, str]]) -> bool:
    """Apply the suggested changes to a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line_num, old_line, new_line in changes:
            # Line number is 1-based, list index is 0-based
            idx = line_num - 1
            # Preserve indentation and trailing whitespace/newline
            prefix = re.match(r'^(\s*)', lines[idx]).group(1)
            suffix = re.search(r'(\s*)$', lines[idx]).group(1)
            lines[idx] = f"{prefix}{new_line}{suffix}"
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
            
        return True
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update import statements to the new package structure")
    parser.add_argument('--update', action='store_true', help="Apply the changes (default is to only report)")
    parser.add_argument('paths', nargs='*', default=DEFAULT_PATHS, help="Paths to search")
    
    args = parser.parse_args()
    
    python_files = find_python_files(args.paths)
    print(f"Found {len(python_files)} Python files to analyze")
    
    files_with_changes = 0
    total_changes = 0
    
    for filepath in python_files:
        changes = analyze_imports(filepath)
        
        if changes:
            files_with_changes += 1
            total_changes += len(changes)
            
            print(f"\n{filepath}:")
            for line_num, old_line, new_line in changes:
                print(f"  Line {line_num}:")
                print(f"    - {old_line}")
                print(f"    + {new_line}")
            
            if args.update:
                success = update_file(filepath, changes)
                if success:
                    print(f"  ✓ Updated successfully")
                else:
                    print(f"  ✗ Update failed")
    
    print(f"\nSummary:")
    print(f"  {files_with_changes} files with outdated imports")
    print(f"  {total_changes} imports need updating")
    
    if args.update:
        print(f"  Changes have been applied")
    else:
        print(f"  Run with --update to apply the changes")

if __name__ == "__main__":
    main()