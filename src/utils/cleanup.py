#!/usr/bin/env python3
"""
Cleanup script for Watchdog AI project.

This script automates cleanup tasks including:
1. Scanning for TODO/FIXME/deprecated markers
2. Checking for broken documentation links
3. Identifying unused legacy artifacts
4. Running quality checks

Usage:
    python -m src.utils.cleanup [--task TASK] [--fix] [--report]

Options:
    --task TASK     Run a specific task (deprecated_code, docs_links, legacy_artifacts, quality_gate)
    --fix           Attempt to fix issues (requires confirmation)
    --report        Generate a cleanup report
    --all           Run all tasks
"""

import os
import re
import sys
import glob
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import shutil
import fnmatch
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("cleanup")

# Project root directory
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
except NameError:
    # For running as a script
    PROJECT_ROOT = Path(os.getcwd()).absolute()

# File patterns to ignore
IGNORE_PATTERNS = [
    "**/venv/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.git/**",
    "**/.github/workflows/*.yml",  # We check these separately with a YAML linter
    "**/node_modules/**",
    "**/.pytest_cache/**",
    "**/build/**",
    "**/dist/**",
    "**/*.egg-info/**",
    "**/archive/**",  # We check this separately for references
]

# Source code directories to focus on
SOURCE_DIRS = [
    "src",
    "tests",
]

# Legacy directories/patterns to check for references
LEGACY_PATTERNS = [
    "archive",
    "legacy",
]

# Code extensions to scan
CODE_EXTENSIONS = [
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".scss",
    ".sh",
    ".yml",
    ".yaml",
]

# Keywords to search for
DEPRECATION_KEYWORDS = [
    "TODO",
    "FIXME",
    "@deprecated",
    "# Deprecated",
    "// Deprecated",
    "/* Deprecated",
    "This is deprecated",
]

# Minimum code coverage threshold
MIN_COVERAGE_THRESHOLD = 85.0

@dataclass
class CodeIssue:
    """Represents an issue found in the code."""
    file: str
    line_num: int
    line: str
    issue_type: str
    keyword: str
    is_production: bool = False

    def __str__(self) -> str:
        return f"{self.file}:{self.line_num}: {self.issue_type} - {self.line.strip()}"

@dataclass
class DocLinkIssue:
    """Represents an issue with documentation links."""
    file: str
    line_num: int
    link: str
    issue: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line_num}: {self.issue} - {self.link}"

@dataclass
class LegacyArtifact:
    """Represents a legacy artifact."""
    path: str
    size: int
    references: List[str] = field(default_factory=list)
    
    @property
    def size_readable(self) -> str:
        """Return the size in a human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if self.size < 1024.0:
                return f"{self.size:.2f} {unit}"
            self.size /= 1024.0
        return f"{self.size:.2f} TB"
    
    @property
    def is_unused(self) -> bool:
        """Check if the artifact is unused (no references)."""
        return len(self.references) == 0
    
    def __str__(self) -> str:
        if self.is_unused:
            return f"{self.path} ({self.size_readable}) - UNUSED"
        else:
            return f"{self.path} ({self.size_readable}) - Used by {len(self.references)} files"

def should_ignore_file(file_path: str) -> bool:
    """Check if a file should be ignored based on patterns."""
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def prompt_for_confirmation(message: str) -> bool:
    """
    Prompt the user for confirmation.
    """
    response = input(f"{message} (y/N): ").strip().lower()
    return response in ('y', 'yes')

def check_ci_yaml_files() -> List[Tuple[str, List[int]]]:
    """
    Check CI YAML files for commented-out sections.
    """
    issues = []
    ci_files = find_files(PROJECT_ROOT, [".github/workflows/*.yml", ".github/workflows/*.yaml"])
    
    logger.info(f"Checking {len(ci_files)} CI YAML files for commented-out sections...")
    
    for ci_file in ci_files:
        rel_path = os.path.relpath(ci_file, PROJECT_ROOT)
        commented_lines = []
        
        with open(ci_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith('#') and not stripped.startswith('##'):
                    # Skip pure section header comments (often used for organization)
                    if not re.match(r'^#\s*[-=]+\s*$', stripped) and not re.match(r'^#\s*[\w\s]+\s*$', stripped):
                        commented_lines.append(line_num)
        
        if commented_lines:
            issues.append((rel_path, commented_lines))
    
    logger.info(f"Found {len(issues)} CI YAML files with commented-out sections")
    if issues:
        for file_path, lines in issues:
            logger.info(f"- {file_path}: {len(lines)} commented lines")
    
    return issues

def remove_legacy_artifacts(artifacts: List[LegacyArtifact]) -> List[str]:
    """
    Remove unused legacy artifacts.
    Requires manual confirmation for each artifact.
    Returns a list of artifacts that were removed.
    """
    removed_artifacts = []
    
    # Filter to unused artifacts only
    unused_artifacts = [a for a in artifacts if a.is_unused]
    
    if not unused_artifacts:
        logger.info("No unused artifacts to remove")
        return removed_artifacts
    
    # Sort by size (largest first)
    unused_artifacts.sort(key=lambda x: x.size, reverse=True)
    
    # Show unused artifacts
    logger.info(f"\nFound {len(unused_artifacts)} unused legacy artifacts:")
    for i, artifact in enumerate(unused_artifacts):
        logger.info(f"{i+1}. {artifact.path} ({artifact.size_readable})")
    
    # Ask for confirmation
    if not prompt_for_confirmation(f"Review {len(unused_artifacts)} unused artifacts for possible removal?"):
        logger.info("Skipping artifact removal")
        return removed_artifacts
    
    # Process each artifact
    for artifact in unused_artifacts:
        full_path = PROJECT_ROOT / artifact.path
        
        if not os.path.exists(full_path):
            logger.error(f"Artifact {artifact.path} does not exist, skipping")
            continue
        
        print(f"\nUnused artifact: {artifact.path} ({artifact.size_readable})")
        print("Options:")
        print("1. Delete artifact")
        print("2. Archive artifact")
        print("3. Skip this artifact")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            # Delete artifact
            if prompt_for_confirmation(f"Are you sure you want to DELETE {artifact.path}?"):
                os.remove(full_path)
                removed_artifacts.append(artifact.path)
                logger.info(f"Deleted {artifact.path}")
        
        elif choice == '2':
            # Archive artifact
            if archive_file(artifact.path):
                removed_artifacts.append(artifact.path)
        
        elif choice == '3':
            # Skip
            logger.info(f"Skipping {artifact.path}")
            continue
        
        else:
            logger.warning("Invalid choice, skipping this artifact")
    
    return removed_artifacts

def run_quality_checks() -> Dict[str, Any]:
    """
    Run quality checks including linting, tests, and code coverage.
    """
    results = {
        "linting": {"success": False, "output": ""},
        "tests": {"success": False, "output": ""},
        "coverage": {"success": False, "percentage": 0.0, "output": ""},
        "visual_tests": {"success": False, "output": ""},
    }
    
    logger.info("Running quality checks...")
    
    # Check if flake8 is installed
    try:
        # Run flake8
        logger.info("Running flake8 linting...")
        process = subprocess.run(
            ["flake8", "src", "tests"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        results["linting"]["output"] = process.stdout + process.stderr
        results["linting"]["success"] = process.returncode == 0
        
        if results["linting"]["success"]:
            logger.info("✅ Linting passed")
        else:
            logger.error("❌ Linting failed")
            logger.error(results["linting"]["output"])
    except FileNotFoundError:
        logger.warning("⚠️ flake8 not found, skipping linting")
        results["linting"]["output"] = "flake8 not installed"
    
    # Run pytest
    try:
        logger.info("Running unit tests...")
        process = subprocess.run(
            ["pytest", "tests/unit", "-v"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        results["tests"]["output"] = process.stdout + process.stderr
        results["tests"]["success"] = process.returncode == 0
        
        if results["tests"]["success"]:
            logger.info("✅ Unit tests passed")
        else:
            logger.error("❌ Unit tests failed")
            logger.error(results["tests"]["output"].split("\n")[-10:])  # Show last few lines
    except FileNotFoundError:
        logger.warning("⚠️ pytest not found, skipping tests")
        results["tests"]["output"] = "pytest not installed"
    
    # Run coverage
    try:
        logger.info("Running code coverage...")
        process = subprocess.run(
            ["pytest", "tests/unit", "--cov=src", "--cov-report=term"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        results["coverage"]["output"] = process.stdout + process.stderr
        
        # Extract coverage percentage
        coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', results["coverage"]["output"])
        if coverage_match:
            coverage_percentage = float(coverage_match.group(1))
            results["coverage"]["percentage"] = coverage_percentage
            results["coverage"]["success"] = coverage_percentage >= MIN_COVERAGE_THRESHOLD
            
            if results["coverage"]["success"]:
                logger.info(f"✅ Code coverage is {coverage_percentage}% (threshold: {MIN_COVERAGE_THRESHOLD}%)")
            else:
                logger.error(f"❌ Code coverage is {coverage_percentage}% (below threshold: {MIN_COVERAGE_THRESHOLD}%)")
        else:
            logger.error("❌ Couldn't parse coverage output")
    except FileNotFoundError:
        logger.warning("⚠️ pytest-cov not found, skipping coverage")
        results["coverage"]["output"] = "pytest-cov not installed"
    
    # Run visual tests if available
    try:
        logger.info("Running visual tests...")
        process = subprocess.run(
            ["pytest", "tests", "-m", "visual", "-v"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False
        )
        results["visual_tests"]["output"] = process.stdout + process.stderr
        
        # Check if tests were skipped (no visual tests available)
        if "no tests ran" in results["visual_tests"]["output"].lower():
            logger.info("ℹ️ No visual tests found, skipping")
            results["visual_tests"]["success"] = True  # Don't fail the check if no tests exist
        else:
            results["visual_tests"]["success"] = process.returncode == 0
            
            if results["visual_tests"]["success"]:
                logger.info("✅ Visual tests passed")
            else:
                logger.error("❌ Visual tests failed")
    except FileNotFoundError:
        logger.warning("⚠️ pytest not found for visual tests")
        results["visual_tests"]["output"] = "pytest not installed"
    
    return results

def generate_cleanup_report(
    code_issues: List[CodeIssue], 
    doc_issues: List[DocLinkIssue],
    legacy_artifacts: List[LegacyArtifact],
    ci_yaml_issues: List[Tuple[str, List[int]]],
    quality_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a cleanup report summarizing all findings.
    """
    # Count production vs non-production code issues
    prod_code_issues = [i for i in code_issues if i.is_production]
    non_prod_code_issues = [i for i in code_issues if not i.is_production]
    
    # Count unused vs used legacy artifacts
    unused_artifacts = [a for a in legacy_artifacts if a.is_unused]
    large_unused_artifacts = [a for a in unused_artifacts if a.size > 2 * 1024 * 1024]  # > 2MB
    
    # Generate a report dictionary
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "summary": {
            "code_issues": {
                "total": len(code_issues),
                "production": len(prod_code_issues),
                "non_production": len(non_prod_code_issues),
                "todo_fixme_count": sum(1 for i in code_issues if i.issue_type == 'TODO/FIXME'),
                "deprecated_count": sum(1 for i in code_issues if i.issue_type == 'Deprecated'),
            },
            "doc_issues": {
                "total": len(doc_issues),
                "broken_links": len(doc_issues),
            },
            "legacy_artifacts": {
                "total": len(legacy_artifacts),
                "unused": len(unused_artifacts),
                "large_unused": len(large_unused_artifacts),
                "total_unused_size_mb": sum(a.size for a in unused_artifacts) / (1024*1024),
            },
            "ci_yaml_issues": {
                "files_with_comments": len(ci_yaml_issues),
                "total_commented_lines": sum(len(lines) for _, lines in ci_yaml_issues),
            },
            "quality_checks": {
                "linting_passed": quality_results.get("linting", {}).get("success", False),
                "tests_passed": quality_results.get("tests", {}).get("success", False),
                "coverage_passed": quality_results.get("coverage", {}).get("success", False),
                "coverage_percentage": quality_results.get("coverage", {}).get("percentage", 0.0),
                "visual_tests_passed": quality_results.get("visual_tests", {}).get("success", False),
            }
        },
        "details": {
            "code_issues": [
                {
                    "file": issue.file,
                    "line": issue.line_num,
                    "type": issue.issue_type,
                    "content": issue.line.strip(),
                    "is_production": issue.is_production,
                }
                for issue in code_issues
            ],
            "doc_issues": [
                {
                    "file": issue.file,
                    "line": issue.line_num,
                    "link": issue.link,
                    "issue": issue.issue,
                }
                for issue in doc_issues
            ],
            "unused_artifacts": [
                {
                    "path": artifact.path,
                    "size_bytes": artifact.size,
                    "size_readable": artifact.size_readable,
                }
                for artifact in unused_artifacts
            ],
            "ci_yaml_issues": [
                {
                    "file": file_path,
                    "commented_lines": lines,
                }
                for file_path, lines in ci_yaml_issues
            ],
        }
    }
    
    return report

def save_report_to_file(report: Dict[str, Any], report_format: str = "md") -> str:
    """
    Save the report to a file. Returns the path to the saved file.
    """
    report_path = PROJECT_ROOT / "docs" / "cleanup_report.md"
    if report_format == "json":
        # Save as JSON
        json_path = PROJECT_ROOT / "docs" / "cleanup_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")
        return str(json_path)
    
    # Save as Markdown
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, "w") as f:
        f.write(f"# Watchdog AI Cleanup Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("## Summary\n\n")
        
        # Code issues summary
        code_summary = report["summary"]["code_issues"]
        f.write("### Code Issues\n\n")
        f.write(f"- Total issues: **{code_summary['total']}**\n")
        f.write(f"- Production code issues: **{code_summary['production']}**\n")
        f.write(f"- Non-production code issues: **{code_summary['non_production']}**\n")
        f.write(f"- TODO/FIXME count: **{code_summary['todo_fixme_count']}**\n")
        f.write(f"- Deprecated code count: **{code_summary['deprecated_count']}**\n\n")
        
        # Doc issues summary
        doc_summary = report["summary"]["doc_issues"]
        f.write("### Documentation Issues\n\n")
        f.write(f"- Total issues: **{doc_summary['total']}**\n")
        f.write(f"- Broken links: **{doc_summary['broken_links']}**\n\n")
        
        # Legacy artifacts summary
        legacy_summary = report["summary"]["legacy_artifacts"]
        f.write("### Legacy Artifacts\n\n")
        f.write(f"- Total artifacts: **{legacy_summary['total']}**\n")
        f.write(f"- Unused artifacts: **{legacy_summary['unused']}**\n")
        f.write(f"- Large unused artifacts (>2MB): **{legacy_summary['large_unused']}**\n")
        f.write(f"- Total unused size: **{legacy_summary['total_unused_size_mb']:.2f} MB**\n\n")
        
        # CI YAML issues summary
        ci_summary = report["summary"]["ci_yaml_issues"]
        f.write("### CI YAML Issues\n\n")
        f.write(f"- Files with commented sections: **{ci_summary['files_with_comments']}**\n")
        f.write(f"- Total commented lines: **{ci_summary['total_commented_lines']}**\n\n")
        
        # Quality checks summary
        quality_summary = report["summary"]["quality_checks"]
        f.write("### Quality Checks\n\n")
        f.write(f"- Linting: **{'PASS' if quality_summary['linting_passed'] else 'FAIL'}**\n")
        f.write(f"- Unit tests: **{'PASS' if quality_summary['tests_passed'] else 'FAIL'}**\n")
        f.write(f"- Code coverage: **{'PASS' if quality_summary['coverage_passed'] else 'FAIL'}** ({quality_summary['coverage_percentage']:.1f}%)\n")
        f.write(f"- Visual tests: **{'PASS' if quality_summary['visual_tests_passed'] else 'FAIL'}**\n\n")
        
        # Detailed sections
        if report["details"]["code_issues"]:
            f.write("## Detailed Code Issues\n\n")
            f.write("| File | Line | Type | Content | Production |\n")
            f.write("|------|------|------|---------|------------|\n")
            for issue in report["details"]["code_issues"]:
                content = issue["content"].replace("|", "\\|")  # Escape pipe characters
                if len(content) > 50:
                    content = content[:47] + "..."
                f.write(f"| {issue['file']} | {issue['line']} | {issue['type']} | {content} | {'Yes' if issue['is_production'] else 'No'} |\n")
            f.write("\n")
        
        if report["details"]["doc_issues"]:
            f.write("## Detailed Documentation Issues\n\n")
            f.write("| File | Line | Link | Issue |\n")
            f.write("|------|------|------|-------|\n")
            for issue in report["details"]["doc_issues"]:
                link = issue["link"].replace("|", "\\|")  # Escape pipe characters
                f.write(f"| {issue['file']} | {issue['line']} | {link} | {issue['issue']} |\n")
            f.write("\n")
        
        if report["details"]["unused_artifacts"]:
            f.write("## Unused Artifacts\n\n")
            f.write("| Path | Size |\n")
            f.write("|------|------|\n")
            for artifact in report["details"]["unused_artifacts"]:
                f.write(f"| {artifact['path']} | {artifact['size_readable']} |\n")
            f.write("\n")
        
        if report["details"]["ci_yaml_issues"]:
            f.write("## CI YAML Issues\n\n")
            for issue in report["details"]["ci_yaml_issues"]:
                f.write(f"### {issue['file']}\n\n")
                f.write(f"- {len(issue['commented_lines'])} commented lines: {', '.join(map(str, issue['commented_lines'][:10]))}{'...' if len(issue['commented_lines']) > 10 else ''}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if code_summary['production'] > 0:
            f.write("- **High Priority**: Clean up the TODO/FIXME and deprecated code in production files\n")
        
        if doc_summary['broken_links'] > 0:
            f.write("- **Medium Priority**: Fix broken documentation links\n")
        
        if legacy_summary['large_unused'] > 0:
            f.write("- **Medium Priority**: Remove large unused legacy artifacts to reduce repository size\n")
        
        if ci_summary['files_with_comments'] > 0:
            f.write("- **Low Priority**: Clean up commented sections in CI workflow files\n")
        
        if not quality_summary['linting_passed'] or not quality_summary['tests_passed']:
            f.write("- **High Priority**: Fix linting issues and failing tests\n")
        
        if not quality_summary['coverage_passed']:
            f.write(f"- **Medium Priority**: Improve test coverage to meet the {MIN_COVERAGE_THRESHOLD}% threshold\n")
    
    logger.info(f"Markdown report saved to {report_path}")
    return str(report_path)

def find_files(base_dir: Union[str, Path], patterns: List[str], ignore_patterns: List[str] = None) -> List[str]:
    """Find files matching patterns while ignoring specified patterns."""
    if ignore_patterns is None:
        ignore_patterns = IGNORE_PATTERNS
    
    base_path = Path(base_dir)
    matched_files = []
    
    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            rel_path = str(file_path.relative_to(base_path))
            if not any(fnmatch.fnmatch(rel_path, ignore) for ignore in ignore_patterns):
                matched_files.append(str(file_path))
    
    return matched_files

def scan_for_deprecated_code() -> List[CodeIssue]:
    """
    Scan the codebase for TODO, FIXME, and deprecated markers.
    """
    issues = []
    
    # Get all code files
    source_files = []
    for source_dir in SOURCE_DIRS:
        dir_path = PROJECT_ROOT / source_dir
        if not dir_path.exists():
            continue
        
        for ext in CODE_EXTENSIONS:
            source_files.extend(find_files(PROJECT_ROOT, [f"{source_dir}/**/*{ext}"]))
    
    logger.info(f"Scanning {len(source_files)} source files for deprecated code markers...")
    
    # Scan each file
    for file_path in source_files:
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        is_production = rel_path.startswith('src')
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for keyword in DEPRECATION_KEYWORDS:
                    if keyword.lower() in line.lower():
                        issue_type = 'TODO/FIXME' if any(k in keyword for k in ['TODO', 'FIXME']) else 'Deprecated'
                        issues.append(CodeIssue(
                            file=rel_path,
                            line_num=line_num,
                            line=line,
                            issue_type=issue_type,
                            keyword=keyword,
                            is_production=is_production
                        ))
    
    logger.info(f"Found {len(issues)} deprecated code markers")
    logger.info(f"- Production code issues: {sum(1 for i in issues if i.is_production)}")
    return issues

def check_markdown_links() -> List[DocLinkIssue]:
    """
    Check for broken links in markdown documentation.
    """
    issues = []
    doc_files = find_files(PROJECT_ROOT, ["docs/**/*.md", "*.md"])
    
    logger.info(f"Checking links in {len(doc_files)} markdown files...")
    
    # Regular expression to find markdown links
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    for doc_file in doc_files:
        rel_doc_path = os.path.relpath(doc_file, PROJECT_ROOT)
        doc_dir = os.path.dirname(doc_file)
        
        with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in link_pattern.finditer(line):
                    link_text, link_target = match.groups()
                    
                    # Skip external links and anchors
                    if link_target.startswith(('http://', 'https://', '#', 'mailto:')):
                        continue
                    
                    # Check if the link target exists
                    if link_target.startswith('/'):
                        # Absolute path within the project
                        target_path = os.path.join(PROJECT_ROOT, link_target.lstrip('/'))
                    else:
                        # Relative path
                        target_path = os.path.normpath(os.path.join(doc_dir, link_target))
                    
                    if not os.path.exists(target_path):
                        issues.append(DocLinkIssue(
                            file=rel_doc_path,
                            line_num=line_num,
                            link=link_target,
                            issue="Broken link"
                        ))
    
    logger.info(f"Found {len(issues)} broken documentation links")
    return issues

def find_legacy_artifacts() -> List[LegacyArtifact]:
    """
    Identify legacy artifacts and check if they're still referenced.
    """
    artifacts = []
    
    # Find all files in legacy directories
    legacy_paths = []
    for pattern in LEGACY_PATTERNS:
        legacy_dir = PROJECT_ROOT / pattern
        if legacy_dir.exists() and legacy_dir.is_dir():
            for root, _, files in os.walk(legacy_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    legacy_paths.append(file_path)
    
    logger.info(f"Checking {len(legacy_paths)} potential legacy artifacts...")
    
    # For each legacy file, check if it's referenced elsewhere
    for file_path in legacy_paths:
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        size = os.path.getsize(file_path)
        
        # Skip very small files
        if size < 100:
            continue
        
        # Find references to this file
        references = []
        file_name = os.path.basename(file_path)
        
        # Skip common filenames that might have many false positives
        if file_name in ['__init__.py', 'README.md', 'requirements.txt']:
            continue
        
        # Get all potential source files
        source_files = []
        for source_dir in SOURCE_DIRS:
            source_path = PROJECT_ROOT / source_dir
            if source_path.exists():
                for ext in CODE_EXTENSIONS:
                    source_files.extend(find_files(PROJECT_ROOT, [f"{source_dir}/**/*{ext}"]))
        
        # Check for references
        for source_file in source_files:
            # Skip self-references
            if source_file == file_path:
                continue
                
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Check for imports or references to the file
                if rel_path in content or file_name in content:
                    references.append(os.path.relpath(source_file, PROJECT_ROOT))
        
        # Add to artifacts list
        artifacts.append(LegacyArtifact(
            path=rel_path,
            size=size,
            references=references
        ))
    
    # Sort by size (largest first)
    artifacts.sort(key=lambda x: x.size, reverse=True)
    
    unused_artifacts = [a for a in artifacts if a.is_unused]
    large_artifacts = [a for a in unused_artifacts if a.size > 2 * 1024 * 1024]  # > 2MB
    
    logger.info(f"Found {len(unused_artifacts)} unused legacy artifacts")
    logger.info(f"- Large unused artifacts (>2MB): {len(large_artifacts)}")
    logger.info(f"- Total size of unused artifacts: {sum(a.size for a in unused_artifacts) / (1024*1024):.2f} MB")
    
    return artifacts

def main():
    """
    Main entry point for the cleanup script.
    """
    parser = argparse.ArgumentParser(description="Watchdog AI Cleanup Script")
    parser.add_argument("--task", choices=["deprecated_code", "docs_links", "legacy_artifacts", "ci_yaml", "quality_gate", "all"],
                      help="Run a specific cleanup task")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues (requires confirmation)")
    parser.add_argument("--report", action="store_true", help="Generate a cleanup report")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    
    args = parser.parse_args()
    
    # Initialize result containers
    code_issues = []
    doc_issues = []
    legacy_artifacts = []
    ci_yaml_issues = []
    quality_results = {}
    
    # Run requested tasks
    tasks_to_run = []
    if args.task == "all" or args.task is None:
        tasks_to_run = ["deprecated_code", "docs_links", "legacy_artifacts", "ci_yaml", "quality_gate"]
    elif args.task:
        tasks_to_run = [args.task]
    
    logger.info("=== Watchdog AI Cleanup Script ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    for task in tasks_to_run:
        if task == "deprecated_code":
            logger.info("\n=== Scanning for deprecated code ===")
            code_issues = scan_for_deprecated_code()
            
            if args.fix and code_issues:
                logger.info("\n=== Fixing deprecated code issues ===")
                if prompt_for_confirmation(f"Fix {len(code_issues)} deprecated code issues?"):
                    modified_files = fix_deprecated_code(code_issues)
                    if modified_files:
                        logger.info(f"Modified {len(modified_files)} files to fix deprecated code issues")
        
        elif task == "docs_links":
            logger.info("\n=== Checking documentation links ===")
            doc_issues = check_markdown_links()
            
            if args.fix and doc_issues:
                logger.info("\n=== Fixing broken documentation links ===")
                if prompt_for_confirmation(f"Fix {len(doc_issues)} broken documentation links?"):
                    modified_files = fix_broken_links(doc_issues)
                    if modified_files:
                        logger.info(f"Modified {len(modified_files)} files to fix broken links")
        
        elif task == "legacy_artifacts":
            logger.info("\n=== Finding legacy artifacts ===")
            legacy_artifacts = find_legacy_artifacts()
            
            if args.fix and legacy_artifacts:
                unused_artifacts = [a for a in legacy_artifacts if a.is_unused]
                if unused_artifacts:
                    logger.info("\n=== Removing unused legacy artifacts ===")
                    if prompt_for_confirmation(f"Remove {len(unused_artifacts)} unused legacy artifacts?"):
                        removed_artifacts = remove_legacy_artifacts(legacy_artifacts)
                        if removed_artifacts:
                            logger.info(f"Removed {len(removed_artifacts)} legacy artifacts")
        
        elif task == "ci_yaml":
            logger.info("\n=== Checking CI YAML files ===")
            ci_yaml_issues = check_ci_yaml_files()
            
            if args.fix and ci_yaml_issues:
                logger.info("\n=== Fixing CI YAML files ===")
                if prompt_for_confirmation(f"Fix {len(ci_yaml_issues)} CI YAML files with commented sections?"):
                    modified_files = fix_ci_yaml_files(ci_yaml_issues)
                    if modified_files:
                        logger.info(f"Modified {len(modified_files)} CI YAML files")
        
        elif task == "quality_gate":
            logger.info("\n=== Running quality checks ===")
            quality_results = run_quality_checks()
    
    # Generate report if requested
    if args.report or args.json:
        logger.info("\n=== Generating cleanup report ===")
        
        # Run any checks that haven't been run yet but are needed for the report
        if not code_issues:
            code_issues = scan_for_deprecated_code()
        if not doc_issues:
            doc_issues = check_markdown_links()
        if not legacy_artifacts:
            legacy_artifacts = find_legacy_artifacts()
        if not ci_yaml_issues:
            ci_yaml_issues = check_ci_yaml_files()
        if not quality_results:
            quality_results = run_quality_checks()
        
        # Generate report
        report = generate_cleanup_report(
            code_issues=code_issues,
            doc_issues=doc_issues,
            legacy_artifacts=legacy_artifacts,
            ci_yaml_issues=ci_yaml_issues,
            quality_results=quality_results
        )
        
        # Save report
        format_type = "json" if args.json else "md"
        report_path = save_report_to_file(report, format_type)
        logger.info(f"Report saved to {report_path}")
    
    # Print summary
    logger.info("\n=== Cleanup Summary ===")
    logger.info(f"Deprecated code issues: {len(code_issues)} (production: {sum(1 for i in code_issues if i.is_production)})")
    logger.info(f"Documentation link issues: {len(doc_issues)}")
    logger.info(f"Legacy artifacts: {len(legacy_artifacts)} (unused: {sum(1 for a in legacy_artifacts if a.is_unused)})")
    logger.info(f"CI YAML issues: {len(ci_yaml_issues)} files with {sum(len(lines) for _, lines in ci_yaml_issues)} commented lines")
    
    if quality_results:
        logger.info("\nQuality Gate:")
        logger.info(f"Linting: {'PASSED' if quality_results.get('linting', {}).get('success', False) else 'FAILED'}")
        logger.info(f"Tests: {'PASSED' if quality_results.get('tests', {}).get('success', False) else 'FAILED'}")
        logger.info(f"Coverage: {'PASSED' if quality_results.get('coverage', {}).get('success', False) else 'FAILED'} " + 
                   f"({quality_results.get('coverage', {}).get('percentage', 0.0):.1f}%)")
        logger.info(f"Visual Tests: {'PASSED' if quality_results.get('visual_tests', {}).get('success', False) else 'FAILED'}")
    
    # Return code based on issues found
    return 0

def fix_deprecated_code(issues: List[CodeIssue]) -> List[str]:
    """
    Fix deprecated code issues by archiving or removing them.
    Requires manual confirmation for each issue.
    Returns a list of files that were modified.
    """
    modified_files = set()
    
    # Group issues by file for more efficient processing
    issues_by_file = {}
    for issue in issues:
        if issue.file not in issues_by_file:
            issues_by_file[issue.file] = []
        issues_by_file[issue.file].append(issue)
    
    # Sort issues by line number in reverse (to avoid shifting line numbers)
    for file_path, file_issues in issues_by_file.items():
        file_issues.sort(key=lambda x: x.line_num, reverse=True)
    
    # Process each file
    for file_path, file_issues in issues_by_file.items():
        full_path = PROJECT_ROOT / file_path
        
        if not os.path.exists(full_path):
            logger.error(f"File {file_path} does not exist, skipping")
            continue
        
        # Read file content
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Show issues in this file
        logger.info(f"\nFile: {file_path}")
        logger.info("Issues:")
        for i, issue in enumerate(file_issues):
            logger.info(f"{i+1}. Line {issue.line_num}: {issue.line.strip()}")
        
        # Ask for confirmation
        if not prompt_for_confirmation(f"Process {len(file_issues)} issues in {file_path}?"):
            logger.info(f"Skipping {file_path}")
            continue
        
        # Process each issue, asking for action
        modified = False
        backup_created = False
        
        for issue in file_issues:
            print(f"\nIssue at line {issue.line_num}: {issue.line.strip()}")
            print("Options:")
            print("1. Remove line")
            print("2. Comment out line")
            print("3. Fix manually (open editor)")
            print("4. Skip this issue")
            print("5. Archive to /archive directory")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            # Create backup file if not already done
            if not backup_created:
                backup_path = f"{full_path}.bak"
                shutil.copy2(full_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
                backup_created = True
            
            line_idx = issue.line_num - 1  # Convert to 0-based index
            
            if choice == '1':
                # Remove line
                lines.pop(line_idx)
                modified = True
                logger.info(f"Removed line {issue.line_num}")
            
            elif choice == '2':
                # Comment out line
                lines[line_idx] = f"# ARCHIVED: {lines[line_idx]}"
                modified = True
                logger.info(f"Commented out line {issue.line_num}")
            
            elif choice == '3':
                # Open editor - just inform user they should edit manually
                print(f"Please edit {full_path} manually to fix line {issue.line_num}")
                if prompt_for_confirmation("Mark as fixed anyway?"):
                    modified = True
                    logger.info(f"Marked line {issue.line_num} as manually fixed")
            
            elif choice == '4':
                # Skip
                logger.info(f"Skipping line {issue.line_num}")
                continue
            
            elif choice == '5':
                # Archive
                archive_file(file_path, issue)
                modified = True
                logger.info(f"Archived content from line {issue.line_num}")
            
            else:
                logger.warning("Invalid choice, skipping this issue")
        
        # Write changes back to file
        if modified:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            modified_files.add(file_path)
            logger.info(f"Updated {file_path}")
    
    return list(modified_files)

def archive_file(file_path: str, issue: CodeIssue = None) -> bool:
    """
    Archive a file or a specific issue to the archive directory.
    
    Args:
        file_path: The file to archive
        issue: Optional issue to archive (specific line)
        
    Returns:
        bool: True if successful, False otherwise
    """
    full_path = PROJECT_ROOT / file_path
    if not os.path.exists(full_path):
        logger.error(f"File {file_path} does not exist, cannot archive")
        return False
    
    # Create archive directory if it doesn't exist
    archive_dir = PROJECT_ROOT / "archive" / "cleanup_archive"
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create archive file path, preserving directory structure
    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
    archive_path = archive_dir / rel_path
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    
    if issue is None:
        # Archive whole file
        shutil.copy2(full_path, archive_path)
        logger.info(f"Archived {file_path} to {archive_path}")
        return True
    else:
        # Archive specific issue
        with open(archive_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n# Archived from {file_path} line {issue.line_num} on {__import__('datetime').datetime.now()}\n")
            f.write(issue.line)
        logger.info(f"Archived line {issue.line_num} from {file_path} to {archive_path}")
        return True

def fix_broken_links(issues: List[DocLinkIssue]) -> List[str]:
    """
    Fix broken documentation links.
    Requires manual confirmation for each issue.
    Returns a list of files that were modified.
    """
    modified_files = set()
    
    # Group issues by file
    issues_by_file = {}
    for issue in issues:
        if issue.file not in issues_by_file:
            issues_by_file[issue.file] = []
        issues_by_file[issue.file].append(issue)
    
    # Process each file
    for file_path, file_issues in issues_by_file.items():
        full_path = PROJECT_ROOT / file_path
        
        if not os.path.exists(full_path):
            logger.error(f"File {file_path} does not exist, skipping")
            continue
        
        # Read file content
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Show issues in this file
        logger.info(f"\nFile: {file_path}")
        logger.info("Broken links:")
        for i, issue in enumerate(file_issues):
            logger.info(f"{i+1}. Line {issue.line_num}: {issue.link}")
        
        # Ask for confirmation
        if not prompt_for_confirmation(f"Fix {len(file_issues)} broken links in {file_path}?"):
            logger.info(f"Skipping {file_path}")
            continue
        
        # Create backup file
        backup_path = f"{full_path}.bak"
        shutil.copy2(full_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Process each issue
        modified = False
        content_lines = content.split('\n')
        
        for issue in file_issues:
            print(f"\nBroken link at line {issue.line_num}: {issue.link}")
            print("Options:")
            print("1. Replace with correct link")
            print("2. Remove link (keep text)")
            print("3. Skip this issue")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            line_idx = issue.line_num - 1  # Convert to 0-based index
            line = content_lines[line_idx]
            
            if choice == '1':
                # Replace with correct link
                new_link = input("Enter the correct link: ").strip()
                old_pattern = f"]({issue.link})"
                new_pattern = f"]({new_link})"
                content_lines[line_idx] = line.replace(old_pattern, new_pattern)
                modified = True
                logger.info(f"Replaced link at line {issue.line_num}: {issue.link} -> {new_link}")
            
            elif choice == '2':
                # Remove link (keep text)
                link_pattern = re.compile(r'\[([^\]]+)\]\(' + re.escape(issue.link) + r'\)')
                match = link_pattern.search(line)
                if match:
                    link_text = match.group(1)
                    content_lines[line_idx] = line.replace(match.group(0), link_text)
                    modified = True
                    logger.info(f"Removed link at line {issue.line_num}, keeping text: {link_text}")
                else:
                    logger.error(f"Could not find link pattern in line {issue.line_num}")
            
            elif choice == '3':
                # Skip
                logger.info(f"Skipping line {issue.line_num}")
                continue
            
            else:
                logger.warning("Invalid choice, skipping this issue")
        
        # Write changes back to file
        if modified:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            modified_files.add(file_path)
            logger.info(f"Updated {file_path}")
    
    return list(modified_files)

def fix_ci_yaml_files(issues: List[Tuple[str, List[int]]]) -> List[str]:
    """
    Fix CI YAML files by removing commented-out sections.
    Requires manual confirmation for each file.
    Returns a list of files that were modified.
    """
    modified_files = []
    
    if not issues:
        logger.info("No CI YAML files with commented sections to fix")
        return modified_files
    
    # Show files with commented sections
    logger.info(f"\nFound {len(issues)} CI YAML files with commented sections:")
    for i, (file_path, lines) in enumerate(issues):
        logger.info(f"{i+1}. {file_path} ({len(lines)} commented lines)")
    
    # Ask for confirmation
    if not prompt_for_confirmation(f"Review {len(issues)} CI YAML files for possible cleanup?"):
        logger.info("Skipping CI YAML cleanup")
        return modified_files
    
    # Process each file
    for file_path, commented_lines in issues:
        full_path = PROJECT_ROOT / file_path
        
        if not os.path.exists(full_path):
            logger.error(f"File {file_path} does not exist, skipping")
            continue
        
        # Read file content
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Show commented lines
        logger.info(f"\nFile: {file_path}")
        logger.info("Commented lines:")
        for i, line_num in enumerate(commented_lines[:10]):
            line_idx = line_num - 1  # Convert to 0-based index
            if 0 <= line_idx < len(lines):
                logger.info(f"{i+1}. Line {line_num}: {lines[line_idx].strip()}")
        if len(commented_lines) > 10:
            logger.info(f"... and {len(commented_lines) - 10} more")
        
        # Ask for confirmation
        if not prompt_for_confirmation(f"Clean up {len(commented_lines)} commented lines in {file_path}?"):
            logger.info(f"Skipping {file_path}")
            continue
        
        # Create backup file
        backup_path = f"{full_path}.bak"
        shutil.copy2(full_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Remove all commented lines or prompt for each one
        if len(commented_lines) > 10 and prompt_for_confirmation("Remove all commented lines at once?"):
            # Remove all commented lines at once
            # Sort in reverse order to avoid shifting line numbers
            modified = False
            for line_num in sorted(commented_lines, reverse=True):
                line_idx = line_num - 1  # Convert to 0-based index
                if 0 <= line_idx < len(lines):
                    lines.pop(line_idx)
                    modified = True
            logger.info(f"Removed all {len(commented_lines)} commented lines from {file_path}")
        else:
            # Prompt for each line
            modified = False
            # Sort in reverse order to avoid shifting line numbers
            for line_num in sorted(commented_lines, reverse=True):
                line_idx = line_num - 1  # Convert to 0-based index
                if 0 <= line_idx < len(lines):
                    print(f"\nCommented line {line_num}: {lines[line_idx].strip()}")
                    if prompt_for_confirmation("Remove this line?"):
                        lines.pop(line_idx)
                        modified = True
                        logger.info(f"Removed line {line_num}")
                    else:
                        logger.info(f"Kept line {line_num}")
        
        # Write changes back to file
        if modified:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            modified_files.append(file_path)
            logger.info(f"Updated {file_path}")
    
    return modified_files

if __name__ == "__main__":
    sys.exit(main())

