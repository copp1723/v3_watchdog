#!/usr/bin/env python3
"""
Script to check performance thresholds from test results.
"""

import xml.etree.ElementTree as ET
import sys
import json
from pathlib import Path
import os

def parse_test_results(xml_file):
    """Parse test results XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    performance_data = {}
    for testcase in root.findall('.//testcase'):
        if 'performance' in testcase.get('name', ''):
            # Extract performance metrics from test output
            system_out = testcase.find('system-out')
            if system_out is not None:
                metrics_text = system_out.text
                if metrics_text and 'Performance Metrics:' in metrics_text:
                    # Parse metrics from output
                    metrics = {}
                    for line in metrics_text.split('\n'):
                        if ':' in line and 'Performance Metrics:' not in line:
                            key, value = line.split(':')
                            metrics[key.strip()] = float(value.strip().replace('s', ''))
                    performance_data[testcase.get('name')] = metrics
    
    return performance_data

def check_thresholds(performance_data):
    """Check performance metrics against thresholds."""
    thresholds = {
        'metrics_calculation': 3.0,
        'heatmap_creation': 3.0,
        'trends_creation': 3.0,
        'memory_limit': 500,
    }
    
    violations = []
    for test_name, metrics in performance_data.items():
        for metric, value in metrics.items():
            if metric in thresholds and value > thresholds[metric]:
                violations.append({
                    'test': test_name,
                    'metric': metric,
                    'value': value,
                    'threshold': thresholds[metric]
                })
    
    return violations

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python check_performance.py <test-results.xml>")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    if not Path(xml_file).exists():
        print(f"Error: File {xml_file} not found")
        sys.exit(1)
    
    # Parse test results
    performance_data = parse_test_results(xml_file)
    
    # Check thresholds
    violations = check_thresholds(performance_data)
    
    # Output results
    if violations:
        print("\nPerformance threshold violations found:")
        for violation in violations:
            print(f"\nTest: {violation['test']}")
            print(f"Metric: {violation['metric']}")
            print(f"Value: {violation['value']:.2f}")
            print(f"Threshold: {violation['threshold']:.2f}")
        
        # Create GitHub summary
        summary_file = Path(os.environ.get('GITHUB_STEP_SUMMARY', 'performance-summary.md'))
        with open(summary_file, 'w') as f:
            f.write("## Performance Test Results\n\n")
            f.write("⚠️ Performance threshold violations found:\n\n")
            for violation in violations:
                f.write(f"- **{violation['test']}**: {violation['metric']} = {violation['value']:.2f} (threshold: {violation['threshold']:.2f})\n")
        
        sys.exit(1)
    else:
        print("\nAll performance tests passed thresholds!")
        
        # Create GitHub summary
        summary_file = Path(os.environ.get('GITHUB_STEP_SUMMARY', 'performance-summary.md'))
        with open(summary_file, 'w') as f:
            f.write("## Performance Test Results\n\n")
            f.write("✅ All performance tests passed thresholds!\n\n")
            f.write("### Metrics:\n")
            for test_name, metrics in performance_data.items():
                f.write(f"\n#### {test_name}\n")
                for metric, value in metrics.items():
                    f.write(f"- {metric}: {value:.2f}s\n")

if __name__ == '__main__':
    main() 