"""
Example script demonstrating the usage of the insight_validator module.

This script loads sample dealership data, validates it using the insight_validator module,
and generates reports on the data quality issues found.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the insight_validator module
from ..src.validators.insight_validator import (
    flag_all_issues,
    summarize_flags,
    generate_flag_summary
)

def main():
    """Main function to demonstrate the insight_validator module."""
    print("Insight Validator Example")
    print("========================\n")
    
    # Get the path to the sample data
    sample_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'tests',
        'assets',
        'sample_dealership_data.csv'
    )
    
    # Check if the file exists
    if not os.path.exists(sample_data_path):
        print(f"Sample data file not found: {sample_data_path}")
        return
    
    # Load the sample data
    print(f"Loading sample data from: {sample_data_path}")
    df = pd.read_csv(sample_data_path)
    
    # Display basic information about the dataset
    print(f"\nDataset Info:")
    print(f"- Rows: {len(df)}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Column Names: {', '.join(df.columns)}")
    
    # Flag all issues in the data
    print("\nFlagging issues in the data...")
    flagged_df = flag_all_issues(df)
    
    # Display the flagged data
    print("\nFlagged Data Preview:")
    print(flagged_df.head())
    
    # Get summary of issues
    print("\nGenerating summary of issues...")
    summary = summarize_flags(flagged_df)
    
    # Display summary
    print("\nIssue Summary:")
    for key, value in summary.items():
        if key == 'issue_summary':
            print(f"- {key}:")
            for issue, count in value.items():
                print(f"  - {issue}: {count}")
        else:
            print(f"- {key}: {value}")
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    md_report = generate_flag_summary(flagged_df)
    
    # Save the report to a file
    report_path = os.path.join(
        os.path.dirname(__file__),
        'data_quality_report.md'
    )
    with open(report_path, 'w') as f:
        f.write(md_report)
    
    print(f"\nMarkdown report saved to: {report_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(flagged_df, summary)
    
    print("\nExample complete!")


def create_visualizations(df, summary):
    """Create visualizations of the flagged issues."""
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten the axes for easier indexing
    axes = axes.flatten()
    
    # 1. Negative Gross Bar Chart
    if 'flag_negative_gross' in df.columns:
        negative_gross_counts = df['flag_negative_gross'].value_counts()
        ax = axes[0]
        sns.barplot(
            x=negative_gross_counts.index.map({True: 'Negative', False: 'Positive/Zero'}),
            y=negative_gross_counts.values,
            ax=ax
        )
        ax.set_title('Gross Profit Status')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        
        # Add data labels
        for i, v in enumerate(negative_gross_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center')
    
    # 2. Missing Lead Source Bar Chart
    if 'flag_missing_lead_source' in df.columns:
        lead_source_counts = df['flag_missing_lead_source'].value_counts()
        ax = axes[1]
        sns.barplot(
            x=lead_source_counts.index.map({True: 'Missing', False: 'Present'}),
            y=lead_source_counts.values,
            ax=ax
        )
        ax.set_title('Lead Source Status')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        
        # Add data labels
        for i, v in enumerate(lead_source_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center')
    
    # 3. VIN Issues Bar Chart
    vin_issues = {
        'Duplicate VINs': summary.get('duplicate_vins_count', 0),
        'Missing/Invalid VINs': summary.get('missing_vins_count', 0),
        'Valid VINs': len(df) - summary.get('duplicate_vins_count', 0) - summary.get('missing_vins_count', 0)
    }
    ax = axes[2]
    sns.barplot(
        x=list(vin_issues.keys()),
        y=list(vin_issues.values()),
        ax=ax
    )
    ax.set_title('VIN Status')
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    
    # Add data labels
    for i, v in enumerate(vin_issues.values()):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    # 4. Overall Data Quality Pie Chart
    ax = axes[3]
    clean_records = len(df) - summary.get('total_issues', 0)
    issue_records = summary.get('total_issues', 0)
    
    ax.pie(
        [clean_records, issue_records],
        labels=['Clean Records', 'Records with Issues'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.1)
    )
    ax.set_title('Overall Data Quality')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    viz_path = os.path.join(
        os.path.dirname(__file__),
        'data_quality_visualizations.png'
    )
    plt.savefig(viz_path)
    
    print(f"Visualizations saved to: {viz_path}")


if __name__ == "__main__":
    main()
