"""
Example script demonstrating the Lead Flow Optimization features.

This script shows how to use the LeadFlowOptimizer to analyze lead flow data,
identify bottlenecks in the sales process, track time from lead creation to closed,
and generate actionable recommendations for dealership staff.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validators.lead_flow_optimizer import (
    LeadFlowOptimizer,
    prepare_lead_data,
    load_test_data
)
from src.insight_tagger import InsightTagger, tag_insight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_bottlenecks(bottlenecks):
    """
    Create a bar chart showing process bottlenecks.
    
    Args:
        bottlenecks: Dictionary of bottlenecks information
    """
    stages = []
    avg_days = []
    thresholds = []
    colors = []
    
    for stage, data in bottlenecks.items():
        if 'average_days' in data and data['average_days'] is not None:
            # Format stage name for display
            display_name = stage.replace('time_', '').replace('_to_', ' → ').title()
            stages.append(display_name)
            avg_days.append(data['average_days'])
            thresholds.append(data['threshold'])
            colors.append('red' if data.get('is_bottleneck', False) else 'blue')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bar_positions = range(len(stages))
    bars = ax.bar(bar_positions, avg_days, color=colors)
    
    # Plot threshold lines
    for i, threshold in enumerate(thresholds):
        ax.hlines(threshold, i - 0.4, i + 0.4, colors='black', linestyles='dashed')
    
    # Add labels and title
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.set_ylabel('Average Days')
    ax.set_title('Lead Flow Process Bottlenecks')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Bottleneck (Over Threshold)'),
        Patch(facecolor='blue', label='Normal (Under Threshold)'),
        Patch(facecolor='white', edgecolor='black', linestyle='dashed', label='Threshold')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add value labels on top of bars
    for i, v in enumerate(avg_days):
        ax.text(i, v + 0.5, f'{v:.1f}d', ha='center')
    
    plt.tight_layout()
    plt.savefig('lead_flow_bottlenecks.png')
    logger.info("Bottlenecks chart saved as lead_flow_bottlenecks.png")

def plot_rep_performance(rep_metrics):
    """
    Create charts showing sales rep performance.
    
    Args:
        rep_metrics: Dictionary of rep performance metrics
    """
    # Extract data
    reps = []
    conversion_rates = []
    avg_close_times = []
    
    for rep, data in rep_metrics.items():
        if data.get('total_leads', 0) >= 2:  # Only include reps with enough leads
            reps.append(rep)
            conversion_rates.append(data.get('conversion_rate', 0))
            avg_close_times.append(data.get('avg_days_to_close', 0) or 0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot conversion rates
    bars1 = ax1.bar(reps, conversion_rates, color='green')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Rep Conversion Rates')
    ax1.set_ylim(0, max(conversion_rates) * 1.2 if conversion_rates else 100)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(conversion_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Plot average close times
    bars2 = ax2.bar(reps, avg_close_times, color='orange')
    ax2.set_ylabel('Average Days to Close')
    ax2.set_title('Rep Closing Speed')
    ax2.set_ylim(0, max(avg_close_times) * 1.2 if avg_close_times else 30)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(avg_close_times):
        ax2.text(i, v + 1, f'{v:.1f}d', ha='center')
    
    plt.tight_layout()
    plt.savefig('rep_performance.png')
    logger.info("Rep performance chart saved as rep_performance.png")

def plot_source_performance(source_metrics):
    """
    Create charts showing lead source performance.
    
    Args:
        source_metrics: Dictionary of source performance metrics
    """
    # Extract data
    sources = []
    conversion_rates = []
    lead_counts = []
    
    for source, data in source_metrics.items():
        if data.get('total_leads', 0) >= 2:  # Only include sources with enough leads
            sources.append(source)
            conversion_rates.append(data.get('conversion_rate', 0))
            lead_counts.append(data.get('total_leads', 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot conversion rates
    bars1 = ax1.bar(sources, conversion_rates, color='purple')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Lead Source Conversion Rates')
    ax1.set_ylim(0, max(conversion_rates) * 1.2 if conversion_rates else 100)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(conversion_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Plot lead counts
    bars2 = ax2.bar(sources, lead_counts, color='blue')
    ax2.set_ylabel('Number of Leads')
    ax2.set_title('Lead Volume by Source')
    ax2.set_ylim(0, max(lead_counts) * 1.2 if lead_counts else 20)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(lead_counts):
        ax2.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('source_performance.png')
    logger.info("Source performance chart saved as source_performance.png")

def create_insights_from_optimizer(optimizer):
    """
    Create and tag insights from optimizer results.
    
    Args:
        optimizer: LeadFlowOptimizer instance with processed data
        
    Returns:
        List of tagged insights
    """
    insights = []
    tagger = InsightTagger()
    
    # Create bottleneck insights
    bottlenecks = optimizer.identify_bottlenecks()
    for stage, data in bottlenecks.items():
        if data.get('is_bottleneck', False):
            # Format stage name for display
            display_name = stage.replace('time_', '').replace('_to_', ' → ').title()
            
            insight = {
                "title": f"Bottleneck in {display_name} Stage",
                "summary": f"Process bottleneck detected in the {display_name} stage. " +
                          f"Average time of {data.get('average_days', 0):.1f} days exceeds " +
                          f"threshold of {data.get('threshold', 0):.1f} days.",
                "metrics": {
                    "average_days": data.get('average_days', 0),
                    "median_days": data.get('median_days', 0),
                    "threshold": data.get('threshold', 0),
                    "percent_over_threshold": (data.get('average_days', 0) / data.get('threshold', 1) - 1) * 100
                },
                "recommendations": [
                    f"Review the {display_name} process to identify and address causes of delay",
                    "Implement process improvements to reduce time in this stage",
                    "Consider additional training for staff involved in this stage"
                ]
            }
            
            # Tag the insight
            tagged_insight = tag_insight(insight)
            insights.append(tagged_insight)
    
    # Create aged leads insight
    aged_leads = optimizer.flag_aged_leads()
    if aged_leads.get('count', 0) > 0:
        insight = {
            "title": "Aged Leads Requiring Attention",
            "summary": f"{aged_leads.get('count', 0)} leads ({aged_leads.get('percentage', 0):.1f}%) " +
                      f"are older than {aged_leads.get('threshold_days', 30)} days and require attention.",
            "metrics": {
                "aged_lead_count": aged_leads.get('count', 0),
                "percentage": aged_leads.get('percentage', 0),
                "threshold_days": aged_leads.get('threshold_days', 30)
            },
            "recommendations": [
                "Review aged leads and determine appropriate action (reactivate or close)",
                "Assign high-performing reps to follow up on promising aged leads",
                "Implement a lead aging alert system to prevent future accumulation"
            ]
        }
        
        # Tag the insight
        tagged_insight = tag_insight(insight)
        insights.append(tagged_insight)
    
    # Create source performance insight
    source_metrics = optimizer.get_source_performance()
    if source_metrics:
        # Find best and worst sources
        best_source = None
        best_rate = 0
        worst_source = None
        worst_rate = 100
        
        for source, data in source_metrics.items():
            if data.get('total_leads', 0) >= 3:  # Only consider sources with enough leads
                rate = data.get('conversion_rate', 0)
                if rate > best_rate:
                    best_rate = rate
                    best_source = source
                if rate < worst_rate:
                    worst_rate = rate
                    worst_source = source
        
        if best_source and worst_source and best_source != worst_source:
            insight = {
                "title": "Lead Source Performance Variance",
                "summary": f"{best_source} leads are converting at {best_rate:.1f}%, " +
                          f"while {worst_source} leads are only at {worst_rate:.1f}%.",
                "metrics": {
                    f"{best_source}_conversion": best_rate,
                    f"{worst_source}_conversion": worst_rate,
                    "conversion_gap": best_rate - worst_rate
                },
                "recommendations": [
                    f"Increase investment in high-performing {best_source} leads",
                    f"Review lead quality and follow-up process for {worst_source} leads",
                    "Implement source-specific follow-up procedures based on performance"
                ]
            }
            
            # Tag the insight
            tagged_insight = tag_insight(insight)
            insights.append(tagged_insight)
    
    return insights

def main():
    """
    Main function demonstrating lead flow optimization.
    """
    logger.info("Starting Lead Flow Optimization Example")
    
    # Check if we should use test data or sample generated data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_path, "tests", "assets", "test_lead_flow_data.csv")
    
    if os.path.exists(test_data_path):
        logger.info(f"Loading test data from {test_data_path}")
        df = pd.read_csv(test_data_path)
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if col.endswith('_date')]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        logger.info("Test data file not found, generating sample data")
        df = load_test_data()
    
    logger.info(f"Loaded {len(df)} lead records")
    
    # Initialize the optimizer
    optimizer = LeadFlowOptimizer()
    
    # Process the data
    logger.info("Processing lead flow data...")
    results = optimizer.process_lead_data(df)
    
    # Get summary
    summary = optimizer.get_summary()
    
    # Get recommendations
    recommendations = optimizer.generate_recommendations()
    
    # Display summary
    logger.info("Lead Flow Analysis Results:")
    logger.info(f"Overall lead conversion rate: {summary.get('overall_conversion_rate', 0):.1f}%")
    logger.info(f"Aged leads count: {summary.get('aged_leads_count', 0)}")
    
    if "top_bottlenecks" in summary:
        logger.info("\nTop bottlenecks:")
        for bottleneck in summary["top_bottlenecks"]:
            logger.info(f"- {bottleneck.get('stage')}: {bottleneck.get('average_days', 0):.1f} days")
    
    if recommendations:
        logger.info("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. [{rec.get('priority', 'medium')}] {rec.get('title')}: {rec.get('action')}")
    
    # Create charts
    logger.info("\nGenerating charts...")
    plot_bottlenecks(optimizer.identify_bottlenecks())
    plot_rep_performance(optimizer.get_rep_performance())
    plot_source_performance(optimizer.get_source_performance())
    
    # Create insights
    logger.info("\nGenerating tagged insights...")
    insights = create_insights_from_optimizer(optimizer)
    logger.info(f"Created {len(insights)} tagged insights")
    
    # Save results to file
    output = {
        "summary": summary,
        "recommendations": recommendations,
        "bottlenecks": optimizer.identify_bottlenecks(),
        "aged_leads": optimizer.flag_aged_leads(),
        "rep_performance": optimizer.get_rep_performance(),
        "source_performance": optimizer.get_source_performance(),
        "model_performance": optimizer.get_model_performance(),
        "insights": insights,
        "timestamp": datetime.now().isoformat()
    }
    
    with open('lead_flow_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info("Analysis saved to lead_flow_analysis.json")
    logger.info("Lead Flow Optimization Example completed")

if __name__ == "__main__":
    main()