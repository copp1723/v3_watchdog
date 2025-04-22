"""
Demonstration script for the notification system and executive PDF generator.

This script showcases:
1. Creating and sending email notifications with insights
2. Generating executive-level PDF reports
3. Classifying insights using the InsightTagger
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

from src.scheduler.notification_service import NotificationService
from src.scheduler.base_scheduler import Report, ScheduledReport, DeliveryMethod, ReportFrequency, ReportFormat
from src.pdf_generator import generate_executive_pdf
from src.insight_tagger import InsightTagger, InsightPriority, InsightType, InsightStore, tag_insights


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("notification_demo")


def generate_sample_insights(count=5):
    """
    Generate sample insights for the demo.
    
    Args:
        count: Number of insights to generate
        
    Returns:
        List of sample insights
    """
    sample_insights = [
        {
            "title": "Sales Performance Exceeding Targets",
            "summary": "Sales for this month have exceeded targets by 15%, with particularly strong performance in the SUV segment.",
            "metrics": [
                {"label": "Sales Target", "value": "$1,500,000"},
                {"label": "Actual Sales", "value": "$1,725,000"},
                {"label": "Over Target", "value": "15%"}
            ],
            "recommendations": [
                "Continue promotional focus on SUV models",
                "Analyze sales rep techniques for top performers to share with team"
            ]
        },
        {
            "title": "Critical Inventory Aging Alert",
            "summary": "25 vehicles have been in inventory for over 90 days, representing a significant financial risk. The BMW models are particularly affected.",
            "metrics": [
                {"label": "Aging Vehicles", "value": "25"},
                {"label": "Oldest Vehicle", "value": "127 days"},
                {"label": "Capital Tied Up", "value": "$875,000"}
            ],
            "recommendations": [
                "Implement immediate pricing review on vehicles >90 days",
                "Consider special promotional event for aging luxury models",
                "Adjust purchasing strategy to reduce similar models"
            ]
        },
        {
            "title": "Website Lead Conversion Improvement",
            "summary": "Website lead conversion has improved by 12% after implementing the new follow-up protocol last month.",
            "metrics": [
                {"label": "Previous Rate", "value": "18%"},
                {"label": "Current Rate", "value": "30%"},
                {"label": "Improvement", "value": "12%"}
            ],
            "recommendations": [
                "Expand new follow-up protocol to all lead sources",
                "Provide additional training on digital engagement"
            ]
        },
        {
            "title": "Correlation: Weekend Traffic and Sales",
            "summary": "Strong correlation detected between weekend lot traffic and mid-week sales, suggesting customers are visiting on weekends and returning to purchase.",
            "metrics": [
                {"label": "Correlation", "value": "0.78"},
                {"label": "Weekend Traffic", "value": "↑ 35%"},
                {"label": "Mid-week Sales", "value": "↑ 28%"}
            ],
            "recommendations": [
                "Enhance weekend visitor experience with additional staff",
                "Develop specific follow-up strategy for weekend visitors",
                "Create special offer for customers who return within 5 days"
            ]
        },
        {
            "title": "Service Department Trend Analysis",
            "summary": "Service appointments have shown a consistent upward trend of 5% month-over-month for the past quarter, indicating effective customer retention.",
            "metrics": [
                {"label": "MoM Growth", "value": "5%"},
                {"label": "Q2 Appointments", "value": "1,245"},
                {"label": "Repeat Customers", "value": "68%"}
            ],
            "recommendations": [
                "Expand service capacity to maintain customer satisfaction",
                "Analyze high-retention service advisors' techniques"
            ]
        },
        {
            "title": "Financing Product Mix Opportunity",
            "summary": "Extended warranty attach rate is 15% below industry average, representing significant missed revenue opportunity.",
            "metrics": [
                {"label": "Current Rate", "value": "22%"},
                {"label": "Industry Avg", "value": "37%"},
                {"label": "Monthly Revenue Gap", "value": "$45,000"}
            ],
            "recommendations": [
                "Review warranty presentation process with F&I team",
                "Consider adjusting warranty pricing or coverage options",
                "Implement specific warranty focus in sales training"
            ]
        },
        {
            "title": "URGENT: Lead Response Time Alert",
            "summary": "Average lead response time has increased to 8.5 hours, well above the 1-hour target. This is likely causing significant lost sales opportunities.",
            "metrics": [
                {"label": "Current Avg", "value": "8.5 hours"},
                {"label": "Target", "value": "1 hour"},
                {"label": "Est. Lost Sales", "value": "12-15 units/month"}
            ],
            "recommendations": [
                "Immediately review lead assignment process",
                "Implement automated routing system for leads",
                "Establish clear accountability metrics for response time"
            ]
        }
    ]
    
    # Return the requested number of insights
    return sample_insights[:min(count, len(sample_insights))]


def demo_email_notification(email_recipients):
    """
    Demonstrate email notification functionality.
    
    Args:
        email_recipients: List of email addresses to send notifications to
    """
    logger.info("Demonstrating email notification functionality")
    
    # Create notification service
    notification_service = NotificationService()
    
    # Generate sample insights
    insights = generate_sample_insights(5)
    
    # Tag insights to prioritize them
    tagger = InsightTagger()
    for insight in insights:
        # Convert to expected format for tagger
        insight_for_tagging = {
            "summary": insight["summary"],
            "metrics": {f"metric_{i}": m["value"] for i, m in enumerate(insight["metrics"])},
            "recommendations": insight["recommendations"]
        }
        
        # Tag the insight
        tagged = tagger.tag_insight(insight_for_tagging)
        
        # Add priority and type back to original insight
        insight["priority"] = tagged["priority"]
        insight["type"] = tagged["type"]
    
    # Log the insights with their priorities
    logger.info("Insights with priorities:")
    for i, insight in enumerate(insights, 1):
        logger.info(f"{i}. {insight['title']} - Priority: {insight['priority']}, Type: {insight['type']}")
    
    # Send email with insights
    message_id = notification_service.send_insight_email(
        recipients=email_recipients,
        insights=insights,
        subject="Daily Insights from Watchdog AI Demo",
        parameters={
            "dealership_name": "Demo Dealership",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "dashboard_url": "https://watchdog.ai/dashboard"
        }
    )
    
    if message_id:
        logger.info(f"Email queued successfully with ID: {message_id}")
        
        # Check email status
        status = notification_service.queue.get_email_status(message_id)
        logger.info(f"Email status: {json.dumps(status, indent=2)}")
    else:
        logger.error("Failed to queue email")
    
    # Send a critical alert email
    critical_insight = next((i for i in insights if i.get("priority") == InsightPriority.CRITICAL), None)
    if critical_insight:
        alert = {
            "title": critical_insight["title"],
            "description": critical_insight["summary"],
            "metrics": critical_insight["metrics"],
            "recommendations": critical_insight["recommendations"]
        }
        
        alert_message_id = notification_service.send_alert_email(
            recipients=email_recipients,
            alert=alert,
            subject="Critical Alert from Watchdog AI Demo"
        )
        
        if alert_message_id:
            logger.info(f"Alert email queued successfully with ID: {alert_message_id}")
        else:
            logger.error("Failed to queue alert email")
    
    # Wait for emails to be processed
    if message_id:
        logger.info("Waiting for emails to be processed...")
        import time
        time.sleep(3)  # Give the worker thread some time to process
        
        # Check status again
        status = notification_service.queue.get_email_status(message_id)
        logger.info(f"Updated email status: {json.dumps(status, indent=2)}")
    
    # Shutdown notification service
    notification_service.shutdown()


def demo_pdf_generator():
    """Demonstrate PDF generator functionality."""
    logger.info("Demonstrating PDF generator functionality")
    
    # Generate sample data for the report
    report_data = {
        "dealership_name": "Demo Dealership",
        "report_type": "Weekly Executive Summary",
        "total_sales": 128,
        "total_revenue": "1,245,500",
        "avg_gross": "3,240",
        "avg_days": "45",
        "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": "Strong performance this week with 128 vehicles sold and $1.2M in revenue. SUV sales continue to dominate, representing 62% of total sales.",
        "kpis": [
            {
                "label": "Total Sales",
                "value": "128",
                "delta": 13,
                "delta_formatted": "+11.3%"
            },
            {
                "label": "Total Revenue",
                "value": "$1,245,500",
                "delta": 125000,
                "delta_formatted": "+11.1%"
            },
            {
                "label": "Avg. Gross Profit",
                "value": "$3,240",
                "delta": -70,
                "delta_formatted": "-2.1%"
            },
            {
                "label": "Avg. Days to Close",
                "value": "45",
                "delta": -3,
                "delta_formatted": "-6.3%"
            }
        ],
        "top_performers": [
            {"name": "John Doe", "value": "18 units / $72,450"},
            {"name": "Jane Smith", "value": "15 units / $61,800"},
            {"name": "Robert Johnson", "value": "14 units / $53,900"},
            {"name": "Emily Wilson", "value": "12 units / $43,200"}
        ],
        "insights": generate_sample_insights(3),
        "uptime": "99.8%",
        "data_freshness": "Last Updated 2 hours ago"
    }
    
    # Generate PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"executive_report_{timestamp}.pdf"
    
    try:
        pdf_path = generate_executive_pdf(
            data=report_data,
            output_file=output_filename,
            title="Weekly Executive Summary",
            dealership="Demo Dealership"
        )
        
        logger.info(f"PDF generated successfully: {pdf_path}")
        logger.info(f"View the PDF at: {os.path.abspath(pdf_path)}")
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")


def demo_insight_classification():
    """Demonstrate insight classification functionality."""
    logger.info("Demonstrating insight classification functionality")
    
    # Generate sample insights
    raw_insights = generate_sample_insights(7)
    
    # Convert to format expected by tagger
    insights_for_tagging = []
    for insight in raw_insights:
        insights_for_tagging.append({
            "summary": insight["summary"],
            "metrics": {f"metric_{i}": m["value"] for i, m in enumerate(insight["metrics"])},
            "recommendations": insight["recommendations"]
        })
    
    # Tag insights
    tagged_insights = tag_insights(insights_for_tagging)
    
    # Create a temporary store for insights
    temp_store_path = "demo_insights.json"
    store = InsightStore(storage_path=temp_store_path)
    
    # Add tagged insights to store
    for insight in tagged_insights:
        store.add_insight(insight)
    
    # Print insights by priority
    logger.info("=== Insights by Priority ===")
    for priority in [InsightPriority.CRITICAL, InsightPriority.RECOMMENDED, InsightPriority.OPTIONAL, InsightPriority.INFORMATIONAL]:
        insights = store.get_insights_by_priority(priority)
        logger.info(f"{priority.upper()}: {len(insights)} insights")
        for i, insight in enumerate(insights, 1):
            summary = insight["summary"]
            if len(summary) > 80:
                summary = summary[:77] + "..."
            logger.info(f"  {i}. {summary}")
    
    # Print insights by type
    logger.info("\n=== Insights by Type ===")
    for insight_type in [InsightType.ALERT, InsightType.ANOMALY, InsightType.TREND, InsightType.CORRELATION, 
                        InsightType.FORECAST, InsightType.COMPARISON, InsightType.BREAKDOWN, InsightType.SUMMARY]:
        insights = store.get_insights_by_type(insight_type)
        if insights:
            logger.info(f"{insight_type.upper()}: {len(insights)} insights")
            for i, insight in enumerate(insights, 1):
                summary = insight["summary"]
                if len(summary) > 80:
                    summary = summary[:77] + "..."
                logger.info(f"  {i}. {summary}")
    
    # Get critical insights
    critical_insights = store.get_insights_by_priority(InsightPriority.CRITICAL)
    if critical_insights:
        logger.info("\n=== Critical Insights for Immediate Attention ===")
        for i, insight in enumerate(critical_insights, 1):
            logger.info(f"  {i}. {insight['summary']}")
    
    # Clean up the temporary store file
    if os.path.exists(temp_store_path):
        os.remove(temp_store_path)
        logger.info(f"Removed temporary insight store: {temp_store_path}")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="Demonstration of Watchdog AI notification and reporting features")
    parser.add_argument("--email", nargs="+", help="Email address(es) to send notifications to")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report demo")
    parser.add_argument("--classify", action="store_true", help="Run insight classification demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    
    args = parser.parse_args()
    
    # Run all demos if --all is specified
    if args.all:
        args.pdf = True
        args.classify = True
        if not args.email:
            args.email = ["demo@example.com"]  # Fake email for demo
    
    # Run email notification demo if email addresses are provided
    if args.email:
        demo_email_notification(args.email)
    
    # Run PDF generator demo if requested
    if args.pdf:
        demo_pdf_generator()
    
    # Run insight classification demo if requested
    if args.classify:
        demo_insight_classification()
    
    # Show help if no arguments provided
    if not (args.email or args.pdf or args.classify or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()