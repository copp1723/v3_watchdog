# Watchdog AI Notification System & Executive PDF Generator

This module provides automated insight delivery via email and generates executive-level weekly summaries using Streamlit PDF export functionality.

## Features

### Email Delivery System (Backend + Templates)

- **Modular SMTP/SendGrid Backend**: Flexible email delivery with proper authentication
- **Retry Logic + Queue Support**: Reliable delivery with exponential backoff and failed message handling
- **Daily/Weekly Scheduling**: Schedule reports to be delivered at regular intervals
- **Jinja2 Template Rendering**: Beautiful HTML email templates rendered from insight JSON

### Executive PDF Summary Engine

- **PDF Generation from Streamlit UI**: Convert Streamlit components to PDF reports
- **Multiple Output Formats**: Generate reports in PDF, HTML, CSV formats
- **Fallback Rendering**: If wkhtmltopdf fails, automatically falls back to WeasyPrint
- **Rich Visualization**: Include charts, metrics, tables and insights in a professional layout

### Insight Classification Engine

- **Priority Classification**: Tags insights as "critical", "recommended", "optional" or "informational"
- **Insight Type Detection**: Categorizes insights as alerts, trends, anomalies, etc.
- **Heuristic Analysis**: Uses metrics and text patterns to determine importance
- **Persistent Storage**: Store insights with their classifications for future use

## Usage

### Email Notifications

```python
from src.scheduler.notification_service import NotificationService

# Initialize notification service
service = NotificationService()

# Send an insight email
service.send_insight_email(
    recipients=["user@example.com"],
    insights=[
        {
            "title": "Sales Performance Insight",
            "summary": "Sales have increased by 15% this month",
            "metrics": [
                {"label": "Monthly Sales", "value": "$1.2M"},
                {"label": "Increase", "value": "15%"}
            ],
            "recommendations": [
                "Continue current marketing strategy",
                "Focus on high-performing vehicle segments"
            ]
        }
    ],
    subject="Daily Insights"
)

# Send an alert for critical issues
service.send_alert_email(
    recipients=["manager@example.com"],
    alert={
        "title": "Inventory Alert",
        "description": "25 vehicles over 90 days in inventory",
        "metrics": [
            {"label": "Aging Vehicles", "value": "25"},
            {"label": "Capital Tied", "value": "$875,000"}
        ],
        "recommendations": [
            "Review pricing strategy",
            "Consider promotional event"
        ]
    }
)
```

### PDF Generation

```python
from src.pdf_generator import generate_executive_pdf

# Generate an executive summary PDF
pdf_path = generate_executive_pdf(
    data={
        "dealership_name": "Your Dealership",
        "report_type": "Weekly Executive Summary",
        "total_sales": 128,
        "total_revenue": "1,245,500",
        "avg_gross": "3,240",
        "insights": [...],  # List of insights
        "kpis": [...],      # List of KPIs
        "top_performers": [...]  # Top performers
    },
    output_file="weekly_report.pdf"
)
```

### Insight Classification

```python
from src.insight_tagger import InsightTagger, tag_insights, InsightStore

# Tag a single insight
tagger = InsightTagger()
tagged_insight = tagger.tag_insight({
    "summary": "Sales have decreased significantly in the past week.",
    "metrics": {
        "sales_change_pct": -25.5,
        "profit_delta": -15000
    },
    "recommendations": [
        "Implement urgent marketing campaign",
        "Review pricing strategy"
    ]
})

# Categorizes as: CRITICAL priority, TREND type

# Tag multiple insights and store them
insights = [...]  # List of insights
tagged_insights = tag_insights(insights, save=True)

# Retrieve insights by priority
store = InsightStore()
critical_insights = store.get_insights_by_priority("critical")
```

## Demo Script

A demonstration script is provided to showcase the functionality:

```bash
# Run all demos with a test email
python src/notification_demo.py --all --email user@example.com

# Generate PDF only
python src/notification_demo.py --pdf

# Classify insights
python src/notification_demo.py --classify
```

## Components

1. **Email Notification System**
   - `NotificationService`: Main class for sending notifications
   - `EmailMessage`: Represents an email to be sent
   - `NotificationQueue`: Queue for processing emails with retries

2. **PDF Generator**
   - `PDFGenerator`: Core class for generating PDFs
   - `generate_executive_pdf()`: Helper for creating executive summaries
   - `generate_from_streamlit_ui()`: Convert Streamlit components to PDF

3. **Insight Tagger**
   - `InsightTagger`: Classifies insights by priority and type
   - `InsightStore`: Persistent storage for tagged insights
   - `tag_insights()`: Helper for tagging multiple insights

## Testing

Unit tests are provided for all components:

```bash
# Run email notification tests
python -m unittest tests.unit.test_notification_service

# Run PDF generator tests
python -m unittest tests.unit.test_pdf_generator

# Run insight tagger tests
python -m unittest tests.unit.test_insight_tagger
```