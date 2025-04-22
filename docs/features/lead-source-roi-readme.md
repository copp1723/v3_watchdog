# Lead Source ROI and Admin Preferences System

This module adds lead source ROI analysis and admin notification preferences to the Watchdog AI platform.

## Features

### Lead Source ROI Analysis

- **Data Ingestion & Normalization**: Provides robust ingestion with automatic normalization of inconsistent lead source names (e.g., "Website" vs "Web Lead")
- **ROI Calculation**: Implements standard ROI formula (Revenue - Cost) / Cost with handling of edge cases like zero-cost sources
- **Timeframe Filters**: Supports weekly and monthly calculations for trend analysis
- **Validation**: Integrates with the existing validation system to ensure data quality
- **Visualization**: Includes bar charts, ROI ranking, trend visualization, and interactive tooltips

### Admin Preferences System

- **Notification Preferences**: Configure delivery frequency (daily, weekly, monthly)
- **Content Preferences**: Select specific insight types to receive
- **Recipient Management**: Add/remove email recipients
- **Test Functionality**: Send test notifications to verify configuration
- **Persistent Storage**: Save preferences to file for consistent application

## Components

The solution consists of several key components:

1. `lead_source_roi.py`: Core ROI calculation and data processing logic
2. `roi_visualizer.py`: Visualization components for ROI metrics
3. `admin_preferences.py`: Admin notification preferences management
4. `lead_source_roi_dashboard.py`: Streamlit UI for ROI analysis
5. Integration with existing app structure via `main_app.py`

## Usage

The functionality is integrated into the main Watchdog AI application. Launch the application with:

```
python app.py
```

Then navigate to the "Lead Source ROI" and "Admin" tabs in the user interface.

## Lead Source ROI Dashboard

The ROI dashboard provides comprehensive analysis features:

- **ROI Overview**: Summary metrics and ROI comparison chart
- **Source Cost Management**: Add/update costs for lead sources
- **Trend Analysis**: Historical ROI performance over time
- **Export Tools**: Export analysis results to CSV or JSON

## Admin Preferences

The admin preferences page enables:

- **Delivery Settings**: Configure notification frequency and delivery methods
- **Insight Type Selection**: Choose which insights to include in reports
- **Recipient Management**: Add/remove notification recipients
- **Testing and Saving**: Test notification delivery and save preferences

## Technical Implementation

### Lead Source Normalization

The system uses pattern matching and fuzzy matching to standardize inconsistent source names, enabling accurate aggregation and comparison.

```python
LeadSourceNormalizer().normalize("Web Lead")  # Returns "website"
LeadSourceNormalizer().normalize("CarGurus")  # Returns "cargurus"
```

### ROI Calculation

The ROI calculation handles edge cases like zero-cost sources (returns infinity) and negative values:

```python
# Standard calculation:
roi = (revenue - cost) / cost if cost > 0 else float('inf')
```

### Data Model

The lead source ROI schema defines required fields:
- `LeadSource`: Source of the lead
- `LeadDate`: Date the lead was received
- `LeadCount`: Number of leads
- `LeadCost`: Cost of the lead (optional)
- `Revenue`: Revenue generated (optional)
- `Closed`: Whether the lead resulted in a sale (optional)

## Testing

The implementation includes comprehensive unit tests for:
- Lead source name normalization
- ROI calculation logic
- Edge case handling
- Data processing workflows

## Future Enhancements

Potential future improvements:
- **Attribution models**: Add support for first-click, last-click, and multi-touch attribution
- **Campaign integration**: Connect with ad campaign data from Google, Meta, etc.
- **ML-based forecasting**: Predict future ROI based on historical patterns
- **Notification automation**: Schedule automatic delivery of ROI reports