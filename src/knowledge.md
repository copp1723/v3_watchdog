# Watchdog AI Knowledge Base

## Architecture

The system follows a three-phase pipeline:

1. Schema & Validation
   - Required sheets: sales (required), inventory, leads
   - Core required columns: gross, lead_source
   - Optional columns with fallbacks: date, sales_rep, vin
   - Automatic date column creation if missing
   - Validation before any analysis

2. Core Insight Engine
   - Direct pandas operations (no LLM)
   - Keyword-driven analysis
   - Structured InsightResult output
   - Automatic currency handling (strips $, commas)
   - Numeric conversion with error handling

3. Streamlit UI
   - Simple file upload and validation
   - Chat-like question interface
   - Chart visualization
   - Insight history

## Data Requirements

### Sales Sheet (Required)
Required columns:
- gross: total_gross, front_gross, gross_profit (numeric or currency format)
- lead_source: leadsource, source, lead_type

Optional columns with fallbacks:
- date: sale_date, transaction_date, deal_date (auto-created if missing)
- sales_rep: salesperson, rep_name, employee (defaults to "Unknown")
- vin: vin_number, vehicle_id (optional)

### Data Handling
- Currency values: Automatically strips $ and , characters
- Numeric conversion: Uses pd.to_numeric with errors='coerce'
- Invalid values: Replaced with NaN and excluded from calculations
- String normalization: Lowercase for matching, Title case for display

### Inventory Sheet (Optional)
- vin: vin_number, vehicle_id
- days_in_stock: age, days_on_lot
- price: list_price, asking_price, msrp

### Leads Sheet (Optional)
- date: lead_date, inquiry_date
- source: lead_source, origin, channel
- status: lead_status, state

## Supported Questions

The system can analyze:
1. Lead source performance
   - "How many sales from CarGurus?"
   - "Show me AutoTrader performance"
   - "Compare all lead sources"

2. Sales rep metrics
   - "Who are the top sales reps?"
   - "Show me bottom performers"
   - "Sales rep leaderboard"

3. Gross profit analysis
   - "Show me negative gross deals"
   - "Average gross by lead source"
   - "Gross profit trends"

4. Time-based analysis
   - "How many sales today?"
   - "Show me last week's performance"
   - "This month's gross profit"

## Response Format

All insights include:
- Title: Clear description of the analysis
- Summary: One-line overview with key metrics
- Metrics: Detailed numbers and calculations
- Chart: Visual representation when applicable
- Recommendations: Action items based on findings