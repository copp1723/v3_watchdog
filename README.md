[![CI](https://github.com/copp1723/v3_watchdog/actions/workflows/ci.yml/badge.svg)](https://github.com/copp1723/v3_watchdog/actions/workflows/ci.yml)

# Watchdog AI - Dealership Analytics

Intelligent analytics platform for automotive dealerships, powered by AI.

## Features

### 🔍 Smart Insights
- Natural language queries about your dealership data
- AI-powered analysis of sales, leads, and inventory
- Interactive visualizations and downloadable reports

![Lead Source Analysis](assets/tech_grid.png)

### 💬 Guided Analysis
Try asking questions like:
- "What was our highest-value lead source last month?"
- "Show me our top-performing vehicles by revenue"
- "Compare website leads vs walk-in performance"

### 📊 Data Validation
- Automatic schema validation
- Data quality scoring
- Detailed validation reports
- Support for CSV and Excel files

### 📱 Mobile-First Design
- Responsive layout for all devices
- Touch-friendly interface
- Optimized performance

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/watchdog-ai.git
cd watchdog-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd src
streamlit run app.py
```

4. Open your browser to http://localhost:8501

## Data Requirements

Your data file should include these required columns. The system supports multiple column name formats:

### Sale Date (Any of these formats):
- SaleDate (preferred)
- sale_date
- Sale_Date
- SALE_DATE
- date
- sale date

### Sale Price:
- SalePrice (preferred)
- sale_price
- Sale_Price
- SALE_PRICE
- price
- sale price

### Vehicle VIN:
- VIN (preferred)
- vehicle_vin
- VehicleVIN
- Vehicle_VIN
- VEHICLE_VIN
- vin

### Total Gross:
- TotalGross (preferred)
- total_gross
- Total_Gross
- TOTAL_GROSS
- Gross_Profit
- gross profit

### Lead Source:
- LeadSource (preferred)
- lead_source
- Lead_Source
- LEAD_SOURCE
- source
- lead source

## Sample Data

Here's an example of a valid CSV file format:

```csv
LeadSource,TotalGross,VIN,SaleDate,SalePrice
Website,2500.00,1HGCM82633A123456,2024-01-01,25000.00
CarGurus,3000.00,1HGCM82633A123457,2024-01-02,30000.00
Walk-in,2800.00,1HGCM82633A123458,2024-01-03,28000.00
```

You can find this sample data in `examples/sample_data.csv`. Use this as a template for formatting your own data files.

## Development

### Architecture
- Streamlit for UI components
- Altair for interactive visualizations
- Pandas for data processing
- Modular design with reusable components

### Theme Customization
Theme constants are centralized in `src/ui/theme.py`. Customize colors, spacing, animations, and more.

### Performance
- Data loading and processing is cached
- Lazy loading for large datasets
- Optimized chart rendering

### Testing
- Unit tests cover core logic for validation, normalization, and utilities.
- E2E tests (`tests/e2e/`) cover key user flows, including:
  - Data upload and validation.
  - LLM-driven column mapping with user clarifications (`test_data_upload_llm_mapping.py`).
  - Insight generation and chat analysis.

### Column Mapping System
The platform features an advanced LLM-driven column mapping system that:

- Uses "Jeopardy-style" reasoning to semantically map dataset columns to a canonical schema
- Handles ambiguous column names through interactive user clarifications
- Supports Redis caching for improved performance and reduced LLM API costs
- Identifies lead source value columns that appear as headers
- Provides confidence scores for each mapping decision
- Can optionally drop unmapped columns after confirmation (configurable)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Data Format

The application expects CSV or Excel files with the following columns:
- `date`: Date of the sale (YYYY-MM-DD format)
- `lead_source`: Source of the lead (e.g., cargurus, autotrader)
- `sales_rep`: Name of the sales representative
- `vehicle_model`: Model of the vehicle sold
- `sale_price`: Price of the sale (numeric)

### Supported Header Variants

The application attempts to normalize column headers. It recognizes common variations (case-insensitive, ignoring spaces and underscores) for canonical columns. Examples:

- **LeadSource:** `leadsource`, `lead source`, `lead_source`
- **LeadSource Category:** `leadsource category`, `lead_source_category`, `category`
- **DealNumber:** `dealnumber`, `deal number`, `deal_number`, `deal id`
- **SellingPrice:** `sellingprice`, `selling price`, `sale price`, `sale_price`, `price`
- **FrontGross:** `frontgross`, `front gross`, `front_gross`, `frontend gross`
- **BackGross:** `backgross`, `back gross`, `back_gross`, `backend gross`, `f&i gross`
- **Total Gross:** `total gross`, `total_gross`, `gross`, `totalgross`
- **SalesRepName:** `salesrepname`, `sales rep name`, `sales_rep_name`, `salesperson`, `sales rep`, `sales_rep`
- **SplitSalesRep:** `splitsalesrep`, `split sales rep`, `split_sales_rep`, `split salesperson`
- **VehicleYear:** `vehicleyear`, `vehicle year`, `vehicle_year`, `year`
- **VehicleMake:** `vehiclemake`, `vehicle make`, `vehicle_make`, `make`
- **VehicleModel:** `vehiclemodel`, `vehicle model`, `vehicle_model`, `model`
- **VehicleStockNumber:** `vehiclestocknumber`, `vehicle stock number`, `vehicle_stock_number`, `stock number`, `stock_number`, `stock #`
- **VehicleVIN:** `vehiclevin`, `vehicle vin`, `vehicle_vin`, `vin`
- **Sale_Date:** `sale_date`, `sale date`, `saledate`, `date`, `deal date`

*Note: If multiple headers map to the same canonical name, the first one encountered is typically used.*

## Project Structure

- `src/`: Source code
  - `assets/`: Static assets
    - `watchdog_logo.png`: Application logo (New)
    - `tech_grid.png`: Background image

## Branding

The application logo is located at `assets/watchdog_logo.png`.
The header uses Streamlit's native `st.columns` and `st.image` for layout.

*(Add screenshot of the new header here)*

## Recent Updates

### Column Mapping Enhancements
- Implemented Redis caching for column mapping to improve performance and reduce API costs
- Added configuration option to automatically drop unmapped columns
- Enhanced user interaction for column mapping clarifications
- Extended Redis connection handling with better error recovery
- Added cache statistics tracking for monitoring cache hit/miss rates


## Containerized Dev Workflow

To build and run the application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t v3_watchdog .
    ```

2.  **Run the container:**
    ```bash
    # Make sure you have a .env file in the root directory or pass variables with -e
    docker run -p 8501:8501 --env-file .env v3_watchdog
    ```
    *   Replace `.env` with the path to your environment file if it's named differently or located elsewhere.
    *   You can also pass environment variables directly using the `-e` flag (e.g., `-e SENTRY_DSN=your_dsn`).

Alternatively, use Docker Compose for a simpler setup:

1.  **Build and run services:**
    ```bash
    # Ensure you have a .env file in the root directory for environment variables
    docker-compose up --build
    ```
    This command reads the `docker-compose.yml` file, builds the services.

## Configuration

The application supports the following environment variables:

```
OPENAI_API_KEY=your-openai-key-here
USE_MOCK=true  # Set to false to use real LLM API
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100

# Redis caching for column mapping
REDIS_CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
COLUMN_MAPPING_CACHE_TTL=86400  # 24 hours
COLUMN_MAPPING_CACHE_PREFIX=watchdog:column_mapping:

# Column mapping settings
DROP_UNMAPPED_COLUMNS=false  # Set to true to automatically drop unmapped columns

# AgentOps monitoring (optional)
AGENTOPS_API_KEY=your-agentops-key-here
```