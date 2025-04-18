 # Watchdog AI – Dealership Analytics

A smart, AI-powered insights platform for automotive dealerships. Supports natural language queries, data validation, visual dashboards, and compliance-focused features like TTL enforcement.

✅ GDPR/CCPA-ready  
✅ Mobile-first  
✅ Dockerized for easy deployment  

---

![CI](https://github.com/yourusername/watchdog-ai/actions/workflows/ci.yml/badge.svg)

---

## Features

### 🔍 Smart Insights
- Natural language queries about your dealership data
- AI-powered analysis of sales, leads, and inventory
- Interactive visualizations and downloadable reports
- Lead Source Analysis

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

---

## 🚀 Getting Started (Local Dev)

### Clone the repository:
```bash
git clone https://github.com/yourusername/watchdog-ai.git
cd watchdog-ai
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application:
```bash
cd src
streamlit run app.py
```
Then open: [http://localhost:8501](http://localhost:8501)

---

## 🔍 Data Requirements

### Required Columns:
The system supports multiple column name formats. The following are expected (canonical names):

| Field          | Examples (case-insensitive)                                      |
|----------------|------------------------------------------------------------------|
| Sale Date      | SaleDate, sale_date, date, deal date                            |
| Sale Price     | SalePrice, sale_price, price                                     |
| VIN            | VIN, vehicle_vin, VehicleVIN                                     |
| Total Gross    | TotalGross, gross, front_gross, backend gross                   |
| Lead Source    | LeadSource, lead_source, source                                  |


### Sample CSV Format
```csv
LeadSource,TotalGross,VIN,SaleDate,SalePrice
Website,2500.00,1HGCM82633A123456,2024-01-01,25000.00
CarGurus,3000.00,1HGCM82633A123457,2024-01-02,30000.00
Walk-in,2800.00,1HGCM82633A123458,2024-01-03,28000.00
```
File: `examples/sample_data.csv`

---

## 📈 Development

### Architecture
- **Streamlit** for UI
- **Altair** for visualizations
- **Pandas** for data processing
- Modular & reusable component structure

### Theming
Customize styles via: `src/ui/theme.py`

---

## 🚀 Performance
- Caching for data load & processing
- Lazy loading of large datasets
- Efficient chart rendering

---

## 🚧 CI & Validation
- CI via GitHub Actions
- Core test: `tests/unit/test_session_ttl.py`
- Pydantic validation with warnings on deprecated style (see migration notes in code)

---

## 🌍 Containerized Dev Workflow

### Docker
```bash
docker build -t v3_watchdog .
docker run -p 8501:8501 --env-file .env v3_watchdog
```

### Docker Compose
```bash
docker-compose up --build
```
Then visit: [http://localhost:8501](http://localhost:8501)

---

## 💼 Contributing
```bash
# Fork + clone repo
# Create branch:
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add your feature"

# Push and open PR
git push origin feature/your-feature-name
```

---

## 📚 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📁 Project Structure

```
.
├── src/
│   ├── app.py               # Streamlit app
│   ├── insights/            # Insight engine
│   ├── validators/          # Schema validation
│   ├── ui/                  # Theming + layout
├── tests/
│   └── unit/test_session_ttl.py   # TTL enforcement test
├── examples/sample_data.csv
├── assets/watchdog_logo.png
└── .github/workflows/ci.yml
```

---

## 🌐 Branding
- Logo: `assets/watchdog_logo.png`
- Header layout: `st.columns` + `st.image`
 
