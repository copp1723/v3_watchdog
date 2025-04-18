```markdown
[![CI](https://github.com/copp1723/v3_watchdog/actions/workflows/ci.yml/badge.svg)](https://github.com/copp1723/v3_watchdog/actions/workflows/ci.yml)

# Watchdog AI – Dealership Analytics

![Version](https://img.shields.io/badge/version-0.1.0--alpha-blue)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![License](https://img.shields.io/badge/license-MIT-green)

Intelligent analytics platform for automotive dealerships, powered by AI.

---

## 📚 Documentation

- **Getting Started**: [docs/onboarding.md](docs/onboarding.md)  
- **Infrastructure**: [docs/infra.md](docs/infra.md)  
- **Security & Compliance**: [docs/infra.md#backup-and-retention](docs/infra.md#backup-and-retention)  
- **Development Roadmap**: [docs/ROADMAP.md](docs/ROADMAP.md)  
- **API Reference**: [docs/api.md](docs/api.md)  
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)  

---

## 🚀 Features

### 🔍 Smart Insights
- Natural language querying of dealership data  
- AI-powered analysis of sales, leads, and inventory  
- Interactive visualizations and downloadable reports  

![Lead Source Analysis](assets/tech_grid.png)

### 💬 Guided Analysis
Ask questions like:  
- “What was our highest-value lead source last month?”  
- “Show me our top-performing vehicles by revenue”  
- “Compare website leads vs walk-in performance”  

### 📊 Data Validation
- Automatic schema validation  
- Data quality scoring  
- Detailed validation reports  
- Support for CSV and Excel  

### 📱 Mobile‑First Design
- Responsive layout for all devices  
- Touch-friendly interface  
- Optimized performance  

---

## 🏁 Getting Started

### Method 1: Local Python Setup

```bash
git clone https://github.com/copp1723/v3_watchdog.git
cd v3_watchdog
pip install -r requirements.txt
streamlit run src/ui/streamlit_app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

### Method 2: Docker (Recommended)

```bash
git clone https://github.com/copp1723/v3_watchdog.git
cd v3_watchdog
docker-compose up --build
```

Browse to [http://localhost:8501](http://localhost:8501).  
To stop: `docker-compose down`

---

## 📂 Data Requirements

Your CSV/Excel must include these core fields (multiple header variants supported):

| Canonical       | Variants                                     |
| --------------- | -------------------------------------------- |
| **Sale Date**   | SaleDate, sale_date, date, sale date         |
| **Sale Price**  | SalePrice, sale_price, price, sale price     |
| **Vehicle VIN** | VIN, vehicle_vin, Vehicle_VIN                |
| **Total Gross** | TotalGross, total_gross, Gross_Profit        |
| **Lead Source** | LeadSource, lead_source, source              |

*(See full list in README’s “Supported Header Variants” section below.)*

---

## 📝 Sample Data

```csv
LeadSource,TotalGross,VIN,SaleDate,SalePrice
Website,2500.00,1HGCM82633A123456,2024-01-01,25000.00
CarGurus,3000.00,1HGCM82633A123457,2024-01-02,30000.00
Walk-in,2800.00,1HGCM82633A123458,2024-01-03,28000.00
```

Use `examples/sample_data.csv` as a template.

---

## 🛠 Development

### Architecture
- **UI**: Streamlit, Altair  
- **Data**: Pandas, modular processing  
- **AI**: LLM integration for summarization & insights  

### Theme
Centralized constants in `src/ui/theme.py`.

### Performance
- Cached data loading  
- Lazy evaluation for large datasets  
- Optimized chart rendering  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a branch: `git checkout -b feature/xyz`  
3. Commit: `git commit -m "Add xyz"`  
4. Push: `git push origin feature/xyz`  
5. Open a PR  

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🔄 Supported Header Variants

The system normalizes headers (case‑insensitive, spaces/underscores ignored):

- **LeadSource**: `leadsource`, `lead source`, `lead_source`  
- **SellingPrice**: `sellingprice`, `sale_price`, `price`  
- **SalesRepName**: `salesrepname`, `sales rep`, `sales_rep`  
- **VehicleModel**: `vehiclemodel`, `vehicle model`, `model`  
- **Sale_Date**: `sale_date`, `saledate`, `deal date`  
*(…plus others defined in code.)*

---

## 🗂 Project Structure

```
├── assets/           # Static images (logo, tech_grid.png)
├── config/           # Normalization rules, etc.
├── docs/             # Markdown docs & guides
├── examples/         # Sample data
├── src/
│   ├── insights/     # Insight logic & prompts
│   ├── ui/           # Streamlit UI code
│   ├── utils/        # I/O, normalization, audit, session
│   ├── validators/   # Validation modules & registry
│   └── scheduler/    # Report scheduling components
├── tests/            # Unit & E2E tests
├── docker-compose.yml
├── Dockerfile
├── README.md
└── CHANGELOG.md
```

---

## 🆕 Recent Updates

### v0.1.0‑alpha (April 2023)
- Initial alpha release  
- Core framework & component architecture  
- LLM‑powered insight system  
- Modular report scheduler  
- Nova ACT DMS integration  

See [CHANGELOG.md](CHANGELOG.md) for full history.

### Containerized Dev Workflow

1. **Build Docker image**  
   ```bash
   docker build -t v3_watchdog .
   ```
2. **Run container**  
   ```bash
   docker run -p 8501:8501 --env-file .env v3_watchdog
   ```
3. **Or with Docker Compose**  
   ```bash
   docker-compose up --build
   ```
4. **Access**: [http://localhost:8501](http://localhost:8501)  

---

*Feel free to reach out if you have any questions or need further guidance!*