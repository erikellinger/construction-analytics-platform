# Construction Project Performance and Predictive Analytics Platform

A full-stack data engineering and machine learning platform that transforms raw construction project data into actionable business intelligence. The platform ingests budget, schedule, labor, equipment, and safety records into a star-schema data warehouse, surfaces KPIs through an interactive multi-page Streamlit dashboard, and applies trained RandomForest models to predict cost overruns and equipment maintenance needs before they occur.

---

## Key Features

- Real-time project performance tracking with CPI, SPI, and budget variance across the active portfolio
- Equipment utilization analytics with ROI tracking and idle-fleet identification
- Subcontractor performance scorecards ranked by cost performance index and delivery rating
- Labor productivity analysis broken down by trade, certification level, and project type
- OSHA-compliant safety metrics including incident rate, lost-day calculations, and severity trending
- Machine learning models for cost overrun prediction and equipment maintenance forecasting
- Interactive six-page Streamlit dashboard with drill-down filtering and Plotly visualizations
- Star-schema SQLite data warehouse housing 167,000+ synthetic transaction records
- Pytest test suite covering ML model training, prediction, feature importance, and edge-case handling

---

## Machine Learning Models

### Cost Overrun Prediction

Predicts whether a construction project will exceed its budget by more than 10% before the project is halfway complete.

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 100% | 100% |
| Precision | 100% | 100% |
| Recall | 100% | 100% |
| F1 Score | 100% | 100% |

**Input features used at prediction time:**

| Feature | Description |
|---------|-------------|
| `early_phase_cpi` | Cost performance index during the first 25% of the planned timeline |
| `contract_value` | Total contract value in dollars |
| `project_type_enc` | Label-encoded project type (Residential / Commercial / Infrastructure) |
| `planned_duration` | Planned project length in calendar days |
| `square_footage` | Building size in square feet |

**Key insight:** Early-phase CPI (computed from only the first quarter of the project timeline) is the dominant predictor of final budget outcome. Projects with a CPI below 1.0 in the early phase consistently overrun, while projects with CPI above 1.0 finish under budget. This means budget risk can be flagged months before conventional reporting would surface a problem.

---

### Equipment Maintenance Forecasting

Predicts whether a piece of equipment will require maintenance within the next 30 days, using rolling usage and maintenance history snapshots.

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 100% | 68.75% |
| Precision | 100% | 75.00% |
| Recall | 100% | 75.00% |
| F1 Score | 100% | 75.00% |

**Input features used at prediction time:**

| Feature | Description |
|---------|-------------|
| `days_since_maintenance` | Days elapsed since the last recorded maintenance event |
| `maintenance_event_count` | Total number of maintenance events in the equipment's history |
| `avg_maint_cost_per_event` | Mean cost per maintenance event |
| `total_hours_used` | Cumulative operating hours |
| `recent_fuel_cost_trend` | Ratio of fuel cost in the last 90 days to the prior 90 days |
| `age_days` | Equipment age in days from purchase date |
| `fuel_cost_per_hour` | Cumulative fuel spend divided by cumulative hours |
| `avg_hours_per_day` | Average daily operating hours over the equipment's life |
| `equipment_type_enc` | Label-encoded equipment type |

**Key insight:** Days since last maintenance and cumulative maintenance event count are the strongest predictors of near-term service needs. A rising fuel cost trend serves as an early warning signal for degrading mechanical efficiency before a formal maintenance event is recorded.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Dashboard | Streamlit |
| Database ORM | SQLAlchemy 2.0+ |
| Database engine | SQLite (file-based, no server required) |
| Data manipulation | Pandas, NumPy |
| Visualizations | Plotly |
| Machine learning | scikit-learn (RandomForestClassifier, LabelEncoder) |
| Model persistence | joblib |
| Synthetic data | Faker |
| Testing | pytest |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Portfolio Overview** | High-level KPIs across all active projects: total contract value, aggregate CPI/SPI, schedule health, and budget variance trend charts |
| **Project Detail** | Drill-down view for a single project with month-by-month cost curves, earned value metrics, and change-order impact |
| **Equipment Analytics** | Fleet utilization heatmap, ROI by equipment type, fuel cost trends, and maintenance cost breakdown |
| **Labor and Subcontractors** | Trade-level productivity metrics, hours-by-project breakdown, and ranked subcontractor scorecards with CPI and delivery rating |
| **Safety and Compliance** | OSHA incident rate tracking, severity distribution, near-miss trending, inspection pass/fail rates by project and inspector |
| **ML Predictions** | Live cost overrun risk scores for the active portfolio and maintenance probability rankings for the active fleet, with feature importance charts and model performance metrics |

---

## Database Schema

The warehouse uses a classic star schema: six dimension tables define the business entities and one date spine; six fact tables record every measurable event.

### Dimension Tables

| Table | Description |
|-------|-------------|
| `dim_projects` | Project metadata: type, contract value, planned/actual dates, status, square footage |
| `dim_subcontractors` | Company name, trade specialty, contact info, rating, years in business |
| `dim_equipment` | Equipment type, purchase date, purchase cost, hourly rate, status |
| `dim_employees` | Employee trade, hourly rate, hire date, certification level, active status |
| `dim_materials` | Material category, unit of measure, unit cost, supplier |
| `dim_dates` | Date spine covering year, quarter, month, ISO week, day of week, weekend and holiday flags |

### Fact Tables

| Table | Grain | Key Measures |
|-------|-------|-------------|
| `fact_project_costs` | One row per project per cost entry per day | Actual amount, budgeted amount, cost category |
| `fact_labor_hours` | One row per employee per project per day | Hours worked, hourly rate (snapshotted) |
| `fact_equipment_usage` | One row per equipment piece per project per day | Hours used, fuel cost, maintenance cost |
| `fact_material_deliveries` | One row per delivery | Quantity, total cost, on-time vs. delayed status |
| `fact_safety_incidents` | One row per incident | Incident type, severity, days lost |
| `fact_inspections` | One row per inspection | Inspection type, pass/fail/conditional result |

All fact tables carry indexed foreign keys to their dimension tables. Enum columns are enforced through SQLAlchemy `Enum` types with uppercase string values.

---

## How to Run Locally

**Prerequisites:** Python 3.11+, pip

```bash
# 1. Clone the repository
git clone https://github.com/your-username/construction-analytics-platform.git
cd construction-analytics-platform

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize the database and generate synthetic data (~167,000 records)
python src/data_generator.py

# 5. Train both ML models and save artifacts to models/
python src/ml_models.py

# 6. Launch the Streamlit dashboard
streamlit run src/app.py

# 7. Run the test suite
pytest tests/ -v
```

The dashboard opens at `http://localhost:8501` by default. The ML Predictions page requires step 5 to be completed first.

---

## Business Insights Demonstrated

**Cost overrun early warning**
Early-phase CPI computed from only the first 25% of a project's planned timeline predicts the final budget outcome with 100% accuracy on held-out test data. Flagging at-risk projects this early gives project managers months of runway to tighten change-order controls, activate contingency reserves, and re-negotiate scope before costs escalate.

**Equipment maintenance optimization**
Predictive maintenance scoring for the active fleet reduces the risk of unplanned downtime by surfacing equipment that is statistically likely to require service within 30 days. The rising fuel cost trend feature captures mechanical degradation before it manifests as a formal work order, enabling proactive scheduling.

**Subcontractor performance management**
Subcontractors are ranked by a composite scorecard that combines CPI, on-time delivery rate, and client rating. Top performers with CPI greater than 1.0 and ratings above 4.5 are highlighted for preferred-vendor consideration on future projects.

**Safety compliance**
OSHA recordable incident rates, near-miss counts, and lost-day totals are computed per project and trended over time, giving safety managers the data needed for toolbox-talk prioritization and regulatory reporting.

---

## Testing

The test suite uses pytest with session-scoped fixtures that build an in-memory SQLite database and train both ML models exactly once per session, keeping the full run under five seconds.

```bash
pytest tests/ -v
```

| Test | Scenario |
|------|---------|
| `test_cost_overrun_model_trains` | Artifact keys, feature count, label encoder classes |
| `test_cost_overrun_predictions` | Valid probability range, prediction labels, feature dict |
| `test_maintenance_model_trains` | Artifact keys, feature count, equipment type encoding |
| `test_maintenance_predictions` | Valid probability, days estimate >= 1, feature dict |
| `test_saved_models_exist` | Both .pkl files written to disk and non-empty |
| `test_feature_importance` | Importances sum to 1.0, non-negative, sorted descending |
| `test_handles_missing_data` | Graceful fallback for projects/equipment with no records |

All 7 tests pass. The `conftest.py` seeds the test database with 8 completed projects (4 overrun, 4 under-budget) and 4 equipment pieces with 365 daily usage records each, providing enough labeled examples for a stratified train/test split without relying on the production database.

---

## Project Structure

```
construction-analytics-platform/
├── src/
│   ├── app.py              # Six-page Streamlit dashboard
│   ├── analytics.py        # SQL-based KPI and metric calculations
│   ├── database.py         # SQLAlchemy ORM models and star schema definition
│   ├── data_generator.py   # Synthetic data generation (Faker + domain logic)
│   └── ml_models.py        # Model training, evaluation, and prediction functions
├── tests/
│   ├── conftest.py         # Session-scoped fixtures: patched paths, test DB, trained models
│   └── test_ml_models.py   # 7 unit tests for ML training and prediction
├── models/                 # Saved .pkl model artifacts (git-ignored)
├── data/                   # SQLite database file (git-ignored)
├── notebooks/              # Exploratory analysis notebooks
├── logs/                   # Application logs (git-ignored)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## What I Learned

**Data warehouse design**
Designing the star schema required thinking carefully about grain, slowly changing dimensions, and which facts belonged in which table. Understanding the difference between additive, semi-additive, and non-additive measures shaped how KPIs were computed in `analytics.py`.

**Advanced SQL**
The analytics layer relies heavily on CTEs, window functions, and multi-table joins to compute metrics like earned value, rolling incident rates, and equipment ROI without pulling raw data into Python first.

**Feature engineering for time-series data**
The equipment maintenance model uses a monthly-snapshot approach to build a supervised dataset from a time-series. Engineering the fuel cost trend ratio and early-phase CPI required thinking about information leakage and ensuring that only data available at prediction time was used as features.

**Production-grade ML pipelines**
Both models are wrapped in functions that handle edge cases (missing data, unseen label encoder classes, empty result sets), serialize artifacts with metadata alongside the model object, and are testable in isolation from the database.

**End-to-end dashboard development**
Building the Streamlit app reinforced the importance of caching strategy: `@st.cache_resource` for model artifacts shared across sessions, `@st.cache_data(ttl=300)` for DataFrames that should refresh periodically, and avoiding re-execution of expensive queries on every interaction.

**Testing data pipelines**
Designing the test fixtures taught me to think about the minimum viable dataset needed to exercise every code path, how to isolate tests from the production database using in-memory SQLite, and how to use pytest's session scope to train ML models once rather than per test.

---

## Skills Demonstrated

**Technical**
- Python (data engineering, ORM, ML, testing)
- SQL (star schema, CTEs, window functions, aggregations)
- Machine learning (RandomForest, feature engineering, cross-validation, model serialization)
- Data visualization (Plotly Express, Plotly Graph Objects)
- Dashboard development (Streamlit multi-page apps, caching, session state)
- Database design (SQLAlchemy 2.0, SQLite, foreign keys, indexes, enum enforcement)
- Software engineering practices (fixtures, mocking, session-scoped test resources)

**Business / Domain**
- Earned value management (CPI, SPI, budget variance)
- Construction project lifecycle and cost control
- OSHA safety compliance metrics
- Equipment lifecycle costing and ROI analysis
- Subcontractor procurement and performance management
- Predictive maintenance concepts and business case

---

## Future Enhancements

- **Cloud deployment:** Host the dashboard on Streamlit Community Cloud or AWS with S3-backed artifact storage and RDS for the database
- **Real-time data ingestion:** Replace the synthetic generator with a webhook receiver or file-drop pipeline for live project management system exports
- **Additional ML models:** Schedule delay prediction, subcontractor default risk scoring, and material price escalation forecasting
- **Role-based access control:** Project-manager view vs. executive portfolio view with row-level security
- **Automated retraining:** Schedule model retraining on a cadence using a lightweight orchestrator such as Prefect or GitHub Actions
- **Export and reporting:** PDF report generation for weekly project status packages and OSHA 300 log exports
