# Global Environmental Trends (2000–2024)

Interactive analytics dashboard and reproducible notebooks for climate signals, risk indicators, and explainable forecasting.
Stakeholder-focused, transparent, and fully documented for Code Institute Data Analytics capstone.

---

## Assessment Alignment Checklist

- **LO1.1 Ethics, privacy, governance:** Documented in README and dashboard notes; transparent modeling, versioned outputs, and clear limitations.
- **LO1.2 Legal and social implications:** Licensing, GDPR principles, and responsible use are covered in README.
- **LO2.1 Insight communication:** Narrative captions, plain-language summaries, and stakeholder-focused commentary in dashboard and notebooks.
- **LO2.2 Visualisation and narratives:** Minimum four plot types, guided dashboard flow, and accessible color palette.
- **LO2.3 Documentation:** README sections, notebook structure, and code comments align with analysis stages.
- **LO3.1 Project plan:** Implementation, maintenance, updates, and evaluation included in README and project structure.
- **LO3.2 Reflection:** Challenges, limitations, and learning journey recorded in README and notebook markdown.

All checklist items are addressed and stipulated in the README, notebooks, and dashboard.

---

## Business Requirements

### Context

Stakeholders want a clear view of climate risk signals (temperature, emissions, extreme events) and a simple, explainable projection to support awareness and planning. The solution must be accessible to non-technical users and transparent about limitations.

### Stakeholder Needs

**Primary Stakeholders:**
- Public and students: accessible insights and context
- Sustainability teams: trend awareness and risk indicators
- Analysts and policy teams: correlations and hypothesis validation

### Business Requirements

**BR1: Temperature trend awareness**
- Show how average temperatures changed from 2000 to 2024 by country
- **Success Criteria:** clear trend visuals and summary metrics

**BR2: Emissions relationship**
- Assess association between CO2 emissions per capita and average temperature
- **Success Criteria:** correlation findings with clear caveats

**BR3: Climate risk indicators**
- Track extreme weather events over time and identify high-count countries
- **Success Criteria:** ranking chart and time trends

**BR4: Mitigation signals**
- Assess association between renewable energy percent and CO2 emissions
- **Success Criteria:** correlation findings and stakeholder interpretation

**BR5: Environmental resilience**
- Explore forest area percent vs extreme events or rainfall
- **Success Criteria:** exploratory evidence with limitations

**BR6: Forecasting**
- Provide a short-term, explainable temperature projection (3-5 years)
- **Success Criteria:** time-aware split, MAE/RMSE reported, forecast export

### User Stories

**US1:** As a policy analyst, I want to see long-term temperature trends by country, so I can communicate warming patterns clearly.

**US2:** As a sustainability lead, I want to compare emissions and renewable energy indicators, so I can discuss mitigation progress.

**US3:** As a non-technical user, I want a simple dashboard view, so I can explore climate data without coding.

**US4:** As a researcher, I want transparent hypothesis tests, so I can evaluate the evidence behind each claim.

---

## Project Objectives

1. Load and clean the environmental dataset with quality checks
2. Perform EDA to identify trends and correlations
3. Validate hypotheses using simple statistical tests
4. Train and evaluate an explainable forecasting model
5. Deliver a Streamlit dashboard for stakeholder exploration
6. Document findings, limitations, and next steps

---

## Project Hypotheses

- **H1:** Higher CO2 emissions per capita are associated with higher average temperature (association, not causation)
- **H2:** Higher renewable energy percent is associated with lower or stabilizing CO2 emissions per capita
- **H3:** Extreme weather events increase over time from 2000 to 2024
- **H4 (optional):** Higher forest area percent is associated with fewer extreme events

---

## Dataset Description

**Source:** Kaggle - Global Environmental Trends 2000-2024

**Records:** 156 rows across 19 countries (2000-2024)

**Features (selected):**
- `Year`
- `Country`
- `Avg_Temperature_degC`
- `CO2_Emissions_tons_per_capita`
- `Sea_Level_Rise_mm`
- `Rainfall_mm`
- `Population`
- `Renewable_Energy_pct`
- `Extreme_Weather_Events`
- `Forest_Area_pct`

---

## Project Structure

```
global_env_trend/
├── app.py
├── data/
│   ├── raw/
│   │   └── v1/
│   │       └── environmental_trends.csv
│   └── processed/
│       └── v1/
│           ├── environmental_trends_clean.csv
│           └── model_predictions.csv
├── jupyter_notebooks/
│   ├── 01_data_ingestion_quality_checks.ipynb
│   ├── 02_eda_descriptive_stats.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   └── 04_predictive_modeling.ipynb
├── Procfile
├── README.md
├── requirements.txt
└── setup.sh
```

---

## Key Findings

### Hypothesis Testing

**H1: CO2 Emissions vs Temperature**
- Result: moderate negative association (see notebook for exact correlation)
- Interpretation: likely reflects climate zone and development differences, not causality

**H2: Renewable Energy vs CO2 Emissions**
- Result: moderate negative association
- Interpretation: higher renewables are associated with lower emissions per capita, but not causal proof

**H3: Extreme Weather Events Trend**
- Result: increasing trends in most countries, with variability
- Interpretation: supports climate risk awareness, with reporting bias caveats

**H4: Forest Area vs Extreme Events**
- Result: weak association
- Interpretation: forest coverage alone is not a strong predictor at country level

### Modeling

- Per-country linear regression with a time-aware split
- Forecasts exported to `data/processed/v1/model_predictions.csv`
- Metrics reported in the modeling notebook

---

## Installation and Setup

### Prerequisites
- Python 3.12.x
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd global_env_trend
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Jupyter Notebooks

Open and run the notebooks in order:

```bash
jupyter notebook jupyter_notebooks/01_data_ingestion_quality_checks.ipynb
```

### 2. Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Or in Windows with a full path to Python:

```bash
".venv/Scripts/python.exe" -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Technologies Used

**Core:**
- Python 3.12.x
- Jupyter Notebook

**Data Processing:**
- pandas
- numpy

**Visualization:**
- matplotlib
- seaborn
- plotly

**Machine Learning:**
- scikit-learn

**Dashboard:**
- Streamlit

---

## Methodology

### 1. Data Preparation
- Load raw data and run quality checks
- Save a cleaned, versioned dataset

### 2. Exploratory Data Analysis
- Descriptive statistics
- Trend charts and correlation scan

### 3. Hypothesis Testing
- Correlations and simple, explainable statistics
- Plain-language interpretations with limitations

### 4. Forecast Modeling
- Per-country linear regression on Year
- Time-aware split for evaluation
- Forecast exports for 2025-2029

---

## Results and Performance

### Model Performance

The forecasting model is intentionally simple and explainable. Performance metrics (MAE, RMSE) are reported in the modeling notebook and displayed in the Streamlit app. These values should be interpreted as short-term, trend-based accuracy rather than long-term climate projections.

### Dashboard Outcomes

- Stakeholders can filter by country and year
- Key signals are summarized in KPIs
- Associations are visualized without causal claims
- Forecasts are clearly labeled as projections

---

## Learning Journey and Reflections

**Key Learnings:**
- Transparent data quality checks are essential for trust
- Correlation needs careful interpretation in climate context
- Simple models are more explainable for public-facing insights
- Streamlit enables rapid, stakeholder-friendly delivery

---

## Version Control and Project Management

All data outputs are versioned under `data/raw/v1` and `data/processed/v1` to ensure reproducibility and auditability.

---

## Future Improvements

- Add confidence intervals for forecasts
- Expand coverage beyond 19 countries
- Compare linear regression with more flexible time-series models
- Add scenario-based forecasting using policy or emissions inputs
- Include batch prediction or CSV upload in the dashboard

---

## Credits

**Dataset:** [Kaggle - Global Environmental Trends 2000-2024](https://www.kaggle.com/datasets/adilshamim8/temperature)

**Author:** Sergio Kadje (Code Institute Data Analytics Capstone Project)

**Institution:** Code Institute

**Date:** February 2026

**AI Assistance:** GitHub Copilot used for code suggestions and documentation improvements.

- Python 3.12 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sergionzeutcho-rgb/global_env_trend.git
   cd global_env_trend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

The project contains four Jupyter notebooks in the `jupyter_notebooks/` folder, executed in sequence:

1. **01_data_ingestion_quality_checks.ipynb**
   - Loads raw data from `data/raw/v1/environmental_trends.csv`
   - Validates data types, missing values, and duplicates
   - Exports cleaned data to `data/processed/v1/environmental_trends_clean.csv`
   - Run first to ensure data quality

2. **02_eda_descriptive_stats.ipynb**
   - Performs exploratory analysis on the cleaned dataset
   - Computes summary statistics by country and year
   - Generates correlation heatmaps and trend visualizations
   - Builds intuition about relationships between variables

3. **03_hypothesis_testing.ipynb**
   - Tests all four hypotheses using Pearson correlation
   - Computes H1 (CO2–temperature), H2 (renewables–CO2), H3 (extreme events trend), H4 (forest–extreme events)
   - Outputs correlation coefficients and interpretation
   - Validates association claims with statistical caveats

4. **04_predictive_modeling.ipynb**
   - Builds per-country linear regression models
   - Splits data at year 2018 (train 2000–2018, test 2019–2024)
   - Calculates MAE and RMSE on test set
   - Forecasts 2025–2029 temperatures and exports to `data/processed/v1/model_predictions.csv`

**To run all notebooks:**

From the repository root, use the Makefile or shell script (if available):
```bash
bash setup.sh
```

Or open Jupyter and execute notebooks in sequence:
```bash
jupyter notebook jupyter_notebooks/
```

### Output Locations

- **Cleaned data:** `data/processed/v1/environmental_trends_clean.csv`
- **Model predictions:** `data/processed/v1/model_predictions.csv`
- **Dashboard data source:** Use `environmental_trends_clean.csv` in Tableau

### Viewing the Dashboard

The Tableau project is configured to use the processed dataset. To view or edit:

1. Open the Tableau workbook (available via Tableau Public link in this README once published)
2. Or, download the `.twbx` file and open locally in Tableau Public/Desktop
3. Use filters to explore trends by country and year range

### Reproducing the Analysis

To ensure reproducibility:

1. Start with `data/raw/v1/environmental_trends.csv` (original Kaggle source)
2. Run notebooks 01–04 in sequence
3. Verify output files are generated in `data/processed/v1/`
4. Compare results with provided outputs (hypothesis correlations and model metrics)

If results differ, check for:
- Different Python or package versions (see [requirements.txt](requirements.txt))
- Floating-point precision (expected ±0.0001 tolerance)
- Dataset updates (regenerate if new data is added to raw folder)

### Implementation

- Ingest data and store raw versioned files
- Clean and validate data, export processed datasets
- Run EDA and hypothesis checks
- Build a simple forecasting model
- Export predictions and build the BI dashboard

### Updates

- Add new-year data to data/raw/v2
- Rerun notebooks and export data/processed/v2
- Version the dashboard extract and record changes

### Maintenance and Updates

- Add new-year data to data/raw/v2
- Rerun notebooks and export data/processed/v2
- Refresh dashboard extracts

### Evaluation

- Usability checks with 2 to 3 users
- Record feedback and iterate on labels, layout, and narrative
- Monitor model drift and schedule periodic retraining

## Challenges and Reflection (LO3.2)

- Limited time coverage (2000-2024)
- Possible gaps in country coverage and data completeness
- Missing contextual variables (GDP, policy changes, energy mix)
- Modelling limitations due to short time series per country

## UX, Accessibility, and Dashboard Standards

- Information hierarchy: titles, filters, and KPIs are placed to guide attention from summary to detail
- Consistency: a single color palette for emissions, renewables, and temperature across all charts
- Accessibility: colorblind-friendly palette, high-contrast labels, and avoid red-green only cues
- User control: filters (country, year range) with reset buttons and clear tooltips

## How AI Was Used

- Ideation and design thinking
- Drafting business requirements and hypotheses
- Structuring the README and dashboard narrative

## Dashboard Link

- Tableau Public: (to be added after publishing)

## Credits and References

- Code Institute Data Analytics capstone template
- Kaggle dataset: Global Environmental Trends 2000-2024
