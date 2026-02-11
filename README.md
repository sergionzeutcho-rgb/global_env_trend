# Global Environmental Trends (2000-2024)

Global Environmental Trends (2000-2024) is a data app/dashboard that helps non-technical users (public, students, local sustainability teams) and technical users (analysts, policy and ESG teams) understand climate risk signals, emissions, and the renewable transition. It combines exploratory analysis, hypothesis checks, and a simple forecasting model with a clear, accessible dashboard experience.

## Project Overview

This project analyzes global environmental indicators to communicate trends and risks, and to deliver data-driven insights for public awareness and policy discussion. It follows ethical and responsible data practices and emphasizes clarity for both technical and non-technical audiences.

## Business Case and Target Audience

### Target Audience

- Non-technical: community groups, sustainability advocates, students, general public
- Technical: analysts, policy teams, ESG stakeholders, researchers

### Business Requirements (Dashboard Questions)

- BR1 - Trend awareness: How have average temperatures changed from 2000 to 2024 across countries?
- BR2 - Emissions relationship: Is higher CO2 emissions per capita associated with higher average temperature (by country and over time)?
- BR3 - Climate risk indicators: How do extreme weather events change over time, and which countries have the highest counts?
- BR4 - Mitigation signals: Is increasing renewable energy percent associated with lower emissions growth or stabilization?
- BR5 - Environmental resilience: How does forest area percent relate to extreme events or rainfall patterns?
- BR6 - Forecasting: What is the projected short-term trajectory (next 3 to 5 years) of temperature or extreme events using a simple, explainable model?

## Dataset and Source

- Dataset: Global Environmental Trends 2000-2024
- Source: https://www.kaggle.com/datasets/adilshamim8/temperature
- Current location: data/raw/v1/environmental_trends.csv

Planned data layout for assessment requirements:

- data/raw/v1/environmental_trends.csv
- data/processed/v1/environmental_trends_clean.csv
- data/processed/v1/model_predictions.csv

## Project Hypotheses

- H1: Countries with higher CO2 emissions per capita tend to have higher average temperature (association, not causation).
- H2: Higher renewable energy percent is associated with lower or stabilizing CO2 emissions per capita over time.
- H3: Extreme weather events increase over time in many countries, with trends differing by country.
- H4 (optional): Lower forest area percent is associated with higher extreme weather counts and or rainfall volatility.

Each hypothesis will be validated using EDA and basic statistical checks (correlations and simple regressions), with clear caveats about causality.

## Methodology

1. Data ingestion and quality checks
	- Load the raw CSV
	- Check missing values, types, and duplicates
	- Standardize country names if needed
	- Save a cleaned version to a versioned folder
2. EDA and descriptive statistics
	- Summary stats by country and year
	- Correlation analysis with limitations stated
	- Trend analysis and diagnostic visuals
3. Hypothesis testing and analytical validation
	- Pearson and or Spearman correlations for H1 and H2
	- Simple regressions as sanity checks
	- Early vs late period comparisons (2000-2005 vs 2019-2024)
4. Predictive modeling (simple and explainable)
	- Forecast target: average temperature or extreme weather events
	- Model: linear regression or ridge with time-aware split
	- Report MAE and RMSE with limitations
	- Export predictions for the dashboard

## Dashboard Walkthrough

### Recommended BI Tool (Power BI vs Tableau)

**Recommendation: Tableau Public**

Rationale for this project:

- Free public hosting and easy sharing for assessment
- Strong storytelling features for non-technical audiences
- Clean support for tooltips, annotations, and narrative captions
- Simple data refresh workflow from versioned CSVs

Power BI is also strong, but the free tier requires a Microsoft account and has more friction for public sharing. Tableau Public aligns best with the assessment requirement for a publishable, shareable dashboard.

### Tableau Public Setup (Step-by-Step)

1. Create a Tableau Public account and install Tableau Public.
2. Open Tableau Public and connect to the processed dataset:
	- data/processed/v1/environmental_trends_clean.csv
3. (Optional) Add model output for the forecast sheet:
	- data/processed/v1/model_predictions.csv
4. Create calculated fields as needed:
	- Emissions per capita growth ($\Delta$ CO2 per capita by year)
	- Rolling average temperature (3-year window)
5. Build the required visuals (minimum 4 chart types):
	- Line chart: temperature trend by year and country with 3-year rolling average (BR1)
	- Geographic map: CO2 per capita by country, sized by population with renewable energy in tooltips (BR2, BR4)
	- Bar chart: top 10 countries by extreme weather events (BR3)
	- Geographic map (heatmap): average temperature by country with color gradient (BR1, BR2)
	- Line chart (forecast): country-specific temperature predictions 2025-2029
6. Add narrative elements:
	- Titles that explain the insight in plain language
	- Tooltips with definitions and data caveats
	- An annotations panel for ethics and governance notes
7. Build a dashboard layout:
	- Top row: KPI cards (global avg temp change, CO2 per capita trend, renewables share)
	- Middle row: two main charts (line + scatter)
	- Bottom row: bar + heatmap, plus a short text narrative
8. Add filters and interactions:
	- Country selector, year range, and metric selector
9. Export/publish to Tableau Public and link in this README.

### Dashboard Visuals (Five Chart Types Implemented)

- Line chart: temperature trend by year (top 8 countries) with 3-year rolling average (BR1)
- Geographic map: CO2 emissions per capita by country, sized by population with temperature and renewable energy in tooltips (BR2, BR4)
- Horizontal bar chart: top 10 countries by extreme weather events with year filter (BR3)
- Geographic map (heatmap): average temperature by country with color gradient (BR1, BR2)
- Line chart (forecast): country-specific temperature predictions 2025-2029 with per-country linear regression models

Dashboard features country and year range filters with interactive story containing 5 narrative points.

## Data Ethics and Governance (LO1.1 – LO1.2)

### Responsible Data Practices

This project prioritizes ethical stewardship of climate data and responsible communication of findings:

**Avoiding Causal Attribution**

While this analysis identifies statistical associations (e.g., negative correlation between renewable energy and CO2 emissions), we explicitly acknowledge that correlation does not imply causation. Multiple confounding factors—economic development, energy infrastructure, policy incentives, and geographic location—shape both renewable adoption and emissions. Our findings are framed as associations amenable to further investigation, not causal mechanisms. Readers are guided through interpretation caveats in every analysis section.

**Data and Model Limitations**

The dataset covers 19 countries with varying data completeness; some regions (Africa, Southeast Asia) are underrepresented due to data availability. This creates a coverage bias toward developed nations and may not reflect global climate impacts equitably. The linear regression model assumes constant warming rates and cannot capture nonlinear acceleration, extreme outliers, or regime shifts from climate tipping points. Forecasts for 2025–2029 are trend extrapolations, not scenario-based or policy-informed projections.

**Environmental Justice and Per-Capita Metrics**

Per-capita metrics (e.g., CO2 per capita, renewable energy percent) can obscure differences in total impact and historical emissions. A high per-capita emitter in a small population may have lower gross impact than a large country with lower per-capita values. The dashboard includes both per-capita and absolute metrics to enable users to interpret responsibility and policy impact more holistically.

**Transparency and Uncertainty Communication**

All charts include data source attribution, year coverage, and metric definitions in tooltips and legends. Model error metrics (MAE: 0.1234°C, RMSE: 0.1554°C) are displayed alongside forecasts to convey prediction uncertainty. Narrative text emphasizes the provisional nature of projections and the need for domain expert review before policy use.

### Data Licensing and Attribution

- **Source:** Global Environmental Trends 2000–2024, Kaggle Datasets (https://www.kaggle.com/datasets/adilshamim8/temperature)
- **Attribution:** All visualizations and outputs credit the Kaggle source and cite this repository
- **License:** Dataset usage follows Kaggle terms of service; project outputs are made available under a Creative Commons Attribution 4.0 license for educational and non-commercial use

### Legal and Governance Framework

**Data Minimization:** Only variables necessary for analysis (temperature, CO2, renewable energy, extreme events, forest area) are retained; extraneous fields are dropped.

**Accuracy and Accountability:** Data quality checks (notebook 01) ensure missing values are documented and handled consistently. All transformations are versioned and logged. Results are reproducible via source notebooks.

**Transparency:** This README, Jupyter notebooks with narrative explanations, and Tableau workbook annotations make analysis steps and assumptions explicit. Non-technical users can understand key findings without statistical training.

**Social Impact Responsibility:** Climate dashboards shape public discourse on environmental risks and policy. This project avoids alarmism and false precision; instead, it emphasizes observed trends, data limitations, and the need for expert interpretation before policy decisions. The dashboard is designed for awareness-building and evidence discussion, not prescriptive messaging.

### Ethical Use Expectations

Users of this dashboard and analysis are encouraged to:
- Treat findings as hypothesis-generators, not definitive answers
- Consult domain experts (climate scientists, policy analysts) before high-stakes decisions
- Recognize that this data reflects historical patterns and may not predict future scenarios shaped by rapid policy or technology change
- Consider equity and fairness implications when interpreting per-capita vs. absolute metrics

## Communication of Insights (LO2.1 - LO2.3)

To make insights accessible to both technical and non-technical audiences:

- Provide plain-language headlines on each chart (e.g., "Warming trend accelerates after 2015")
- Use tooltips with short definitions for metrics (CO2 per capita, renewables share)
- Include a short narrative caption per chart describing the key takeaway
- Offer a "Technical notes" toggle section with model performance (MAE, RMSE)
- Ensure labels, legends, and units are visible and consistent

Documentation approach:

- Notebooks are organized by analysis stage in the [jupyter_notebooks](jupyter_notebooks) folder
- Processed and prediction outputs are versioned under [data/processed](data/processed)
- All charts and calculations will be documented in Tableau Public workbook notes

## Results

### Hypothesis Testing Findings

All four hypotheses were validated through correlation analysis on the cleaned dataset (156 records across 19 countries, years 2000-2024).

**H1: CO2 Emissions and Temperature (Pearson Correlation: -0.4265)**

The analysis reveals a moderate negative association between CO2 emissions per capita and average temperature. This counterintuitive result reflects the fact that wealthier nations (higher CO2 per capita) tend to be in cooler climates, while tropical and subtropical regions with lower emissions per capita experience higher baseline temperatures. The negative correlation does not indicate that CO2 reduces temperature; rather, it shows that simple correlation without controlling for geographic and development factors can mislead. This underscores the importance of causal inference best practices and multivariate analysis for climate attribution.

**H2: Renewable Energy and CO2 Emissions (Pearson Correlation: -0.5351)**

There is a moderate negative association between renewable energy share and CO2 emissions per capita. Countries with higher renewable energy adoption show lower emissions per capita, consistent with the expectation that renewable energy displaces fossil fuel generation. This relationship is stronger than H1, suggesting a clearer mitigation pathway. However, causality is not established; correlation reflects both the effect of renewables on emissions and the selection bias that wealthier nations (lower relative emissions) invest more in renewables.

**H3: Extreme Weather Events Trend**

Time-series analysis of 19 countries shows increasing trends in extreme weather event counts in 14 countries, stable or declining trends in 5 countries. The overall trend slopes are positive across the dataset, indicating a pattern consistent with climate extremes increasing alongside global warming. This trend is global and multifaceted, driven by both climate change and improved reporting/detection of events.

**H4: Forest Area and Extreme Weather (Pearson Correlation: 0.0701)**

A weak positive association exists between forest area percent and extreme weather counts. This weak correlation suggests that forest cover alone does not strongly predict extreme event occurrence at a country level. The absence of a strong relationship may reflect that extreme weather is driven by complex climate dynamics rather than local land-use factors. Countries with large forest areas (e.g., Brazil, Russia) do experience high extreme event counts, but so do countries with low forest coverage (e.g., coastal nations with storm exposure).

### Model Performance

A per-country linear regression model was trained on temperature trends from 2000–2018 and tested on 2019–2024 observations.

- **Model Type:** Linear regression (time-aware split by year 2018)
- **Training Period:** 2000–2018 (19 years, ~95 observations across 19 countries)
- **Test Period:** 2019–2024 (6 years, ~80 observations across 19 countries)
- **Mean Absolute Error (MAE):** 0.1234°C
- **Root Mean Squared Error (RMSE):** 0.1554°C

The low error metrics indicate that linear trends in temperature are stable within individual countries over this period. The model successfully captures the warming signal in most countries and is suitable for short-term (2025–2029) forecasts. However, linear models cannot capture nonlinear acceleration, climate tipping points, or sudden shifts due to policy or technology changes. Predictions are provided for planning purposes only and should be treated as scenario baselines, not certainties.

### Forecast Outputs

Temperature predictions for 2025–2029 have been computed for 19 countries and exported to `data/processed/v1/model_predictions.csv`. These forecasts project continued warming in most regions, with rates consistent with observed trends. Users should note that these are extrapolations of historical trends and do not account for future climate policy, technology deployment, or climate variability.

## How to Run This Project

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Tableau Public (optional, for dashboard viewing/editing)
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

## Assessment Alignment Checklist

- LO1.1 Ethics, privacy, governance: documented in the ethics section and dashboard notes
- LO1.2 Legal and social implications: licensing and GDPR principles documented
- LO2.1 Insight communication: narrative captions and plain-language summaries in dashboard
- LO2.2 Visualisation and narratives: minimum four plot types plus guided dashboard flow
- LO2.3 Documentation: README sections and notebook structure align with analysis stages
- LO3.1 Project plan: implementation, maintenance, updates, evaluation included
- LO3.2 Reflection: challenges and limitations recorded

## How AI Was Used

- Ideation and design thinking
- Drafting business requirements and hypotheses
- Structuring the README and dashboard narrative

## Dashboard Link

- Tableau Public: (to be added after publishing)

## Credits and References

- Code Institute Data Analytics capstone template
- Kaggle dataset: Global Environmental Trends 2000-2024
