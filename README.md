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
- Current location: raw_dataset/update_temperature.csv

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

BI tool decision: Tableau Public (chosen for strong storytelling features, easy sharing, and assessment-friendly public hosting).

The dashboard will include at least four distinct chart types:

- Line chart: temperature trend by year and country (BR1)
- Scatter plot: CO2 emissions vs average temperature, with population size and renewable energy coloring (BR2, BR4)
- Bar chart: top 10 countries by extreme weather events (BR3)
- Heatmap: country by year for temperature or emissions intensity (BR1, BR2)

Optional enhancements include KPI cards, dual-axis line charts (CO2 vs renewables), and box plots for distribution comparisons.

## Ethics, Privacy, and Governance (LO1.1)

Even though this dataset is not personal data, responsible practice is still required:

- Avoid causal claims; focus on association and clearly communicate uncertainty
- Consider representativeness and coverage bias across countries
- Discuss environmental justice issues (per-capita metrics can obscure total impact)
- Communicate model limitations and uncertainty to avoid over-trust

A Data Ethics and Governance panel will be included in the dashboard to make this explicit to users.

## Legal and Social Implications (LO1.2)

- Data licensing and attribution: Kaggle source will be cited and linked
- GDPR and governance: The project does not handle personal data but follows principles of minimization, transparency, accuracy, and accountability
- Social impact: Climate dashboards can influence public perception and policy narratives; insights will be framed carefully with limitations

## Results

This section will summarize the final EDA highlights, statistical checks for hypotheses, and model performance (MAE, RMSE). It will include both technical metrics and plain-language summaries.

## Project Plan (LO3.1)

### Implementation

- Ingest data and store raw versioned files
- Clean and validate data, export processed datasets
- Run EDA and hypothesis checks
- Build a simple forecasting model
- Export predictions and build the BI dashboard

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

## How AI Was Used

- Ideation and design thinking
- Drafting business requirements and hypotheses
- Structuring the README and dashboard narrative

## Credits and References

- Code Institute Data Analytics capstone template
- Kaggle dataset: Global Environmental Trends 2000-2024
