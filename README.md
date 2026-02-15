# Global Environmental Trends (2000–2024)

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/temperature). We created a user story in which data analytics can be applied in a real project in the workplace.
* The dataset has 158 rows and represents environmental records from 19 countries across the globe. It indicates environmental indicators (temperature, emissions, rainfall, renewable energy, extreme weather events, forest coverage) and their respective values for the period between 2000 and 2024.

|Variable|Meaning|Units|
|:----|:----|:----|
|Year|Year of observation|2000 - 2024|
|Country|Country name|19 countries including United States, China, Germany, Brazil, Australia, India, Nigeria, Russia, Japan, Canada, Mexico, France, Indonesia, United Kingdom, Spain, South Korea, Italy, Saudi Arabia, South Africa|
|Avg_Temperature_degC|Average temperature in degrees Celsius|5.1 - 28.5|
|CO2_Emissions_tons_per_capita|Per-capita CO2 emissions in metric tons|0.5 - 20.2|
|Sea_Level_Rise_mm|Sea level rise in millimeters|0 - 59|
|Rainfall_mm|Annual rainfall in millimeters|54 - 2726|
|Population|Total population|19153000 - 1426000000|
|Renewable_Energy_pct|Percentage of energy from renewable sources|0 - 87.2|
|Extreme_Weather_Events|Count of extreme weather events reported|2 - 64|
|Forest_Area_pct|Forest area as percentage of total land area|0.5 - 68.5|

## Business Requirements

As a data analyst, environmental stakeholders and policy teams have asked you to help understand global environmental trends and provide insights to support awareness and planning for climate action.

Although stakeholders have access to raw environmental data, they need clear visualizations and interpretable insights to understand how different environmental indicators relate to each other and what the future trends might look like. They want to see patterns across countries and time periods to inform decision-making.

* 1—The client is interested in discovering how environmental indicators correlate with each other. Therefore, the client expects data visualizations of the correlated variables to show these relationships clearly.
* 2—The client is interested in predicting future temperature trends to support planning and risk assessment for the next 3-5 years.

## Hypothesis and how to validate

* Using environmental science knowledge, here are my hypothesis(es):
* 1: We suspect that higher CO2 emissions per capita are associated with higher average temperature. However, this reflects association, not causation, as climate zones and development levels vary by country.
	* How to validate: We will conduct a correlation study (Pearson and Spearman) to assess the strength and direction of the association between CO2 emissions per capita and average temperature.
* 2: We suspect that higher renewable energy percentage is associated with lower or stabilizing CO2 emissions per capita.
	* How to validate: We will conduct a correlation study to examine the relationship between renewable energy percentage and CO2 emissions per capita. A negative correlation would support this hypothesis.
* 3: We suspect that extreme weather events have increased over time from 2000 to 2024.
	* How to validate: We will analyze the time trend of extreme weather events using line plots and calculate the change in event frequency between the first and last years in the dataset. We will also perform regression analysis to test for a statistically significant trend.
* 4: We suspect that higher forest area percentage is associated with fewer extreme weather events.
	* How to validate: We will conduct a correlation study to examine the relationship between forest area percentage and extreme weather events. We acknowledge this may show weak association as country-level forest coverage alone is not a strong predictor.

## The rationale to map the business requirements to the Data Visualisations

* **Business Requirement 1:** Data Visualisation and Correlation study
	* As a client, I want to inspect the data related to environmental records so that I can discover how environmental indicators correlate with each other.
	* As a client, I want to conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to temperature and other key metrics so that I can discover meaningful relationships.
	* As a client, I want to plot the main variables against each other to visualize insights and discover how environmental indicators relate to climate trends.

* **Business Requirement 2:** Temperature Trend Prediction
	* As a client, I want to see predicted temperature trends for the next 3-5 years so that I can plan for climate risks and adaptation strategies.
	* As a client, I want to understand the model's accuracy metrics (MAE, RMSE, R²) so that I can assess the reliability of the predictions.
	* As a client, I want to see visualizations comparing actual historical data with predictions so that I can evaluate the model's performance.

## Dashboard Design

### Page 1: Executive Summary
* Quick project summary and key findings
	* Overview of the project's purpose and goals
	* Key observations section showing:
		* Average temperature change over the selected period
		* CO2 emissions per capita change
		* Renewable energy percentage change
	* Plain-language interpretation section explaining what these trends mean in practical terms
	* Recommended actions based on observed trends (climate adaptation, emissions reduction, renewable energy growth, etc.)
	* Country-specific findings table showing which countries face the greatest challenges or have made the strongest progress
	* Status indicators (red/yellow/green) to highlight urgent attention areas

### Page 2: Data Overview
* Data overview and quality assessment
	* Summary metrics: number of countries, years covered, total records, and indicators tracked
	* Data quality assessment showing missing values, duplicate rows, and data integrity status
	* Missing values by column if applicable
	* Key fields glossary with definitions and context for each metric
	* Sample data table showing the first 10 records from the dataset with country flag emojis
	* Download options for full dataset, data glossary, and quality report in CSV format

### Page 3: Overview
* Key signals and trends
	* Data coverage metrics (countries, years, records)
	* Latest trends showing:
		* Average temperature with change indicator
		* CO2 per person with change indicator
		* Renewable energy percentage with change indicator
	* Plain-language captions explaining what each metric means
	* Temperature over time line chart with guidance on interpretation
	* Extreme weather events bar chart showing top 10 countries in the latest year
	* Quick insights section highlighting:
		* Countries with biggest temperature increases
		* Countries with highest renewable energy growth
		* Overall trends summary

### Page 4: Explore Patterns
* Correlation and relationship analysis
	* Scatter plots showing:
		* CO2 emissions vs. average temperature
		* Renewable energy vs. CO2 emissions
		* Forest area vs. extreme weather events
		* Rainfall vs. temperature
	* Each plot includes:
		* Color coding by country
		* Trend lines where appropriate
		* Plain-language interpretation of what the pattern shows
	* Correlation heatmap showing relationships between all key metrics
	* Technical notes toggle to show/hide detailed statistical information
	* Time series visualizations showing trends over the full period

### Page 5: Modeling & Prediction
* Predictive modeling and forecasting
	* Model overview explaining the approach (per-country linear regression)
	* Model performance metrics section showing:
		* Mean Absolute Error (MAE)
		* Root Mean Squared Error (RMSE)
		* R² score
	* Plain-language explanation of what these metrics mean
	* Training and test split information (time-aware split at 2018)
	* Temperature forecast visualization showing:
		* Historical data (2000-2024)
		* Predictions for 2025-2029
		* Comparison between actual and predicted values
	* Country selector to view predictions for specific countries
	* Download button to export predictions as CSV
	* Model limitations and caveats clearly stated

### Page 6: Analytics Hub
* Advanced analytics and insights
	* Data quality dashboard with completeness and uniqueness metrics
	* Overall quality score calculation
	* Correlation heatmap showing relationships between all key metrics with color-coded values
	* Anomaly detection identifying unusual data points by country using z-score method
	* Downloadable quality reports and anomaly data in CSV format

### Page 7: Comparison Tool
* Country-to-country comparison
	* Multi-select up to 5 countries for side-by-side comparison
	* Latest year comparison bar charts for temperature, emissions, renewables, and extreme events
	* Trends over time with selectable metrics
	* Interactive line charts showing historical patterns for selected countries
	* Downloadable comparison data in CSV format

### Page 8: Scenario Builder
* Interactive what-if analysis tool
	* Create named scenarios with target years (2025-2050)
	* Adjust environmental parameters: CO₂ reduction, renewable energy increase, forest area change, extreme events, rainfall, and population growth
	* Real-time temperature prediction based on scenario inputs
	* Visual comparison between baseline and scenario outcomes
	* Shows temperature difference and impact interpretation

## Project Hypothesis and Validation

### Hypothesis 1: CO2 Emissions vs Temperature
* **Hypothesis:** Higher CO2 emissions per capita are associated with higher average temperature.
* **Validation:** We conducted a Pearson correlation analysis on the dataset.
* **Result:** The analysis showed a moderate negative correlation (approximately -0.3 to -0.4).
* **Interpretation:** The negative correlation likely reflects climate zone and development differences rather than a causal relationship. Countries in colder climates (like Russia, Canada) tend to have higher emissions due to heating needs and industrial activity, while countries in warmer climates (like Nigeria, Indonesia) have lower per-capita emissions. This demonstrates the importance of not inferring causation from correlation in climate data.
* **Conclusion:** The hypothesis is not supported by the data in the expected direction. However, this provides valuable insight into the complexity of climate-emissions relationships and the importance of controlling for geographic and developmental factors.

### Hypothesis 2: Renewable Energy vs CO2 Emissions
* **Hypothesis:** Higher renewable energy percentage is associated with lower or stabilizing CO2 emissions per capita.
* **Validation:** We conducted a correlation study examining the relationship between renewable energy percentage and CO2 emissions per capita.
* **Result:** The analysis showed a moderate negative correlation (approximately -0.4 to -0.5).
* **Interpretation:** Countries with higher renewable energy shares tend to have lower per-capita emissions, which aligns with the expected relationship. However, this is an association and not proof of causation, as countries with lower emissions may invest more in renewables for various policy and economic reasons.
* **Conclusion:** The hypothesis is supported by the observed association. This suggests that renewable energy adoption is a practical indicator for tracking decarbonization efforts, though multiple factors influence this relationship.

### Hypothesis 3: Extreme Weather Events Trend
* **Hypothesis:** Extreme weather events have increased over time from 2000 to 2024.
* **Validation:** We analyzed time trends using line plots and calculated the change in event frequency between 2000 and 2024. We also performed linear regression to test for a statistically significant trend.
* **Result:** Most countries show an increasing trend in extreme weather events, with global averages rising from approximately 10-15 events in 2000 to 25-30 events in 2024.
* **Interpretation:** The upward trend supports climate risk awareness. However, we must acknowledge potential reporting bias, as improved monitoring and reporting systems over this period may partly explain the increase. Despite this caveat, the consistent pattern across multiple countries suggests a real increase in extreme weather frequency.
* **Conclusion:** The hypothesis is supported by the data. The increasing trend in extreme weather events highlights the growing need for disaster preparedness and climate adaptation strategies.

### Hypothesis 4: Forest Area vs Extreme Events
* **Hypothesis:** Higher forest area percentage is associated with fewer extreme weather events.
* **Validation:** We conducted a correlation study between forest area percentage and extreme weather events.
* **Result:** The analysis showed a weak association (correlation close to 0 or slightly negative).
* **Interpretation:** Forest coverage alone is not a strong predictor of extreme weather events at the country level. This makes sense because extreme weather is influenced by many factors including ocean currents, atmospheric patterns, geography, and global climate systems. Forest area may provide localized benefits (soil stability, flood mitigation) but does not significantly reduce country-level extreme weather event counts.
* **Conclusion:** The hypothesis is not strongly supported by the data. While forest conservation remains important for many environmental reasons, it is not a primary factor in reducing extreme weather event frequency at the national scale.

## Data Ethics and Privacy

### Data Provenance and Transparency

* **Data Source:** The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/temperature), compiled by Adil Shamim, and represents aggregated environmental indicators from 19 countries (2000-2024).
* **Transparency Commitment:** We maintain transparency by:
  - Preserving the original raw dataset unchanged for auditability
  - Documenting all data cleaning and preprocessing steps in Notebook 01
  - Publishing all methodology and assumptions in this README
  - Tracking all model predictions with metadata (dates, parameters, versions)
  - Maintaining version history in Git for reproducibility

* **Data Attribution:** We acknowledge and credit:
  - Original data compiler: Adil Shamim (Kaggle)
  - Data source organizations: National meteorological agencies, World Bank, UN agencies
  - Code Institute: Educational framework and assessment criteria
  - Open-source communities: Pandas, Scikit-learn, Streamlit, Plotly developers

### Privacy and Personally Identifiable Information (PII)

* **No Personal Data:** This project exclusively uses aggregated, country-level environmental indicators. The dataset contains:
  - Temporal data (2000-2024 by year)
  - Geographic data (19 country averages)
  - Environmental metrics (temperature, emissions, rainfall, etc.)
  - **No individual records, household data, or personal identifiers**

* **Data Anonymization:** Country-level aggregation ensures complete anonymization:
  - Individual records cannot be reverse-engineered
  - Personal identities are completely protected
  - Household and location privacy maintained
  - Exportable data maintains same aggregation level

* **Data Retention:** Environmental data is:
  - Stored locally in CSV format (no external cloud storage)
  - Version controlled in Git (accessible only to authorized users)
  - Not shared with third parties
  - Deletable by project owner as needed

### Potential Bias and Limitations

* **Measurement Bias:** We acknowledge systematic differences in environmental monitoring:
  - **Developed nations:** Comprehensive satellite monitoring, dense weather station networks, precise CO2 tracking
  - **Developing nations:** Limited monitoring infrastructure, potential underreporting of events, incomplete historical records
  - **Impact:** This bias may underestimate environmental changes in developing countries and affect comparative analyses
  - **Mitigation:** We analyze trends per country separately rather than assuming global uniformity

* **Reporting Bias:** Extreme weather events show increasing trends that may reflect:
  - **Real increase:** Genuine climate-driven increase in extreme events
  - **Better detection:** Improved monitoring systems and reporting infrastructure since 2000
  - **Media effect:** Increased public attention and media coverage amplifying visibility
  - **Data quality:** Different countries use different definitions for "extreme weather events"
  - **Mitigation:** We note this explicitly in the dashboard and encourage users to view trends as "reported" events, not necessarily "actual" events

* **Selection Bias:** Dataset covers only 19 countries, not globally representative:
  - Over-representation of developed nations (10/19 = 53%)
  - Under-representation of Africa (3/19) and Southeast Asia (3/19)
  - Island nations and small developing countries absent
  - **Impact:** Findings may not generalize to underrepresented regions
  - **Mitigation:** Results are clearly framed as "19-country analysis," not global

* **Temporal Bias:** 25-year time period (2000-2024) may not capture:
  - Long-term climate cycles (30+ year periodicities)
  - Pre-2000 historical context needed for climate attribution
  - Potential data collection methodology changes over time
  - **Mitigation:** We use time-aware train-test splits and acknowledge limited data

* **Model Bias:** Linear regression assumptions:
  - Assumes linear trends over time (real climate may be non-linear)
  - Per-country models may not capture global interactions
  - Historical trends may not persist into future (structural break risk)
  - **Mitigation:** Models are explicitly labeled "trend-based" not "causal"

### Ethical Considerations for Climate Analysis

* **Responsible Climate Communication:**
  - ✅ We avoid catastrophizing or fear-mongering
  - ✅ We acknowledge uncertainty and model limitations
  - ✅ We distinguish correlation from causation explicitly
  - ✅ We provide context for findings (e.g., climate zones affect baseline temperatures)
  - ✅ We recommend official sources (IPCC, NOAA) for policy decisions

* **Avoiding Misuse:**
  - ❌ These predictions should NOT be used for official climate policy without validation
  - ❌ Results should NOT be presented as scientific consensus (that's IPCC's role)
  - ❌ Country comparisons should NOT fuel blame or guilt narratives
  - ✅ Instead, use analysis for awareness, education, and discussion catalyst

* **Equity Considerations:**
  - Climate impacts are not equally distributed globally
  - Vulnerable populations in developing nations may be most affected
  - This analysis reflects developed-nation monitoring bias
  - Energy transition recommendations must account for development levels
  - **Result:** We avoid blaming developing nations for emissions without context of per-capita consumption

* **Stakeholder Considerations:**
  - **Students/Educators:** Use for learning data analytics methodology
  - **Policy Makers:** May reference trends but must validate with official climate bodies
  - **Climate Advocates:** May use findings to raise awareness
  - **Fossil Fuel Industry:** Should not use to deny climate trends (methodology transparent for criticism)
  - **Developing Nations:** Should not feel blamed when data reflects monitoring infrastructure, not actual trends

### Usage Policy and Limitations

* **Educational Use Only:**
  - ✅ Permitted: Learning data analytics, teaching statistical methods, raising climate awareness
  - ✅ Permitted: Sharing with educational institutions, citing in research papers
  - ❌ Not permitted: Using for official government policy without expert review
  - ❌ Not permitted: Publishing predictions as peer-reviewed scientific findings

* **Model Uncertainty:**
  - **These are exploratory trends, not definitive forecasts**
  - Linear regression provides point estimates, not probability distributions (see confidence intervals for uncertainty range)
  - Real climate is influenced by policy changes, technological breakthroughs, and natural variability not in this model
  - User should interpret predictions as "if current trends continue" scenarios, not certainties

* **Recommended Validation Path:**
  - IPCC Reports: Official consensus on climate science
  - NOAA Climate Prediction Center: Authoritative forecasts
  - National Meteorological Services: Country-specific validation
  - Peer-reviewed climate literature: Scientific consensus for specific hypotheses

* **Limitations Explicitly Stated:**
  - 25-year time series per country (short for climate studies)
  - Country-level aggregation (hides regional variation)
  - Historical data quality varies by country and metric
  - No causality testing (only association)
  - No consideration of climate policies enacted post-2024
  - No integration of emissions reduction targets or climate commitments

### Accountability and Oversight

* **Transparency and Reproducibility:**
  - All code is open-source and auditable (hosted on GitHub)
  - All notebooks execute from raw data (full reproducibility)
  - All assumptions documented in code comments and README
  - All versions tracked (Git history captures evolution)
  - Users can verify, critique, and improve methodology

* **Feedback and Corrections:**
  - If bias or errors discovered, they can be corrected and redeployed
  - Contributors welcome to submit improvements
  - Model performance monitored against actual outcomes post-deployment
  - Results updated as new data becomes available

* **Disclaimer:**
  - This project is a **learning exercise** demonstrating data analytics capabilities
  - It is **not official climate science** and should not be cited as such
  - It is **subject to errors** in data collection, processing, or analysis
  - **Use at your own risk** for any decision-making
  - **Consult official sources** for any actions with real-world consequences

### Responsible AI Principles Applied

This project adheres to responsible AI principles:

| Principle | Implementation |
|-----------|-----------------|
| **Transparency** | All methodology documented; code open-source; assumptions explicit |
| **Fairness** | Acknowledges measurement bias; avoids blaming any nation; note data gaps |
| **Accountability** | Clear disclaimers; limitations stated; attribution provided |
| **Explainability** | Plain-language explanations; simple models; no black boxes |
| **Privacy** | Aggregated data only; no individuals identifiable; no PII |
| **Safety** | Predictions labeled exploratory; inappropriate uses discouraged |
| **Human Agency** | Users encouraged to validate; official sources recommended; decisions with humans |

## Project Structure

```
global_env_trend/
├── app.py                          # Streamlit dashboard application
├── data/
│   ├── raw/
│   │   └── v1/
│   │       └── environmental_trends.csv    # Original Kaggle dataset
│   └── processed/
│       └── v1/
│           ├── environmental_trends_clean.csv    # Cleaned dataset
│           ├── model_predictions.csv             # Temperature forecasts
│           └── model_predictions_with_ci.csv     # Model predictions with confidence intervals
├── jupyter_notebooks/
│   ├── 01_data_ingestion_quality_checks.ipynb    # Data cleaning and validation
│   ├── 02_eda_descriptive_stats.ipynb            # Exploratory data analysis
│   ├── 03_hypothesis_testing.ipynb               # Statistical hypothesis tests
│   └── 04_predictive_modeling.ipynb              # ML model training and forecasting
├── raw_dataset/                    # Raw dataset directory
├── Procfile                        # Heroku deployment configuration
├── requirements.txt                # Python dependencies
├── setup.sh                        # Streamlit configuration for Heroku
├── .gitignore                      # Git ignore file
├── .python-version                 # Python version specification
├── .slugignore                     # Heroku slug ignore file
└── README.md                       # This file
```

## Methodology and Workflow

### Notebook Execution Order

**The notebooks must be run in sequence:**

1. **01_data_ingestion_quality_checks.ipynb**
   - Purpose: Load raw data, check quality, clean, and export
   - Inputs: `data/raw/v1/environmental_trends.csv`
   - Outputs: `data/processed/v1/environmental_trends_clean.csv`

2. **02_eda_descriptive_stats.ipynb**
   - Purpose: Explore data, generate summary statistics, create visualizations
   - Inputs: `data/processed/v1/environmental_trends_clean.csv`
   - Outputs: EDA insights and plots

3. **03_hypothesis_testing.ipynb**
   - Purpose: Test all 4 hypotheses using correlation and trend analysis
   - Inputs: `data/processed/v1/environmental_trends_clean.csv`
   - Outputs: Hypothesis validation results

4. **04_predictive_modeling.ipynb**
   - Purpose: Train temperature forecasting model, generate predictions
   - Inputs: `data/processed/v1/environmental_trends_clean.csv`
   - Outputs: `data/processed/v1/model_predictions.csv`

**Note:** Notebook 1 must be run first as it creates the clean dataset used by all others. Notebook 4 should be run last as it generates the predictions displayed in the dashboard.

## Unfixed Bugs

* No bugs to report

## Prerequisites

Before running this project, ensure you have:

* **Python 3.12 or higher** (project tested with Python 3.12)
* **pip** (Python package manager)
* **Git** (for cloning the repository)
* **8GB RAM minimum** (recommended for running notebooks with profiling)
* **Operating System:** Windows, macOS, or Linux

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
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

5. **Verify installation:**
   ```bash
   streamlit --version
   jupyter --version
   ```

## Running the Dashboard Locally

After installation, run the Streamlit app:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Dashboard Navigation

Use the sidebar to switch between pages:
- **Executive Summary** - High-level findings and recommendations
- **Data Overview** - Dataset information and quality metrics
- **Overview** - Key environmental trends
- **Explore Patterns** - Correlation and relationship analysis
- **Modeling & Prediction** - Temperature forecasting tool
- **Analytics Hub** - Advanced analytics and anomaly detection
- **Comparison Tool** - Country-to-country comparisons
- **Scenario Builder** - What-if scenario modeling

### Filters

All pages respect the sidebar filters:
- **Countries:** Select one or more countries (default: All)
- **Year range:** Adjust the time period to analyze
- **Show technical notes:** Toggle detailed statistical information

## Deployment

### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly in case all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large or unnecessary files to .slugignore.

## Main Data Analysis and Libraries

### Core Production Libraries

* **NumPy** (version 1.26.1) - Used for numerical computations and array operations in data processing and model calculations
* **Pandas** (version 2.1.1) - Used for data manipulation, cleaning, aggregation, and transformation throughout all jupyter notebooks and the dashboard
* **Matplotlib** (version 3.8.0) - Used to create static visualizations including line plots, scatter plots, and bar charts in the analysis notebooks
* **Seaborn** (version 0.13.2) - Used to display correlation heatmaps and enhanced statistical plots with improved aesthetics
* **Plotly** (version 5.17.0) - Used to display interactive visualizations in the Streamlit dashboard including line charts, scatter plots, and bar charts with hover tooltips
* **Streamlit** (version 1.40.2) - Used as the dashboard framework to create an interactive web application for stakeholder exploration
* **feature-engine** (version 1.6.1) - Used for feature engineering tasks in the ML pipeline such as handling missing values and feature transformations
* **imbalanced-learn** (version 0.11.0) - Used for handling class imbalance in predictive modeling if needed
* **scikit-learn** (version 1.3.1) - Used for machine learning model development including LinearRegression, train-test splitting, and performance metrics (MAE, RMSE, R²)
* **XGBoost** (version 1.7.6) - Alternative gradient boosting library available for more complex modeling approaches if needed

### Development Libraries (used in notebooks only)

* **ydata-profiling** (version 4.12.0) - Used to generate comprehensive exploratory data analysis (EDA) reports in the jupyter notebooks
* **ppscore** (version 1.1.0) - Used in jupyter notebooks to calculate Predictive Power Score and understand better how features and target interact with each other
* **Yellowbrick** (version 1.5) - Used for machine learning visualization including model performance diagnostics and feature importance plots
* **Pillow** (version 10.0.1) - Used for image processing and display in the dashboard if needed

## Credits

* **Dataset:** This project uses the Global Environmental Trends 2000-2024 dataset from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/temperature), originally compiled by Adil Shamim. The dataset provides environmental indicators including temperature, emissions, renewable energy, extreme weather events, and forest coverage for 19 countries over a 25-year period.

* **Code Institute:** Project structure and methodology guidance from the Code Institute Data Analytics program. The project follows the assessment criteria and best practices taught in the capstone project module.

* **Streamlit Documentation:** Used Streamlit's official documentation and examples for building the interactive dashboard with multiple pages and dynamic visualizations.

* **Scikit-learn Documentation:** Referenced for implementing machine learning models, evaluation metrics, and best practices for train-test splitting and model validation.

* **Plotly Documentation:** Used for creating interactive visualizations with professional styling and user-friendly tooltips.

* **GitHub Copilot:** AI assistance was used for code suggestions, documentation improvements, and debugging during development.

## Acknowledgements

* I would like to thank Code Institute for providing the educational framework and assessment structure for this capstone project.
* Special thanks to my mentor for guidance and feedback throughout the project development.
* Thanks to the Kaggle community and Adil Shamim for making the Global Environmental Trends dataset publicly available.
* Thanks to the open-source community for developing and maintaining the excellent Python libraries used in this project.
