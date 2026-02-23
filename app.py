from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent / "data" / "processed" / "v1"
CLEAN_PATH = DATA_DIR / "environmental_trends_clean.csv"
PRED_PATH = DATA_DIR / "model_predictions.csv"
CI_PATH = DATA_DIR / "model_predictions_with_ci.csv"


# ---------------------------------------------------------------------------
# Modeling helper functions
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame, include_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    feature_cols = [
        "Year",
        "CO2_Emissions_tons_per_capita",
        "Sea_Level_Rise_mm",
        "Rainfall_mm",
        "Population",
        "Renewable_Energy_pct",
        "Extreme_Weather_Events",
        "Forest_Area_pct",
    ]
    X = df[feature_cols].copy()
    country_dummies = pd.get_dummies(df["Country"], prefix="Country", drop_first=True)
    X = pd.concat([X, country_dummies], axis=1)
    if include_target and "Avg_Temperature_degC" in df.columns:
        y = df["Avg_Temperature_degC"].copy()
    else:
        y = None
    return X, y


def time_aware_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("Year").reset_index(drop=True)
    cutoff_idx = max(1, int(len(df_sorted) * train_frac))
    train_df = df_sorted.iloc[:cutoff_idx]
    test_df = df_sorted.iloc[cutoff_idx:]
    if test_df.empty:
        test_df = train_df.tail(1)
        train_df = train_df.iloc[:-1]
    return train_df, test_df


def model_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def align_features(row_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in feature_columns if col not in row_df.columns]
    for col in missing_cols:
        row_df[col] = 0
    return row_df[feature_columns]


def build_scenario_row(
    data_dict: dict, country: str, training_columns: list[str]
) -> pd.DataFrame:
    feature_cols = [
        "Year",
        "CO2_Emissions_tons_per_capita",
        "Sea_Level_Rise_mm",
        "Rainfall_mm",
        "Population",
        "Renewable_Energy_pct",
        "Extreme_Weather_Events",
        "Forest_Area_pct",
    ]
    row = pd.DataFrame([{col: data_dict[col] for col in feature_cols}])
    country_cols = [c for c in training_columns if c.startswith("Country_")]
    for col in country_cols:
        row[col] = 0
    target_col = f"Country_{country}"
    if target_col in country_cols:
        row[target_col] = 1
    return row[training_columns]


st.set_page_config(
    page_title="Global Environmental Trends Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Global Environmental Trends 2000-2024 - Data Analytics Dashboard by Code Institute"
    }
)


@st.cache_data(show_spinner="Loading dataset...")
def load_clean_data() -> pd.DataFrame:
    """Load cleaned environmental data with validation"""
    if not CLEAN_PATH.exists():
        st.error(f"❌ Data file not found: {CLEAN_PATH}. Please run notebook 01_data_ingestion_quality_checks.ipynb first.")
        st.stop()
    df = pd.read_csv(CLEAN_PATH)
    df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data(show_spinner="Loading predictions...")
def load_predictions() -> pd.DataFrame:
    """Load model predictions with fallback support"""
    if PRED_PATH.exists():
        df = pd.read_csv(PRED_PATH)
        df["Year"] = df["Year"].astype(int)
        return df
    return pd.DataFrame(columns=["Year", "Country", "Predicted_Avg_Temperature_degC"])


@st.cache_data(show_spinner="Loading confidence intervals...")
def load_ci_predictions() -> pd.DataFrame:
    """Load model predictions with bootstrap confidence intervals"""
    if CI_PATH.exists():
        df = pd.read_csv(CI_PATH)
        df["Year"] = df["Year"].astype(int)
        return df
    return pd.DataFrame()


def filter_data(df: pd.DataFrame, countries: list[str], year_range: tuple[int, int]) -> pd.DataFrame:
    filtered = df[df["Country"].isin(countries)].copy()
    return filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]


# Humanized labels for Plotly axes and legends
LABEL_MAP = {
    "Avg_Temperature_degC": "Average Temperature (°C)",
    "CO2_Emissions_tons_per_capita": "CO₂ Emissions (tons per capita)",
    "Sea_Level_Rise_mm": "Sea Level Rise (mm)",
    "Rainfall_mm": "Rainfall (mm)",
    "Renewable_Energy_pct": "Renewable Energy (%)",
    "Extreme_Weather_Events": "Extreme Weather Events",
    "Forest_Area_pct": "Forest Area (%)",
    "Population": "Population",
    "Year": "Year",
    "Country": "Country",
    "Predicted_Avg_Temperature_degC": "Predicted Temperature (°C)",
}


@st.cache_data(show_spinner=False)
def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation between key metrics (cached for better performance)"""
    metrics = [
        "Avg_Temperature_degC", "CO2_Emissions_tons_per_capita", "Sea_Level_Rise_mm",
        "Rainfall_mm", "Renewable_Energy_pct", "Extreme_Weather_Events", "Forest_Area_pct"
    ]
    available_metrics = [m for m in metrics if m in df.columns]
    return df[available_metrics].corr()


def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 2.5) -> pd.DataFrame:
    """Detect anomalies using z-score method"""
    df_copy = df.copy()
    if column not in df_copy.columns:
        return df_copy
    
    # Calculate z-scores per country
    df_copy["z_score"] = df_copy.groupby("Country")[column].transform(lambda x: np.abs((x - x.mean()) / (x.std() + 1e-8)))
    df_copy["is_anomaly"] = df_copy["z_score"] > threshold
    return df_copy


def export_csv(dataframe: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for download"""
    return dataframe.to_csv(index=False).encode('utf-8')


def get_data_quality_score(df: pd.DataFrame) -> float:
    """Calculate overall data quality score (0-100)"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    completeness = (1 - missing_cells / total_cells) * 60
    uniqueness = (1 - duplicates / max(df.shape[0], 1)) * 40
    return min(100, completeness + uniqueness)


# ---------------------------------------------------------------------------
# Environmental colour palette (used across all charts)
# ---------------------------------------------------------------------------
ENV_COLORS = [
    "#2E7D32",  # forest green
    "#1565C0",  # ocean blue
    "#6D4C41",  # earth brown
    "#00838F",  # teal
    "#558B2F",  # olive
    "#EF6C00",  # amber
    "#AD1457",  # berry
    "#5E35B1",  # violet
    "#00695C",  # dark teal
    "#F9A825",  # gold
    "#37474F",  # charcoal
    "#C62828",  # deep red
    "#0277BD",  # sky blue
    "#4E342E",  # dark brown
    "#1B5E20",  # dark green
    "#FF8F00",  # dark amber
    "#283593",  # indigo
    "#795548",  # warm brown
    "#00BFA5",  # mint
]

# Register custom Plotly theme so every chart uses the palette automatically
_env_template = pio.templates["plotly"]
_env_template.layout.colorway = ENV_COLORS
pio.templates.default = "plotly"

# ---------------------------------------------------------------------------
# Global CSS — hero banner, navigation, dividers, sidebar
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Hero banner ───────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 40%, #43A047 100%);
    padding: 1.8rem 2rem 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
    text-align: center;
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.2rem;
    margin: 0;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.hero-banner p {
    color: #C8E6C9;
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}

/* ── Navigation bar — active page highlight ────────────────────────── */
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background-color: #2E7D32 !important;
    border-color: #2E7D32 !important;
    color: #ffffff !important;
    font-weight: 600;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background-color: #E8F5E9 !important;
    border: 1px solid #A5D6A7 !important;
    color: #1B5E20 !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    background-color: #C8E6C9 !important;
    border-color: #66BB6A !important;
}

/* ── Styled section dividers ───────────────────────────────────────── */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #66BB6A, transparent);
    margin: 1.2rem 0;
}

/* ── Sidebar branding ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #E8F5E9;
}
</style>
""", unsafe_allow_html=True)

# ── Hero banner ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🌍 Global Environmental Trends (2000–2024)</h1>
    <p>Climate-related signals • Interpretable baseline model • Built for discussion &amp; planning</p>
</div>
""", unsafe_allow_html=True)

# Initialize data variables to avoid unbound variable errors
clean_df = pd.DataFrame()
pred_df = pd.DataFrame()

# Load data with error handling
try:
    with st.spinner("Loading environmental data..."):
        clean_df = load_clean_data()
        pred_df = load_predictions()
except FileNotFoundError:
    st.error("⚠️ **Data file not found!** Please run the data cleaning notebook first.")
    st.info("📝 **How to fix:**")
    st.code("jupyter notebook jupyter_notebooks/01_data_ingestion_quality_checks.ipynb", language="bash")
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Error loading data:** {str(e)}")
    st.info("💡 Please check that the data files exist in `data/processed/v1/` directory")
    st.stop()

# Page navigation mapping: display label → page key
PAGE_OPTIONS = [
    ("📝 Executive Summary", "Executive Summary"),
    ("📊 Data Overview", "Data Overview"),
    ("📈 Overview", "Overview"),
    ("🔍 Explore Patterns", "Explore Patterns"),
    ("🎯 Modeling & Prediction", "Modeling & Prediction"),
    ("📊 Analytics Hub", "Analytics Hub"),
    ("🔄 Comparison Tool", "Comparison Tool"),
    ("⚙️ Scenario Builder", "Scenario Builder"),
]
PAGE_LABELS = [label for label, _ in PAGE_OPTIONS]
PAGE_KEYS = [key for _, key in PAGE_OPTIONS]
LABEL_TO_KEY = dict(PAGE_OPTIONS)

if "current_page" not in st.session_state:
    st.session_state.current_page = "Executive Summary"

# ── Sidebar branding ─────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center;padding:0.6rem 0 0.2rem 0'>"
    "<span style='font-size:2.2rem'>🌿</span><br>"
    "<span style='font-weight:700;color:#1B5E20;font-size:1rem'>Environmental Trends</span><br>"
    "<span style='color:#558B2F;font-size:0.78rem'>2000 – 2024 Dashboard</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    PAGE_LABELS,
    index=PAGE_KEYS.index(st.session_state.current_page) if st.session_state.current_page in PAGE_KEYS else 0,
)
st.session_state.current_page = LABEL_TO_KEY.get(page, "Executive Summary")

st.sidebar.markdown("---")

st.sidebar.header("Filters")

# Reset everything button
if st.sidebar.button("🔄 Reset App", help="Reset page, filters, and all widget state to defaults"):
    # Preserve only the keys Streamlit needs internally
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    # Restore clean defaults
    st.session_state.current_page = "Executive Summary"
    st.session_state.year_slider = (int(clean_df["Year"].min()), int(clean_df["Year"].max()))
    st.rerun()

all_countries = sorted(clean_df["Country"].unique().tolist())
country_options = ["All"] + all_countries

# Countries filter – Streamlit multiselect has built-in type-to-search
st.sidebar.markdown("**📍 Countries** (Select or Search)")
st.sidebar.caption(
    "Choose one or more countries to focus the dashboard. "
    "Leave **All** selected to view every country. "
    "The filter applies to all pages except the Scenario Builder, which has its own country selector."
)
selected_countries = st.sidebar.multiselect(
    "Select countries",
    country_options,
    default=["All"],
    help="Type to search. Select one or more countries. 'All' includes every country.",
    label_visibility="collapsed"
)
if not selected_countries or "All" in selected_countries:
    selected_countries = all_countries

min_year, max_year = int(clean_df["Year"].min()), int(clean_df["Year"].max())

# Initialize session state for year slider if not exists
if "year_slider" not in st.session_state:
    st.session_state.year_slider = (min_year, max_year)

# Year slider with tooltip
year_range = st.sidebar.slider(
    "📅 Year range", 
    min_year, 
    max_year,
    key="year_slider",
    help="Filter data by time period (2000-2024)"
)

# Technical notes toggle with tooltip
st.sidebar.markdown("---")
show_technical = st.sidebar.checkbox(
    "🔬 Show technical notes", 
    value=False,
    help="Toggle detailed statistical information, model metrics, and confidence intervals"
)

if show_technical:
    st.sidebar.markdown(
        "**Technical notes enabled** *(shows on Overview, Explore Patterns, Modeling & Prediction)*"
    )

with st.sidebar.expander("Recommendation thresholds", expanded=False):
    st.caption("Adjust thresholds for automatic recommendations")
    st.info(
        "💡 **What these do:** These thresholds control which countries appear in the "
        "'Country-Specific Findings' table on the Executive Summary page. "
        "Countries exceeding these thresholds get flagged with red/yellow warnings."
    )
    temp_threshold = st.number_input(
        "Temp change (degC)", 
        value=0.5, min_value=0.0, step=0.1,
        help="Threshold for significant temperature change"
    )
    co2_threshold = st.number_input(
        "CO2 change (tons per capita)", 
        value=0.5, min_value=0.0, step=0.1,
        help="Threshold for significant CO2 emissions change"
    )
    renew_threshold = st.number_input(
        "Renewables change (%)", 
        value=2.0, min_value=0.0, step=0.5,
        help="Threshold for significant renewable energy change"
    )
    events_threshold = st.number_input(
        "Extreme events change", 
        value=5.0, min_value=0.0, step=1.0,
        help="Threshold for significant change in extreme weather events"
    )

filtered_df = filter_data(clean_df, selected_countries, year_range)

st.sidebar.download_button(
    "Download filtered data",
    filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="environmental_trends_filtered.csv",
    mime="text/csv",
    help="Download the currently filtered dataset as CSV"
)

if filtered_df.empty:
    st.warning("⚠️ **No data matches your current filters**")
    st.info("💡 **Suggestions:**")
    st.markdown("""
    - Try expanding the year range
    - Select more countries
    - Click **Reset App** above to start over
    """)
    st.stop()

# ── Global quick-access navigation (visible on every page) ──
nav_icons = {
    "Executive Summary": "📝", "Data Overview": "📊", "Overview": "📈",
    "Explore Patterns": "🔍", "Modeling & Prediction": "🎯",
    "Analytics Hub": "📊", "Comparison Tool": "🔄", "Scenario Builder": "⚙️",
}
nav_cols = st.columns(len(PAGE_OPTIONS))
for i, (label, key) in enumerate(PAGE_OPTIONS):
    icon = nav_icons.get(key, "")
    is_active = st.session_state.current_page == key
    btn_label = f"{icon} {key}"
    if nav_cols[i].button(
        btn_label, key=f"topnav_{key}", use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.current_page = key
        st.rerun()

st.markdown("---")

if st.session_state.current_page == "Executive Summary":
    st.subheader("📝 Executive Summary")

    st.markdown(
        "📌 **What you're seeing:** This summary shows the key environmental trends over the selected "
        "period, how they have changed, and what actions may be worth considering.\n\n"
        "**Important note:** The patterns you see reflect observed associations in the data, not proven causes. "
        "Environmental change is influenced by many factors working together."
    )

    summary_grouped = filtered_df.groupby("Year", as_index=False).mean(numeric_only=True)
    summary_first = summary_grouped.iloc[0]
    summary_last = summary_grouped.iloc[-1]

    # Key Metrics
    st.subheader("🌡️ Key Observations")
    
    summary_cols = st.columns(3)
    temp_change = summary_last['Avg_Temperature_degC'] - summary_first['Avg_Temperature_degC']
    summary_cols[0].metric(
        "Average Temperature",
        f"{summary_last['Avg_Temperature_degC']:.2f}°C",
        f"{temp_change:+.2f}°C since {int(summary_first['Year'])}",
        delta_color="inverse" if temp_change > 0 else "normal"
    )
    
    co2_change = summary_last['CO2_Emissions_tons_per_capita'] - summary_first['CO2_Emissions_tons_per_capita']
    summary_cols[1].metric(
        "CO2 Emissions (per person)",
        f"{summary_last['CO2_Emissions_tons_per_capita']:.2f} tons",
        f"{co2_change:+.2f} tons since {int(summary_first['Year'])}",
        delta_color="inverse" if co2_change > 0 else "normal"
    )
    
    renew_change = summary_last['Renewable_Energy_pct'] - summary_first['Renewable_Energy_pct']
    summary_cols[2].metric(
        "Renewable Energy",
        f"{summary_last['Renewable_Energy_pct']:.2f}%",
        f"{renew_change:+.2f}% since {int(summary_first['Year'])}",
        delta_color="normal" if renew_change > 0 else "inverse"
    )

    # Plain-language interpretation
    st.subheader("💡 What This Means")
    
    observations = []
    
    temp_delta = summary_last["Avg_Temperature_degC"] - summary_first["Avg_Temperature_degC"]
    if temp_delta > 0.3:
        observations.append(
            f"**🔴 Temperatures are rising**: Average temperature has increased by {temp_delta:.2f}°C. "
            "This affects agriculture, water availability, and extreme weather patterns."
        )
    elif temp_delta > 0:
        observations.append(
            f"**🟡 Slight warming trend**: Temperature has increased by {temp_delta:.2f}°C. "
            "Continue monitoring as small changes can have significant cumulative effects."
        )
    else:
        observations.append("**🟢 Temperature is stable or declining**: This is positive progress.")
    
    co2_delta = summary_last["CO2_Emissions_tons_per_capita"] - summary_first["CO2_Emissions_tons_per_capita"]
    if co2_delta > 0.2:
        observations.append(
            f"**🔴 Emissions are increasing**: Per-capita emissions have risen by {co2_delta:.2f} tons. "
            "This means each person is producing more CO2, which requires policy and behavioral shifts."
        )
    elif co2_delta > 0:
        observations.append(
            f"**🟡 Slight emissions increase**: Per-capita emissions have risen by {co2_delta:.2f} tons. "
            "Any increase suggests energy demand or fossil fuel dependence is growing."
        )
    else:
        observations.append("**🟢 Emissions are declining**: Decarbonization efforts may be working.")
    
    renew_delta = summary_last["Renewable_Energy_pct"] - summary_first["Renewable_Energy_pct"]
    if renew_delta > 2:
        observations.append(
            f"**⚡ Renewable energy is growing**: Renewable share has increased by {renew_delta:.2f}%. "
            "This indicates a transition toward cleaner energy sources."
        )
    elif renew_delta > 0:
        observations.append(
            f"**⚡ Renewable energy is growing slowly**: Renewable share has increased by {renew_delta:.2f}%. "
            "Accelerating the transition could help reduce emissions faster."
        )
    else:
        observations.append("**⚡ Renewable energy share is declining**: Fossil fuels may be taking a larger share.")
    
    for obs in observations:
        st.markdown(obs)

    # Recommendations
    st.subheader("📋 Recommended Actions")
    
    rec_items = []
    rec_details = []

    if temp_delta > 0:
        rec_items.append("Climate adaptation and risk management")
        rec_details.append(
            "With rising temperatures, prioritize: (1) Infrastructure planning for extreme weather, "
            "(2) Agricultural drought preparedness, (3) Public health systems for heat stress, "
            "(4) Community awareness campaigns."
        )
    
    events_delta = summary_last["Extreme_Weather_Events"] - summary_first["Extreme_Weather_Events"]
    if events_delta > 0:
        rec_items.append("Strengthen disaster resilience")
        rec_details.append(
            "Extreme weather events are increasing. Focus on: (1) Early warning systems, "
            "(2) Emergency response protocols, (3) Community shelters and evacuation plans, (4) Insurance schemes."
        )
    
    if co2_delta > 0:
        rec_items.append("Accelerate emissions reduction")
        rec_details.append(
            "Rising per-capita emissions indicate growing energy demand. Actions: (1) Promote energy efficiency "
            "in buildings and transportation, (2) Incentivize renewable energy adoption, (3) Reduce industrial "
            "emissions through technology, (4) Track progress quarterly."
        )
    
    if renew_delta > 0:
        rec_items.append("Capitalize on renewable momentum")
        rec_details.append(
            "Renewable energy growth is a strength. Build on this by: (1) Expanding grid infrastructure for "
            "distributed renewables, (2) Training workforce for clean energy jobs, (3) Removing policy barriers, "
            "(4) Setting ambitious renewable targets."
        )
    
    if not rec_items:
        rec_items.append("Maintain current monitoring")
        rec_details.append(
            "Signals are mixed. Continue: (1) Regular data collection, (2) Stakeholder engagement, "
            "(3) Trend analysis, (4) Scenario planning for future interventions."
        )

    for i, (rec, detail) in enumerate(zip(rec_items, rec_details)):
        with st.expander(f"**{i+1}. {rec}**", expanded=(i==0)):
            st.markdown(detail)

    st.info(
        "⚠️ **Important disclaimer:** These recommendations are based on observed data patterns, not proven causation. "
        "Environmental changes result from complex interactions of multiple factors. Consult domain experts and conduct "
        "rigorous impact assessments before implementing major policy changes."
    )

    # ── Visual Highlights: narrative summary of what the charts across the dashboard reveal ──
    st.subheader("📊 Visual Highlights")
    st.markdown(
        "The charts throughout this dashboard tell a consistent story about the state of our "
        "environment between {start} and {end}. Here is what they reveal.".format(
            start=int(summary_first["Year"]), end=int(summary_last["Year"])
        )
    )

    # Temperature narrative
    st.markdown("**Temperature trends**")
    if temp_delta > 0.3:
        st.markdown(
            "The temperature line chart on the *Overview* page shows a clear upward trajectory. "
            f"Average global temperature rose by roughly {temp_delta:.2f} °C over the observed period. "
            "This warming is not uniform — the per-country box plots on the *Data Overview* page reveal "
            "that some nations experienced much sharper increases than others, with the widest boxes "
            "indicating the most variable temperature records."
        )
    elif temp_delta > 0:
        st.markdown(
            "The temperature line chart on the *Overview* page shows a modest upward drift of "
            f"about {temp_delta:.2f} °C. While not dramatic in isolation, even small sustained "
            "changes can compound over time. The histograms on the *Data Overview* page confirm "
            "that temperature values are distributed roughly symmetrically around the mean."
        )
    else:
        st.markdown(
            "The temperature line chart on the *Overview* page shows a stable or slightly "
            "cooling trend over the period. The histograms on *Data Overview* confirm that "
            "temperature values cluster tightly around the global mean."
        )

    # Emissions & renewables narrative
    st.markdown("**Emissions and energy**")
    st.markdown(
        "The scatter plots on the *Explore Patterns* page illustrate two important relationships. "
        "First, the CO₂-vs-temperature scatter shows that cold-climate industrial nations (e.g. Russia, Canada) "
        "tend to have both high emissions and low temperatures, which is why the overall correlation is "
        "negative — geography confounds the relationship rather than emissions causing cooling. "
        "Second, the renewables-vs-emissions scatter shows a downward trend: countries with a higher share "
        "of renewable energy generally produce fewer emissions per person, supporting the case for continued "
        "investment in clean energy."
    )

    # Extreme weather narrative
    st.markdown("**Extreme weather events**")
    if events_delta > 0:
        st.markdown(
            "The extreme-weather trend line on the *Explore Patterns* page slopes upward, indicating "
            "that the average number of extreme weather events across countries has grown over time. "
            "The bar chart on the *Overview* page spotlights which nations bore the heaviest burden in the "
            "most recent year. Together, these visuals underline the growing need for disaster preparedness "
            "and climate-resilient infrastructure."
        )
    else:
        st.markdown(
            "The extreme-weather trend line on the *Explore Patterns* page is relatively flat, suggesting "
            "no dramatic increase in reported events. However, this may partly reflect reporting limitations "
            "rather than genuine stability."
        )

    # Forecast narrative
    st.markdown("**Forecasting and uncertainty**")
    st.markdown(
        "The *Modeling & Prediction* page extends historical trends into 2025–2029 using a simple "
        "per-country linear model. The confidence-interval bands around each forecast line show how "
        "certain (or uncertain) those projections are. Narrow bands suggest a stable, predictable history; "
        "wide bands warn that the country's temperature has been volatile and forecasts should be treated "
        "with extra caution. These forecasts assume current trends continue unchanged — any shift in "
        "policy, technology, or global emissions would alter the trajectory."
    )
    st.markdown("---")

    # ── Interactive World Map ─────────────────────────────────────────────
    st.subheader("🗺️ Global Snapshot")
    st.markdown(
        "**What this shows:** A world map highlighting the 19 countries in the dataset, coloured by "
        "the metric you select. Darker shading indicates higher values.\n\n"
        "**How to read it:** Hover over a country to see its exact value. Grey countries are not in the "
        "dataset. Use the dropdown to switch between temperature, emissions, renewables, or extreme events."
    )

    # ISO-3166 alpha-3 codes for Plotly choropleth
    COUNTRY_ISO = {
        "Australia": "AUS", "Brazil": "BRA", "Canada": "CAN", "China": "CHN",
        "France": "FRA", "Germany": "DEU", "India": "IND", "Indonesia": "IDN",
        "Italy": "ITA", "Japan": "JPN", "Mexico": "MEX", "Nigeria": "NGA",
        "Russia": "RUS", "South Africa": "ZAF", "South Korea": "KOR",
        "Spain": "ESP", "Saudi Arabia": "SAU", "United Kingdom": "GBR",
        "United States": "USA",
    }

    map_metric = st.selectbox(
        "Select metric to display on map",
        ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita",
         "Renewable_Energy_pct", "Extreme_Weather_Events"],
        format_func=lambda x: LABEL_MAP.get(x, x),
        key="exec_map_metric",
    )

    map_year = int(filtered_df["Year"].max())
    map_data = filtered_df[filtered_df["Year"] == map_year].copy()
    map_data["iso_alpha"] = map_data["Country"].map(COUNTRY_ISO)
    map_data = map_data.dropna(subset=["iso_alpha"])

    if not map_data.empty:
        map_fig = px.choropleth(
            map_data,
            locations="iso_alpha",
            color=map_metric,
            hover_name="Country",
            hover_data={
                map_metric: ":.2f",
                "iso_alpha": False,
            },
            color_continuous_scale="YlOrRd" if map_metric != "Renewable_Energy_pct" else "YlGn",
            labels=LABEL_MAP,
            title=f"{LABEL_MAP.get(map_metric, map_metric)} by Country ({map_year})",
        )
        map_fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
            ),
            height=450,
            margin=dict(t=50, b=10, l=0, r=0),
        )
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption(
            f"💡 Showing data for **{len(map_data)}** countries in **{map_year}**. "
            "Grey areas have no data in this dataset. Colour intensity reflects the selected metric — "
            "darker shading means higher values."
        )
    else:
        st.info("No map data available for the current filters.")

    st.markdown("---")

    st.subheader("🌍 Country-Specific Findings")
    st.markdown(
        "Below is a detailed breakdown by country showing where the biggest changes are happening. "
        "Use this to identify which countries face the greatest challenges or have the strongest progress."
    )
    
    st.info(
        "ℹ️ **How this works:** Countries are flagged based on the **Recommendation thresholds** set in the sidebar. "
        f"Currently: Temperature change ≥ {temp_threshold}°C, CO₂ change ≥ {co2_threshold} tons, "
        f"Renewables change ≥ {renew_threshold}%, Extreme events change ≥ {events_threshold}. "
        "Adjust these values in the sidebar to customize which findings appear."
    )
    
    years_by_country = (
        filtered_df.sort_values("Year")
        .groupby("Country")["Year"]
        .agg(first="first", last="last")
        .reset_index()
    )
    first_by_country = filtered_df.merge(
        years_by_country, left_on=["Country", "Year"], right_on=["Country", "first"], how="inner"
    )
    last_by_country = filtered_df.merge(
        years_by_country, left_on=["Country", "Year"], right_on=["Country", "last"], how="inner"
    )

    deltas = first_by_country[
        [
            "Country",
            "Avg_Temperature_degC",
            "CO2_Emissions_tons_per_capita",
            "Renewable_Energy_pct",
            "Extreme_Weather_Events",
        ]
    ].rename(
        columns={
            "Avg_Temperature_degC": "temp_first",
            "CO2_Emissions_tons_per_capita": "co2_first",
            "Renewable_Energy_pct": "renew_first",
            "Extreme_Weather_Events": "events_first",
        }
    ).merge(
        last_by_country[
            [
                "Country",
                "Avg_Temperature_degC",
                "CO2_Emissions_tons_per_capita",
                "Renewable_Energy_pct",
                "Extreme_Weather_Events",
            ]
        ].rename(
            columns={
                "Avg_Temperature_degC": "temp_last",
                "CO2_Emissions_tons_per_capita": "co2_last",
                "Renewable_Energy_pct": "renew_last",
                "Extreme_Weather_Events": "events_last",
            }
        ),
        on="Country",
        how="inner",
    )
    deltas["temp_delta"] = deltas["temp_last"] - deltas["temp_first"]
    deltas["co2_delta"] = deltas["co2_last"] - deltas["co2_first"]
    deltas["renew_delta"] = deltas["renew_last"] - deltas["renew_first"]
    deltas["events_delta"] = deltas["events_last"] - deltas["events_first"]

    rec_rows = []
    
    for _, row in deltas.iterrows():
        country_recs = []
        status_symbols = []
        
        if row["temp_delta"] >= temp_threshold:
            country_recs.append("⚠️ Significant warming")
            status_symbols.append("🔴")
        elif row["temp_delta"] > 0:
            country_recs.append("Moderate warming")
            status_symbols.append("🟡")
        else:
            status_symbols.append("🟢")
        
        if row["events_delta"] >= events_threshold:
            country_recs.append("⚠️ Extreme events rising")
            if "🔴" not in status_symbols and status_symbols:
                status_symbols[0] = "🔴"
        elif row["events_delta"] > 0:
            if "🟢" in status_symbols:
                status_symbols[0] = "🟡"
        
        if row["co2_delta"] >= co2_threshold:
            country_recs.append("⚠️ Emissions increasing")
            if "🔴" not in status_symbols and status_symbols:
                status_symbols[0] = "🔴"
        elif row["co2_delta"] > 0:
            if "🟢" in status_symbols:
                status_symbols[0] = "🟡"
        
        if row["renew_delta"] >= renew_threshold:
            country_recs.append("✅ Renewable energy growing")
            if status_symbols and "🔴" not in status_symbols[0]:
                status_symbols[0] = "🟢"
        
        if country_recs:
            status = status_symbols[0] if status_symbols else "⚪"
            rec_rows.append(
                {
                    "Status": status,
                    "Country": row["Country"],
                    "Key Findings": "; ".join(country_recs),
                }
            )

    if rec_rows:
        rec_df = pd.DataFrame(rec_rows).sort_values("Country")
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        
        st.markdown(
            "**Legend:** 🔴 = Urgent attention needed | 🟡 = Monitor closely | 🟢 = Positive progress"
        )
    else:
        st.info("No countries exceeded the selected thresholds. Adjust threshold values in the Filters sidebar to see recommendations.")

elif st.session_state.current_page == "Data Overview":
    st.subheader("📊 Data Overview")
    st.write(
        "A quick, plain-language view of what data is available, how it was cleaned, and what each metric means."
    )

    data_cols = st.columns(4)
    data_cols[0].metric("Countries", f"{clean_df['Country'].nunique()}")
    data_cols[1].metric("Years", f"{clean_df['Year'].nunique()}")
    data_cols[2].metric("Records", f"{len(clean_df)}")
    data_cols[3].metric("Metrics", "8 indicators")

    st.markdown("---")

    st.subheader("📊 Descriptive Statistics")
    st.markdown(
        "**What this shows:** A statistical summary of each numeric variable across all countries and years. "
        "This comes from **Notebook 02** (EDA). The mean is the average, std shows how spread out values are, "
        "and the quartiles (25%, 50%, 75%) show how the data is distributed."
    )
    numeric_cols_desc = ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita", "Sea_Level_Rise_mm",
                         "Rainfall_mm", "Renewable_Energy_pct", "Extreme_Weather_Events", "Forest_Area_pct"]
    desc_df = clean_df[numeric_cols_desc].describe().T
    desc_df.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
    st.dataframe(desc_df.round(2), use_container_width=True)
    st.caption(
        "💡 **Reading guide:** High 'Std Dev' means the variable varies a lot across countries. "
        "Compare 'Min' and 'Max' to see the full range. If Mean ≠ Median, the distribution is skewed."
    )

    st.markdown("---")

    st.subheader("📊 Variable Distributions (Histograms)")
    st.markdown(
        "**What this shows:** Each small chart below is a histogram — it groups all country-year records into "
        "bins and counts how many fall in each range. The red dashed line marks the overall average (mean).\n\n"
        "**How to read them:** A tall bar means many observations fall in that range. If the chart looks like a "
        "bell shape, most values are close to the average. If it has a long tail stretching to the right, a few "
        "records are much higher than typical (right-skewed). If the tail stretches left, a few are much lower "
        "(left-skewed). Wider spread means bigger differences between countries.\n\n"
        "These distributions (from **Notebook 02**) help decide which statistical tests are appropriate in later analysis."
    )
    hist_cols_list = ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita", "Sea_Level_Rise_mm",
                      "Rainfall_mm", "Renewable_Energy_pct", "Extreme_Weather_Events", "Forest_Area_pct"]
    # Filter to columns that actually exist
    hist_cols_list = [c for c in hist_cols_list if c in clean_df.columns]
    n_cols = 3
    n_rows = -(-len(hist_cols_list) // n_cols)  # ceiling division

    hist_fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[LABEL_MAP.get(c, c.replace("_", " ")) for c in hist_cols_list],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )
    for idx, col_name in enumerate(hist_cols_list):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        col_data = clean_df[col_name].dropna()
        col_mean = col_data.mean()
        hist_fig.add_trace(
            go.Histogram(x=col_data, nbinsx=20, marker_color="#2E7D32",
                         marker_line_color="#1B5E20", marker_line_width=1,
                         opacity=0.7, showlegend=False),
            row=r, col=c,
        )
        # Dashed mean line
        hist_fig.add_vline(
            x=col_mean, line_dash="dash", line_color="#6D4C41", line_width=1.5,
            annotation_text=f"Mean: {col_mean:.1f}",
            annotation_font_size=10, annotation_font_color="#6D4C41",
            row=r, col=c,
        )
    hist_fig.update_layout(
        title_text="Distribution of Numeric Variables", title_font_size=16, title_x=0.5,
        height=300 * n_rows, margin=dict(t=60, b=30, l=40, r=40),
    )
    st.plotly_chart(hist_fig, use_container_width=True)
    st.caption(
        "💡 **Quick takeaway:** Temperature is roughly bell-shaped, meaning most countries sit near the global average. "
        "CO₂ emissions and rainfall are right-skewed — a handful of high-emitting or very wet countries pull the tail out. "
        "Renewable energy is also skewed because only a few countries have pushed their share above 50 %."
    )

    st.markdown("---")

    st.subheader("📦 Box Plots by Country")
    st.markdown(
        "**What this shows:** Each box represents one country's data across all available years. "
        "The bottom edge of the box is the 25th percentile, the top edge is the 75th percentile, and the "
        "horizontal line inside is the median (middle value). Whiskers extend to 1.5× the box height; "
        "dots beyond them are outliers — unusually high or low observations.\n\n"
        "**How to read them:** A tall, stretched box means that country's values varied a lot over the years. "
        "A short, compact box means the metric stayed relatively stable. If one country's box sits much higher "
        "or lower than the rest, it is an outlier compared to other nations for that metric."
    )
    box_metrics = ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita",
                   "Renewable_Energy_pct", "Extreme_Weather_Events"]
    box_metrics = [bm for bm in box_metrics if bm in clean_df.columns]
    box_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[LABEL_MAP.get(bm, bm.replace("_", " ")) for bm in box_metrics],
        horizontal_spacing=0.08, vertical_spacing=0.14,
    )
    countries_sorted = sorted(clean_df["Country"].unique())
    for idx, bm in enumerate(box_metrics):
        r = idx // 2 + 1
        c = idx % 2 + 1
        for ci_idx, country in enumerate(countries_sorted):
            vals = clean_df.loc[clean_df["Country"] == country, bm].dropna()
            box_fig.add_trace(
                go.Box(y=vals, name=country, showlegend=False,
                       marker_color=ENV_COLORS[ci_idx % len(ENV_COLORS)],
                       line_color=ENV_COLORS[ci_idx % len(ENV_COLORS)]),
                row=r, col=c,
            )
        box_fig.update_xaxes(tickangle=-90, tickfont_size=8, row=r, col=c)
        box_fig.update_yaxes(title_text=LABEL_MAP.get(bm, bm.replace("_", " ")),
                             title_font_size=9, row=r, col=c)
    box_fig.update_layout(
        title_text="Key Metrics by Country", title_font_size=16, title_x=0.5,
        height=900, margin=dict(t=60, b=80, l=50, r=30),
    )
    st.plotly_chart(box_fig, use_container_width=True)
    st.caption(
        "💡 **Key takeaway:** CO₂ emissions boxes vary enormously — industrialised nations like the United States "
        "and Australia sit far above developing nations like India and Nigeria. Renewable energy shows a similar "
        "spread, reflecting different stages of energy transition. Temperature boxes highlight geographic differences "
        "(tropical vs. temperate countries), while extreme-weather boxes reveal which nations face the greatest hazard exposure."
    )

    st.markdown("---")

    st.subheader("✅ Data Quality Assessment")
    st.markdown(
        "The dataset was cleaned in **Notebook 01** using a 5-step process: "
        "(1) Duplicate Country-Year rows removed, "
        "(2) Rows with missing temperature dropped, "
        "(3) Remaining missing numeric values filled with country-level medians, "
        "(4) Data types validated (Year and Population as integers), "
        "(5) All numeric columns checked against expected value ranges."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        missing_count = clean_df.isnull().sum().sum()
        missing_pct = (missing_count / (len(clean_df) * len(clean_df.columns))) * 100
        st.info(f"**Missing values:** {missing_count} ({missing_pct:.2f}%)")
    with col2:
        duplicate_count = clean_df.duplicated().sum()
        st.info(f"**Duplicate rows:** {duplicate_count}")
    with col3:
        st.info(f"**Data integrity:** ✅ Clean")

    # Missing values by column
    st.markdown("**Missing values by column:**")
    missing_by_col = clean_df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    if len(missing_by_col) > 0:
        missing_df = pd.DataFrame(
            {
                "Column": missing_by_col.index,
                "Missing Count": missing_by_col.values,
                "Percentage": (missing_by_col / len(clean_df) * 100).round(2),
            }
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values found!")

    st.markdown("---")

    st.subheader("📖 Key Fields")
    st.markdown("**Glossary:** Definitions and context for each metric")

    glossary_data = {
        "Field": [
            "Year",
            "Country",
            "Avg_Temperature_degC",
            "CO2_Emissions_tons_per_capita",
            "Sea_Level_Rise_mm",
            "Rainfall_mm",
            "Population",
            "Renewable_Energy_pct",
            "Extreme_Weather_Events",
            "Forest_Area_pct",
        ],
        "Description": [
            "Year of observation (5-year intervals: 2000, 2005, 2010, 2015, 2020, 2024)",
            "Country name",
            "Average temperature in degrees Celsius",
            "Per-capita CO2 emissions in metric tons",
            "Sea level rise in millimeters",
            "Annual rainfall in millimeters",
            "Total population",
            "Percentage of energy from renewable sources",
            "Count of extreme weather events reported",
            "Forest area as percentage of total land area",
        ],
        "Data Type": [
            "Integer",
            "String",
            "Float",
            "Float",
            "Float",
            "Float",
            "Integer",
            "Float",
            "Integer",
            "Float",
        ],
    }
    glossary_df = pd.DataFrame(glossary_data)
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("🗂️ Sample Data")
    st.markdown("**First 10 records from the dataset:**")
    display_df = clean_df.head(10).copy()
    display_df["Year"] = display_df["Year"].astype(int)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📥 Download Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 Full Dataset (CSV)",
            export_csv(clean_df),
            "environmental_trends_full.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "📋 Data Glossary (CSV)",
            export_csv(pd.DataFrame({
                "Metric": ["Year", "Country", "Avg_Temperature_degC", "CO2_Emissions_tons_per_capita",
                          "Sea_Level_Rise_mm", "Rainfall_mm", "Population", "Renewable_Energy_pct",
                          "Extreme_Weather_Events", "Forest_Area_pct"],
                "Definition": [
                    "Year of observation",
                    "Country name",
                    "Average temperature in Celsius",
                    "Per-capita CO2 emissions in tons",
                    "Sea level rise in millimeters",
                    "Annual rainfall in millimeters",
                    "Total population",
                    "Renewable energy share (%)",
                    "Count of extreme weather events",
                    "Forest area percentage"
                ]
            })),
            "data_glossary.csv",
            "text/csv"
        )
    with col3:
        st.download_button(
            "📊 Quality Report (CSV)",
            export_csv(pd.DataFrame({
                "Metric": ["Total Records", "Countries", "Year Range", "Complete Records", "Missing Values"],
                "Value": [len(clean_df), clean_df["Country"].nunique(), 
                         f"{clean_df['Year'].min()}-{clean_df['Year'].max()}",
                         f"{(1 - clean_df.isnull().sum().sum() / (clean_df.shape[0] * clean_df.shape[1]))*100:.2f}%",
                         clean_df.isnull().sum().sum()]
            })),
            "quality_report.csv",
            "text/csv"
        )

elif st.session_state.current_page == "Overview":
    st.subheader("📈 Overview — Key Signals")
    st.write(
        "How temperature, emissions, and renewable energy change over time across selected countries. "
        "This gives you a quick snapshot before diving deeper into patterns and predictions."
    )

    st.markdown("**Data Coverage:** How much data do we have?")
    coverage_cols = st.columns(3)
    with coverage_cols[0]:
        st.metric("Countries", f"{filtered_df['Country'].nunique()}")
        st.caption("Number of countries in your selection")
    with coverage_cols[1]:
        st.metric("Years", f"{filtered_df['Year'].nunique()}")
        st.caption("Years of data available")
    with coverage_cols[2]:
        st.metric("Records", f"{len(filtered_df)}")
        st.caption("Total data points")

    grouped = filtered_df.groupby("Year", as_index=False).mean(numeric_only=True)
    first_year = grouped.iloc[0]
    last_year = grouped.iloc[-1]

    st.markdown("**Latest Trends:**")
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        temp_change = last_year['Avg_Temperature_degC'] - first_year['Avg_Temperature_degC']
        st.metric(
            "Average Temperature",
            f"{last_year['Avg_Temperature_degC']:.2f}°C",
            f"{temp_change:+.2f}°C since {int(first_year['Year'])}",
            delta_color="inverse"
        )
        st.caption("🌡️ Higher = warmer; rising temperatures increase risk of heat stress, droughts, and ecosystem disruption.")
    with kpi_cols[1]:
        co2_change = last_year['CO2_Emissions_tons_per_capita'] - first_year['CO2_Emissions_tons_per_capita']
        st.metric(
            "CO2 per Person",
            f"{last_year['CO2_Emissions_tons_per_capita']:.2f} tons",
            f"{co2_change:+.2f} tons since {int(first_year['Year'])}",
            delta_color="inverse"
        )
        st.caption("♻️ Each person's share of emissions; higher = more energy/transport/industry reliance on fossil fuels.")
    with kpi_cols[2]:
        renew_change = last_year['Renewable_Energy_pct'] - first_year['Renewable_Energy_pct']
        st.metric(
            "Renewable Energy",
            f"{last_year['Renewable_Energy_pct']:.2f}%",
            f"{renew_change:+.2f}% since {int(first_year['Year'])}",
            delta_color="normal"
        )
        st.caption("⚡ Share of energy from wind, solar, hydro; higher = less dependence on coal/oil/gas.")

    st.subheader("📈 Temperature Over Time")
    st.markdown(
        "**What this shows:** The line plots the global average temperature for each year in your selection. "
        "An upward slope indicates a warming trend; a downward slope indicates cooling.\n\n"
        "**How to read it:** Focus on the overall direction rather than individual year-to-year dips, which are "
        "caused by natural variability. Even a gradual upward tilt of 0.5 °C over 25 years can disrupt agriculture, "
        "water supplies, and ecosystems."
    )
    line_fig = px.line(
        grouped,
        x="Year",
        y="Avg_Temperature_degC",
        title="Average temperature by year",
        markers=True,
        labels=LABEL_MAP,
    )
    st.plotly_chart(line_fig, use_container_width=True)
    st.caption(
        "💡 **How to read this:** An upward slope = warming trend. Even small increases (0.5°C) can disrupt "
        "agriculture, water systems, and ecosystems over decades. Year-to-year dips are normal (natural variability)."
    )

    latest_year = int(grouped["Year"].max())
    st.subheader("⛈️ Extreme Weather Events (Latest Year)")
    st.markdown(
        f"**What this shows:** A horizontal bar chart ranking the top 10 countries by the number of extreme weather "
        f"events (hurricanes, floods, droughts, heatwaves) recorded in **{latest_year}**.\n\n"
        "**How to read it:** Longer bars mean more events. Countries at the top of the chart face the greatest "
        "exposure to climate-related hazards and may benefit most from early-warning systems, disaster preparedness, "
        "and resilience investments."
    )
    latest_df = filtered_df[filtered_df["Year"] == latest_year]
    top_events = latest_df.nlargest(10, "Extreme_Weather_Events")
    bar_fig = px.bar(
        top_events,
        x="Extreme_Weather_Events",
        y="Country",
        orientation="h",
        title=f"Top 10 countries by extreme weather events in {latest_year}",
        labels=LABEL_MAP,
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption(
        "💡 **Interpretation:** Countries with the longest bars face the greatest exposure to climate hazards. "
        "This can inform where disaster preparedness and adaptation investments are most needed."
    )

    st.subheader("🔍 Quick Insights")
    country_delta = (
        filtered_df.sort_values("Year")
        .groupby("Country")
        .agg(first_year=("Year", "first"), last_year=("Year", "last"))
        .reset_index()
    )
    first_vals = filtered_df.merge(
        country_delta[["Country", "first_year"]],
        left_on=["Country", "Year"],
        right_on=["Country", "first_year"],
        how="inner",
    )
    last_vals = filtered_df.merge(
        country_delta[["Country", "last_year"]],
        left_on=["Country", "Year"],
        right_on=["Country", "last_year"],
        how="inner",
    )
    deltas = first_vals[["Country", "Avg_Temperature_degC"]].rename(
        columns={"Avg_Temperature_degC": "first_temp"}
    ).merge(
        last_vals[["Country", "Avg_Temperature_degC"]].rename(
            columns={"Avg_Temperature_degC": "last_temp"}
        ),
        on="Country",
        how="inner",
    )
    deltas["temp_change"] = deltas["last_temp"] - deltas["first_temp"]
    top_warming = deltas.sort_values("temp_change", ascending=False).head(5)
    st.markdown(
        "**Countries with the biggest temperature increases:**\n- "
        + "\n- ".join(
            f"{row['Country']}: +{row['temp_change']:.2f}°C (most at risk for climate impacts)" for _, row in top_warming.iterrows()
        )
    )
    st.markdown(
        "💡 **Next steps:** Use *Explore Patterns* to spot relationships between emissions and temperature, "
        "and use *Modeling & Prediction* to forecast future trends."
    )

    if show_technical:
        with st.expander("Data coverage notes"):
            st.write(
                "Coverage is limited to the countries present in the dataset. "
                "Trends are averages and may hide local variability."
            )

elif st.session_state.current_page == "Explore Patterns":
    st.subheader("🔍 Explore Patterns")
    st.write(
        "Find relationships between different metrics. If two things move together, it might mean one influences the other—or they might both be influenced by something else. "
        "These patterns are clues for deeper investigation, not proof of cause-and-effect."
    )

    st.subheader("🌍 Emissions vs. Temperature")
    st.markdown(
        "**What this shows:** A scatter plot where every dot is one country in one year. The horizontal axis is "
        "per-capita CO₂ emissions; the vertical axis is average temperature. Colours distinguish countries.\n\n"
        "**How to read it:** If dots form an upward diagonal, higher emissions are associated with warmer temperatures. "
        "If they form a downward diagonal, the opposite is true. If they're scattered randomly, there is no clear "
        "relationship. In this dataset the overall correlation is *negative* because cold-climate industrial nations "
        "(e.g. Russia, Canada) emit a lot but are naturally cold."
    )
    scatter_fig = px.scatter(
        filtered_df,
        x="CO2_Emissions_tons_per_capita",
        y="Avg_Temperature_degC",
        color="Country",
        title="CO2 emissions per capita vs average temperature",
        labels=LABEL_MAP,
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.caption(
        "⚠️ Note: Countries with different sizes and industries will have different patterns. "
        "This doesn't prove causation—just shows association.\n\n"
        "📊 **Hypothesis H1 result (from Notebook 03):** The Pearson correlation between CO2 emissions and temperature "
        "was found to be **moderately negative**, which is the opposite of what we expected. This happens because cold-climate "
        "industrial countries (e.g. Russia, Canada) have both high emissions and low temperatures. Geographic differences confound the relationship."
    )

    st.subheader("⚡ Renewable Energy vs. Emissions")
    st.markdown(
        "**What this shows:** Each dot plots a country-year's renewable energy share (horizontal) against its "
        "per-capita CO₂ emissions (vertical). Colours distinguish countries.\n\n"
        "**How to read it:** A downward-sloping cloud of dots means that countries with more renewable energy "
        "tend to emit less CO₂ per person. Countries in the top-left corner rely heavily on fossil fuels; those "
        "in the bottom-right have transitioned to cleaner energy."
    )
    scatter_fig2 = px.scatter(
        filtered_df,
        x="Renewable_Energy_pct",
        y="CO2_Emissions_tons_per_capita",
        color="Country",
        title="Renewable energy share vs CO2 emissions per capita",
        labels=LABEL_MAP,
    )
    st.plotly_chart(scatter_fig2, use_container_width=True)
    st.caption(
        "💡 Tip: Countries that invested in renewables earlier tend to have lower current emissions.\n\n"
        "📊 **Hypothesis H2 result (from Notebook 03):** The Pearson correlation between renewable energy share and CO2 emissions "
        "is **moderately negative**, supporting the hypothesis that higher renewables are associated with lower emissions."
    )

    st.subheader("🌧️ Rainfall Patterns")
    st.markdown(
        "**What this shows:** A histogram counting how many country-year records fall into each rainfall range.\n\n"
        "**How to read it:** A tall bar on the left means many observations have low rainfall (dry climates); a tall bar "
        "on the right means many have high rainfall (tropical or monsoon climates). A wide, flat shape suggests rainfall "
        "varies greatly across the dataset. Extremely high values may signal flood risk, while very low values may "
        "indicate drought-prone regions."
    )
    hist_fig = px.histogram(
        filtered_df,
        x="Rainfall_mm",
        nbins=20,
        title="Rainfall distribution across selected countries and years",
        labels=LABEL_MAP,
    )
    st.plotly_chart(hist_fig, use_container_width=True)
    st.caption("� The bars show how many observations (country-year combinations) fall into each rainfall range.")

    st.subheader("⛈️ Extreme Weather Events Over Time")
    st.markdown(
        "**What this shows:** A line chart plotting the average number of extreme weather events per year across "
        "all selected countries.\n\n"
        "**How to read it:** An upward slope means extreme events are becoming more frequent over time — consistent "
        "with Hypothesis 3. A flat line would suggest no change; a downward slope would suggest events are decreasing. "
        "Note that part of any upward trend may reflect improved monitoring and reporting in recent decades, not "
        "just actual increases."
    )
    events_trend = filtered_df.groupby("Year", as_index=False)["Extreme_Weather_Events"].mean()
    trend_fig = px.line(
        events_trend,
        x="Year",
        y="Extreme_Weather_Events",
        title="Average extreme weather events per year",
        markers=True,
        labels=LABEL_MAP,
    )
    st.plotly_chart(trend_fig, use_container_width=True)
    st.caption(
        "⚠️ Note: Increasing trends may reflect both actual climate changes and improved monitoring/reporting systems over time.\n\n"
        "📊 **Hypothesis H3 result (from Notebook 03):** A linear regression of extreme events over time shows an **upward slope**, "
        "indicating that extreme weather events have generally increased from 2000 to 2024."
    )

    st.subheader("🌳 Forest Area vs. Extreme Weather Events")
    st.markdown(
        "**What this shows:** Each dot plots a country-year's forest coverage (horizontal) against its count of "
        "extreme weather events (vertical). Colours distinguish countries.\n\n"
        "**How to read it:** A clear downward diagonal would suggest more forest coverage is associated with fewer "
        "extreme events. In practice the dots are scattered with no strong pattern, meaning forest area alone does not "
        "reliably predict how many extreme events a country experiences (Hypothesis 4)."
    )
    scatter_fig3 = px.scatter(
        filtered_df,
        x="Forest_Area_pct",
        y="Extreme_Weather_Events",
        color="Country",
        title="Forest area vs extreme weather events",
        labels=LABEL_MAP,
    )
    st.plotly_chart(scatter_fig3, use_container_width=True)
    st.caption(
        "💡 Insight: Forests provide local benefits (soil stability, flood control) but don't significantly reduce "
        "country-level extreme weather event counts, which are influenced by global climate systems.\n\n"
        "📊 **Hypothesis H4 result (from Notebook 03):** The Pearson correlation between forest area and extreme events "
        "is **weak**, confirming that forest coverage alone is not a strong predictor of extreme weather at the national level."
    )

    # ── Hypothesis Summary Table (from Notebook 03) ──
    st.markdown("---")
    st.subheader("📋 Hypothesis Test Summary")
    st.markdown(
        "**What this table shows:** The statistical results from all four hypothesis tests performed in **Notebook 03**. "
        "Pearson r measures the strength and direction of the relationship (-1 to +1). "
        "A p-value below 0.05 means the result is unlikely due to chance."
    )

    h_results = []
    # H1
    h1_data = filtered_df.dropna(subset=["CO2_Emissions_tons_per_capita", "Avg_Temperature_degC"])
    h1_r, h1_p = stats.pearsonr(h1_data["CO2_Emissions_tons_per_capita"], h1_data["Avg_Temperature_degC"])
    h_results.append({"Hypothesis": "H1: CO₂ vs Temperature", "Pearson r": f"{h1_r:.3f}",
                      "p-value": f"{h1_p:.4e}", "Significant (p<0.05)": "Yes" if h1_p < 0.05 else "No",
                      "Direction": "Negative" if h1_r < 0 else "Positive",
                      "Strength": "Strong" if abs(h1_r) > 0.7 else "Moderate" if abs(h1_r) > 0.3 else "Weak"})
    # H2
    h2_data = filtered_df.dropna(subset=["Renewable_Energy_pct", "CO2_Emissions_tons_per_capita"])
    h2_r, h2_p = stats.pearsonr(h2_data["Renewable_Energy_pct"], h2_data["CO2_Emissions_tons_per_capita"])
    h_results.append({"Hypothesis": "H2: Renewables vs CO₂", "Pearson r": f"{h2_r:.3f}",
                      "p-value": f"{h2_p:.4e}", "Significant (p<0.05)": "Yes" if h2_p < 0.05 else "No",
                      "Direction": "Negative" if h2_r < 0 else "Positive",
                      "Strength": "Strong" if abs(h2_r) > 0.7 else "Moderate" if abs(h2_r) > 0.3 else "Weak"})
    # H3
    h3_trend = filtered_df.groupby("Year")["Extreme_Weather_Events"].mean().reset_index()
    h3_slope, h3_int, h3_r, h3_p, h3_se = stats.linregress(h3_trend["Year"], h3_trend["Extreme_Weather_Events"])
    h_results.append({"Hypothesis": "H3: Extreme Events Trend", "Pearson r": f"{h3_r:.3f}",
                      "p-value": f"{h3_p:.4e}", "Significant (p<0.05)": "Yes" if h3_p < 0.05 else "No",
                      "Direction": "Upward" if h3_slope > 0 else "Downward",
                      "Strength": "Strong" if abs(h3_r) > 0.7 else "Moderate" if abs(h3_r) > 0.3 else "Weak"})
    # H4
    h4_data = filtered_df.dropna(subset=["Forest_Area_pct", "Extreme_Weather_Events"])
    h4_r, h4_p = stats.pearsonr(h4_data["Forest_Area_pct"], h4_data["Extreme_Weather_Events"])
    h_results.append({"Hypothesis": "H4: Forest vs Extreme Events", "Pearson r": f"{h4_r:.3f}",
                      "p-value": f"{h4_p:.4e}", "Significant (p<0.05)": "Yes" if h4_p < 0.05 else "No",
                      "Direction": "Negative" if h4_r < 0 else "Positive",
                      "Strength": "Strong" if abs(h4_r) > 0.7 else "Moderate" if abs(h4_r) > 0.3 else "Weak"})

    h_summary_df = pd.DataFrame(h_results)
    st.dataframe(h_summary_df, use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ **Bonferroni correction note:** With 4 tests on the same data, the stricter threshold is 0.05 ÷ 4 = 0.0125. "
        "All tests measure **association, not causation**."
    )
    st.markdown(
        "**Plain-English summary:**\n"
        "- **H1:** The link between CO₂ and temperature is confounded by geography — cold industrial nations emit more but are colder.\n"
        "- **H2:** Countries with more renewable energy tend to have lower emissions — supports the energy transition narrative.\n"
        "- **H3:** Extreme weather events have generally increased over time, consistent with climate change expectations.\n"
        "- **H4:** Forest coverage alone does not strongly predict extreme weather events at the national level."
    )

    if show_technical:
        with st.expander("Interpretation guardrails"):
            st.write(
                "Correlations can be influenced by geography, development level, and reporting quality. "
                "Use these plots as signals, not proof of causation."
            )

elif st.session_state.current_page == "Modeling & Prediction":
    st.subheader("📉 Baseline Temperature Model")
    st.write(
        "📊 **What this does:** A simple, explainable model that learns the relationship between emissions, renewable energy, weather, and temperature. "
        "It splits data into a training period (to learn) and a test period (to verify). The goal is clarity over precision."
    )

    train_df, test_df = time_aware_split(clean_df)
    X_train, y_train = build_features(train_df)
    X_test, y_test = build_features(test_df)

    if y_train is not None and y_test is not None:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = model_metrics(y_test, y_pred)

        st.markdown("**Model Performance (on test data):**")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("MAE", f"{metrics['MAE']:.3f}°C")
            st.caption("Average prediction error. Lower is better. E.g., 0.5°C means most predictions are within ±0.5°C of actual.")
        with metric_cols[1]:
            st.metric("RMSE", f"{metrics['RMSE']:.3f}°C")
            st.caption("(Root Mean Squared Error) Penalizes large mistakes more. Another way to measure how far off predictions are.")
        with metric_cols[2]:
            st.metric("R² Score", f"{metrics['R2']:.3f}")
            st.caption("How much variation the model explains. 1.0 = perfect; 0.5 = explains half; 0 = no better than guessing average.")

        st.info(
            "💡 **What this means:** These numbers tell you if the model is trustworthy. If R² > 0.6 and MAE is small, predictions are fairly reliable. "
            "If R² < 0.4, the model is missing important factors and should be used with caution."
        )

        st.subheader("📋 Test Results")
        st.markdown("**Actual vs. Predicted temperatures on test data:**")
        results_df = test_df[["Year", "Country", "Avg_Temperature_degC"]].copy()
        results_df["Year"] = results_df["Year"].astype(int)
        results_df["Predicted_Avg_Temperature_degC"] = y_pred
        results_df["Error (°C)"] = (results_df["Avg_Temperature_degC"] - results_df["Predicted_Avg_Temperature_degC"]).round(3)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.caption("🔍 Check the Error column: small numbers mean good predictions; large numbers mean the model struggled.")

        # ── Per-Country Metrics (matching Notebook 04 approach) ──
        st.subheader("📊 Per-Country Model Performance")
        st.markdown(
            "**What this shows:** How a simple per-country linear model (Year → Temperature) performs, "
            "matching the approach in **Notebook 04**. Data is split at 2018: training on 2000–2018, "
            "testing on 2019–2024. R² requires at least 2 test points to compute."
        )
        split_year = 2018
        country_metrics_rows = []
        for c_name, c_group in clean_df.groupby("Country"):
            grp = c_group.sort_values("Year").drop_duplicates("Year")
            c_train = grp[grp["Year"] <= split_year]
            c_test = grp[grp["Year"] > split_year]
            if len(c_train) < 2 or len(c_test) < 1:
                continue
            c_model = LinearRegression()
            c_model.fit(c_train[["Year"]], c_train["Avg_Temperature_degC"])
            c_preds = c_model.predict(c_test[["Year"]])
            c_mae = mean_absolute_error(c_test["Avg_Temperature_degC"], c_preds)
            c_rmse = np.sqrt(mean_squared_error(c_test["Avg_Temperature_degC"], c_preds))
            if len(c_test) > 1:
                c_r2 = r2_score(c_test["Avg_Temperature_degC"], c_preds)
                r2_display = round(c_r2, 3)
            else:
                r2_display = "N/A (1 sample)"
            country_metrics_rows.append({"Country": c_name, "MAE (°C)": round(c_mae, 3),
                                          "RMSE (°C)": round(c_rmse, 3), "R²": r2_display,
                                          "Test samples": len(c_test)})
        if country_metrics_rows:
            cm_df = pd.DataFrame(country_metrics_rows).sort_values("MAE (°C)")
            st.dataframe(cm_df, use_container_width=True, hide_index=True)
            st.caption(
                "💡 **Reading guide:** R² > 0.6 = reliable; R² < 0.3 = model misses important factors; "
                "'N/A' means only 1 test point exists so R² cannot be computed. "
                "MAE < 1.0°C = useful for trends; MAE > 2.0°C = limited value."
            )
    else:
        st.warning("⚠️ Insufficient data for model training. Please check your selected countries and year range.")

    st.subheader("🔮 Future Temperature Forecasts")
    st.markdown(
        "**What to expect:** Based on recent trends, these lines show where each country's temperature might head. "
        "Upward slopes = warming forecast; flat lines = stable; downward = cooling (rare)."
    )
    if pred_df.empty:
        st.warning("⚠️ No prediction file found at data/processed/v1/model_predictions.csv. Run the modeling notebook to generate forecasts.")
    else:
        pred_filtered = pred_df[pred_df["Country"].isin(selected_countries)]
        if pred_filtered.empty:
            st.info("No predictions available for the selected countries.")
        else:
            pred_fig = px.line(
                pred_filtered,
                x="Year",
                y="Predicted_Avg_Temperature_degC",
                color="Country",
                title="Projected temperature (2025+)",
                labels=LABEL_MAP,
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            st.caption("⚠️ **Important:** These are based on past trends. They assume nothing changes. Real futures depend on policy, technology, and behavior.")

            # ── Confidence Interval Visualization (from Notebook 04) ──
            ci_df = load_ci_predictions()
            if not ci_df.empty:
                ci_filtered = ci_df[ci_df["Country"].isin(selected_countries)]
                if not ci_filtered.empty:
                    st.subheader("📊 Forecast Confidence Intervals")
                    st.markdown(
                        "**What this shows:** Each chart below plots a country's forecast line (blue) with a shaded band "
                        "representing the 95 % confidence interval, calculated via bootstrap resampling in **Notebook 04**.\n\n"
                        "**How to read them:** The shaded area is the range within which the true temperature is likely to "
                        "fall. If the band is narrow, the model is confident in its projection. If the band is wide, the "
                        "country's historical temperature has been volatile and the forecast carries more uncertainty. "
                        "Any point outside the band would be a surprising outcome given past trends."
                    )
                    ci_countries = ci_filtered["Country"].unique()
                    if len(ci_countries) > 6:
                        st.info(f"📊 Showing 6 of {len(ci_countries)} selected countries. Filter to fewer countries to see the rest.")
                    for ci_country in ci_countries[:6]:  # Limit to 6 to keep page manageable
                        c_ci = ci_filtered[ci_filtered["Country"] == ci_country]
                        ci_width = (c_ci["Upper_95CI"] - c_ci["Lower_95CI"]).mean()
                        ci_fig = go.Figure()
                        ci_fig.add_trace(go.Scatter(
                            x=c_ci["Year"], y=c_ci["Upper_95CI"],
                            mode="lines", line=dict(width=0), showlegend=False,
                        ))
                        ci_fig.add_trace(go.Scatter(
                            x=c_ci["Year"], y=c_ci["Lower_95CI"],
                            mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(46,125,50,0.15)", name="95% CI",
                        ))
                        ci_fig.add_trace(go.Scatter(
                            x=c_ci["Year"], y=c_ci["Predicted_Temperature"],
                            mode="lines+markers", name="Forecast",
                            line=dict(color="#2E7D32", width=2),
                        ))
                        title_text = f"{ci_country} — Forecast with 95% CI"
                        subtitle = ""
                        if ci_width < 0.01:
                            subtitle = (
                                "<br><sup style='color:gray'>CI band is too narrow to display — "
                                "this country's historical data is almost perfectly linear, "
                                "so every bootstrap resample produces the same model.</sup>"
                            )
                        ci_fig.update_layout(
                            title=title_text + subtitle,
                            xaxis_title="Year", yaxis_title="Temperature (°C)",
                            height=350, margin=dict(t=70 if subtitle else 50, b=30),
                        )
                        st.plotly_chart(ci_fig, use_container_width=True)
                    st.caption(
                        "💡 **Interpretation:** Narrow bands mean the model is confident; wide bands mean more uncertainty. "
                        "Countries with volatile temperature histories will have wider confidence intervals. "
                        "If no band is visible, the country's data is almost perfectly linear — every bootstrap "
                        "resample yields the same model, so the confidence interval has zero width."
                    )
            else:
                st.info(
                    "📊 **Confidence intervals** for these forecasts are available in the exported file "
                    "`model_predictions_with_ci.csv` (generated in Notebook 04). Run the notebook to see CI charts here."
                )

    if show_technical:
        with st.expander("Forecast limitations"):
            st.write(
                "Forecasts are linear trend extrapolations and do not include policy, technology, "
                "or emissions scenario changes. Treat results as short-term projections only."
            )

    st.subheader("💻 Custom Prediction Tool")
    st.write(
        "🎯 **What this does:** Enter specific values (emissions, renewables, extreme events, etc.) to estimate what temperature would be. "
        "Use this to explore 'what-if' scenarios—e.g., 'What if renewables doubled?' or 'What if extreme events tripled?'"
    )
    
    st.info(
        "💡 **Instructions:** Pick a country and year, then adjust each factor. Press 'Predict temperature' to see the model's estimate. "
        "All fields use recent actual values as defaults—change them to explore different scenarios."
    )

    full_X, full_y = build_features(clean_df)
    model_ready = False
    if full_y is not None and full_X is not None:
        full_model = LinearRegression()
        full_model.fit(full_X, full_y)
        model_ready = True
    else:
        st.warning("⚠️ Unable to build model - insufficient data. Please check your filters.")
        full_model = None

    tool_country_options = all_countries
    default_country = tool_country_options[0]
    country_data = filtered_df[filtered_df["Country"] == default_country].sort_values("Year")
    default_row = country_data.iloc[-1] if not country_data.empty else filtered_df.iloc[-1]

    with st.form("prediction_form"):
        pred_country = st.selectbox("📍 Country", tool_country_options, index=0)
        pred_year = st.number_input(
            "📅 Year",
            min_value=int(clean_df["Year"].min()),
            max_value=int(clean_df["Year"].max()) + 5,
            value=int(default_row["Year"]),
            step=1,
            help="Pick a year to predict for. Going beyond 2024 extrapolates current trends."
        )
        pred_co2 = st.number_input(
            "🏭 CO2 emissions per capita (tons)",
            value=float(default_row["CO2_Emissions_tons_per_capita"]),
            min_value=0.0,
            help="Higher = more emissions from energy, transport, industry. Try reducing this to see how important it is."
        )
        pred_sea = st.number_input(
            "🌊 Sea level rise (mm)",
            value=float(default_row["Sea_Level_Rise_mm"]),
            min_value=0.0,
            help="How much ocean levels are rising. Related to warming; higher values = more climate change impact."
        )
        pred_rain = st.number_input(
            "🌧️ Rainfall (mm)",
            value=float(default_row["Rainfall_mm"]),
            min_value=0.0,
            help="Annual rainfall. Major droughts have low values (~200mm); wet regions have 2000mm+."
        )
        pred_pop = st.number_input(
            "👥 Population",
            value=float(default_row["Population"]),
            min_value=0.0,
            step=1.0,
            help="Total population. Larger populations typically consume more energy and emit more."
        )
        pred_renew = st.number_input(
            "⚡ Renewable energy (%)",
            value=float(default_row["Renewable_Energy_pct"]),
            min_value=0.0,
            max_value=100.0,
            help="0% = all fossil fuels; 100% = all renewables. Try increasing this to reduce emissions."
        )
        pred_events = st.number_input(
            "⛈️ Extreme weather events",
            value=float(default_row["Extreme_Weather_Events"]),
            min_value=0.0,
            step=1.0,
            help="Count of hurricanes, floods, heatwaves, etc. More events = sign of climate instability."
        )
        pred_forest = st.number_input(
            "🌳 Forest area (%)",
            value=float(default_row["Forest_Area_pct"]),
            min_value=0.0,
            max_value=100.0,
            help="0% = no forests; 100% = completely forested. Forests absorb CO2 and regulate climate."
        )
        submitted = st.form_submit_button("🔮 Predict temperature")

    if submitted and model_ready and full_model is not None:
        input_data = {
            "Year": pred_year,
            "CO2_Emissions_tons_per_capita": pred_co2,
            "Sea_Level_Rise_mm": pred_sea,
            "Rainfall_mm": pred_rain,
            "Population": pred_pop,
            "Renewable_Energy_pct": pred_renew,
            "Extreme_Weather_Events": pred_events,
            "Forest_Area_pct": pred_forest,
        }
        input_X = build_scenario_row(input_data, pred_country, full_X.columns.tolist())
        pred_value = full_model.predict(input_X)[0]
        
        st.success(f"🌡️ **Predicted temperature: {pred_value:.2f}°C**")
        st.markdown(
            f"**Scenario:** {pred_country} in {int(pred_year)} with these conditions:\n"
            f"- CO₂: {pred_co2:.2f} tons/person | Renewables: {pred_renew:.1f}% | Forest: {pred_forest:.1f}%\n"
            f"- Extreme events: {int(pred_events)} | Rainfall: {pred_rain:.0f}mm\n\n"
            "**What does this mean?** This temperature is what the model expects based on the inputs you provided and historical patterns. "
            "Use this to compare different scenarios and understand which factors have the biggest influence on temperature."
        )

elif st.session_state.current_page == "Analytics Hub":
    st.subheader("📊 Analytics Hub")
    st.write(
        "🔬 Dive deep into data relationships, quality metrics, and advanced analysis. "
        "Use these tools to understand patterns and anomalies."
    )
    
    # Data Quality Dashboard
    st.markdown("---")
    st.subheader("📈 Data Quality Dashboard")
    
    quality_score = get_data_quality_score(clean_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Quality Score", f"{quality_score:.1f}/100 ⭐")
    with col2:
        missing_pct = (clean_df.isnull().sum().sum() / (clean_df.shape[0] * clean_df.shape[1])) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.2f}%")
    with col3:
        duplicates = clean_df.duplicated().sum()
        st.metric("Unique Records", f"{(1 - duplicates/len(clean_df))*100:.2f}%")
    
    _file_mtime = os.path.getmtime(CLEAN_PATH) if CLEAN_PATH.exists() else None
    if _file_mtime:
        st.info(f"✅ **Data last updated:** {datetime.fromtimestamp(_file_mtime).strftime('%B %d, %Y at %I:%M %p')}")
    else:
        st.info("✅ **Data last updated:** Unknown")
    
    # Correlation Heatmap
    st.markdown("---")
    st.subheader("🔗 Metric Correlations")
    st.markdown(
        "**What this shows:** A heatmap of Pearson correlations between every pair of environmental metrics. "
        "Each cell contains a number from −1 to +1.\n\n"
        "**How to read it:** **Red cells** (positive) mean the two variables rise together. **Blue cells** (negative) "
        "mean one goes up while the other goes down. **White cells** (near zero) indicate no linear relationship. "
        "Values above ±0.7 are strong; ±0.3–0.7 are moderate; below ±0.3 are weak."
    )
    
    corr_matrix = calculate_correlation_matrix(filtered_df)
    heatmap_labels = [LABEL_MAP.get(c, c.replace("_", " ")) for c in corr_matrix.columns]
    corr_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=heatmap_labels,
        y=heatmap_labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    corr_fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(corr_fig, use_container_width=True)
    st.caption(
        "💡 **How to read this:** Red cells = variables increase together; blue = one goes up, the other goes down; "
        "white = no relationship. Values close to ±1.0 are strong; near 0 are weak. Year is excluded to avoid "
        "spurious correlations between variables that simply both trend upward over time."
    )
    
    # Anomaly Detection
    st.markdown("---")
    st.subheader("🚨 Anomaly Detection")
    st.markdown(
        "**What this shows:** Records where a metric is more than 2 standard deviations away from that country's "
        "own average — in other words, values that are unusually high or low for that particular nation.\n\n"
        "**How to read it:** Each row in the table is an anomaly. A high anomaly may signal a one-off event "
        "(e.g. a heatwave year) or a data-quality issue worth investigating."
    )
    
    anomaly_col = st.selectbox("Select metric to check for anomalies", 
                              ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita", "Extreme_Weather_Events"])
    
    anomaly_df = detect_anomalies(filtered_df, anomaly_col, threshold=2.0)
    anomalies = anomaly_df[anomaly_df["is_anomaly"]][["Year", "Country", anomaly_col]].sort_values(anomaly_col, ascending=False)
    
    if len(anomalies) > 0:
        st.warning(f"🚨 Found {len(anomalies)} anomalies")
        anomalies_display = anomalies.head(10).copy()
        st.dataframe(anomalies_display, use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ No anomalies detected in {anomaly_col}")
    
    # Download Analysis
    st.markdown("---")
    st.subheader("💾 Downloads")
    download_cols = [c for c in anomaly_df.columns if c not in ("z_score", "is_anomaly")]
    st.download_button(
        "📥 Download Quality Report",
        export_csv(anomaly_df[download_cols]),
        "data_anomalies.csv",
        "text/csv"
    )

elif st.session_state.current_page == "Comparison Tool":
    st.subheader("🔄 Country Comparison Tool")
    st.write(
        "Compare environmental metrics side-by-side across countries. "
        "Identify leaders and laggards in climate action."
    )
    
    comparison_countries = st.multiselect(
        "Select countries to compare (choose 2-5)",
        all_countries,
        default=all_countries[:3],
        max_selections=5
    )
    
    if len(comparison_countries) < 2:
        st.warning("Please select at least 2 countries to compare.")
    else:
        comparison_df = filtered_df[filtered_df["Country"].isin(comparison_countries)].copy()
        
        if comparison_df.empty:
            st.warning("⚠️ No data available for the selected countries in the current filter range.")
            st.info("💡 Try expanding your year range or selecting different countries.")
        else:
            # Latest values comparison
            st.subheader("📊 Latest Year Comparison")
            st.markdown(
                "Each bar chart below compares the selected countries on one metric for the most recent year "
                "in the dataset. Taller bars represent higher values. Use these side-by-side comparisons to "
                "spot leaders and laggards at a glance."
            )
            latest_year = comparison_df["Year"].max()
            latest_data = comparison_df[comparison_df["Year"] == latest_year].sort_values("Country")
            
            metrics_to_compare = ["Avg_Temperature_degC", "CO2_Emissions_tons_per_capita", "Renewable_Energy_pct", "Extreme_Weather_Events"]
            
            for metric in metrics_to_compare:
                fig = px.bar(
                    latest_data,
                    x="Country",
                    y=metric,
                    title=f"{LABEL_MAP.get(metric, metric.replace('_', ' '))} in {latest_year}",
                    color="Country",
                    text_auto=".2f",
                    labels=LABEL_MAP,
                )
                st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "💡 **How to compare:** Taller bars = higher values for that metric. Look for which countries lead "
                "or lag in each area — this helps identify best practices and areas needing attention."
            )
            
            # Trends comparison
            st.subheader("📈 Trends Over Time")
            st.markdown(
                "Select a metric below to see how each country's value has evolved year by year. "
                "Converging lines mean countries are becoming more similar; diverging lines mean the gap "
                "is widening. Steeper slopes indicate faster change."
            )
            metric_choice = st.selectbox("Select metric to track", metrics_to_compare)
            
            trend_fig = px.line(
                comparison_df,
                x="Year",
                y=metric_choice,
                color="Country",
                markers=True,
                title=f"{LABEL_MAP.get(metric_choice, metric_choice)} trends",
                labels=LABEL_MAP,
            )
            st.plotly_chart(trend_fig, use_container_width=True)
            st.caption(
                "💡 **What to look for:** Converging lines mean countries are becoming more similar; "
                "diverging lines mean the gap is growing. Steeper slopes = faster change."
            )
            
            # Download comparison
            st.download_button(
                "📥 Download Comparison Data",
                export_csv(latest_data),
                "country_comparison.csv",
                "text/csv"
            )

elif st.session_state.current_page == "Scenario Builder":
    st.subheader("⚙️ Scenario Builder")
    st.write(
        "Create 'what-if' scenarios to explore how environmental factors correlate with temperature. "
        "Adjust multiple indicators to see their combined effect on temperature estimates."
    )

    st.warning(
        "⚠️ **Important:** This tool uses a multivariate regression model trained on "
        "historical data (2000–2024) to explore relationships between environmental "
        "factors and temperature. It shows **correlations, not causation**. Country "
        "geography (latitude, altitude, climate zone) is the dominant predictor; "
        "environmental-factor adjustments produce small incremental changes on top of "
        "that baseline. Use this for educational exploration, not policy decisions."
    )

    # ── Train scenario model on full dataset for stable encoding ──────────
    full_X_scenario, full_y_scenario = build_features(clean_df)
    scenario_ready = False
    scenario_model = None
    scenario_r2 = None
    scenario_mae = None

    if full_y_scenario is not None and full_X_scenario is not None:
        scenario_model = LinearRegression()
        scenario_model.fit(full_X_scenario, full_y_scenario)
        scenario_ready = True

        # Compute training-set metrics for transparency
        _y_pred_sc = scenario_model.predict(full_X_scenario)
        scenario_r2 = r2_score(full_y_scenario, _y_pred_sc)
        scenario_mae = float(np.mean(np.abs(full_y_scenario - _y_pred_sc)))
    else:
        st.warning("⚠️ Unable to build model for scenarios — insufficient data.")

    st.markdown("---")
    st.subheader("📋 Create Your Scenario")

    if not scenario_ready:
        st.warning("❌ Scenario builder is unavailable. Please adjust your filters.")
    else:
        # ── Model performance summary ────────────────────────────────────
        with st.expander("📈 Model performance & methodology", expanded=False):
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Training R²", f"{scenario_r2:.4f}" if scenario_r2 is not None else "N/A")
            with m2:
                st.metric("Training MAE", f"{scenario_mae:.3f} °C" if scenario_mae is not None else "N/A")
            with m3:
                st.metric("Training samples", f"{len(full_y_scenario)}")
            st.markdown(
                "**How it works:** A multivariate linear regression is fitted on all "
                "19 countries × 6 time periods. Features include Year, CO₂ emissions, "
                "sea-level rise, rainfall, population, renewable-energy share, extreme-"
                "weather events, forest area, and one-hot-encoded country indicators.\n\n"
                "**Key caveat:** Country indicators alone explain ~99.99 % of variance "
                "(each country has a characteristic base temperature driven by geography). "
                "The environmental sliders therefore produce **small, incremental** "
                "temperature shifts. This is realistic: a single country's policy change "
                "will not alter its base climate, but it can nudge the trend."
            )

        # ── Country selector (independent of sidebar) ────────────────────
        scenario_countries = sorted(clean_df["Country"].unique().tolist())
        col_top1, col_top2, col_top3 = st.columns(3)
        with col_top1:
            scenario_country = st.selectbox(
                "Country for scenario",
                scenario_countries,
                index=scenario_countries.index("United States")
                if "United States" in scenario_countries else 0,
                help="Choose the country whose baseline data will seed the scenario"
            )
        with col_top2:
            scenario_name = st.text_input(
                "Scenario name",
                "My Environmental Scenario",
                help="Give your scenario a descriptive name"
            )
        with col_top3:
            scenario_year = st.number_input(
                "Target year",
                value=2030,
                min_value=2025,
                max_value=2050,
                help="Select a future year for prediction (2025–2050)"
            )

        # ── Baseline: latest row for the chosen country ──────────────────
        country_rows = clean_df[clean_df["Country"] == scenario_country].sort_values("Year")
        baseline_row = country_rows.iloc[-1]  # most recent year
        baseline_year = int(baseline_row["Year"])

        st.caption(
            f"Baseline: **{scenario_country}** in **{baseline_year}** — "
            f"Temp {baseline_row['Avg_Temperature_degC']:.1f} °C, "
            f"CO₂ {baseline_row['CO2_Emissions_tons_per_capita']:.1f} t/cap, "
            f"Renewables {baseline_row['Renewable_Energy_pct']:.1f} %"
        )

        st.markdown("**Adjust these factors from current levels:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            co2_reduction = st.slider(
                "CO₂ change (%)", -50, 50, 0,
                help="% change from current level (negative = reduction)"
            )
            renew_increase = st.slider(
                "Renewable energy change (%)",
                -20, 50, 0,
                help="Change in renewable energy share"
            )
        with col2:
            forest_increase = st.slider(
                "Forest area change (%)",
                -20, 30, 0,
                help="Change in forest coverage percentage"
            )
            extreme_change = st.slider(
                "Extreme events change (%)",
                -50, 100, 0,
                help="Projected change in frequency of extreme weather events"
            )
        with col3:
            rainfall_change = st.slider(
                "Rainfall change (mm)",
                -200, 200, 0,
                help="Change in annual rainfall amount"
            )
            pop_growth = st.slider(
                "Population growth (%)",
                -10, 50, 0,
                help="Projected population growth rate"
            )

        if st.button("🚀 Run Scenario"):
            if scenario_model is None:
                st.error("Model not available. Please check your data.")
            else:
                # ── Build scenario input from country baseline ────────────
                scenario_data = {
                    "Year": scenario_year,
                    "CO2_Emissions_tons_per_capita": baseline_row["CO2_Emissions_tons_per_capita"] * (1 + co2_reduction / 100),
                    "Renewable_Energy_pct": min(100, baseline_row["Renewable_Energy_pct"] * (1 + renew_increase / 100)),
                    "Forest_Area_pct": min(100, baseline_row["Forest_Area_pct"] * (1 + forest_increase / 100)),
                    "Extreme_Weather_Events": max(0, baseline_row["Extreme_Weather_Events"] * (1 + extreme_change / 100)),
                    "Rainfall_mm": max(0, baseline_row["Rainfall_mm"] + rainfall_change),
                    "Population": baseline_row["Population"] * (1 + pop_growth / 100),
                    "Sea_Level_Rise_mm": baseline_row["Sea_Level_Rise_mm"],
                }

                train_cols = full_X_scenario.columns.tolist()
                scenario_X = build_scenario_row(
                    scenario_data, scenario_country, train_cols
                )
                pred_temp = scenario_model.predict(scenario_X)[0]

                # ── Baseline prediction (same country, same year) ─────────
                baseline_dict = baseline_row.to_dict()
                baseline_X = build_scenario_row(
                    baseline_dict, scenario_country, train_cols
                )
                baseline_temp = scenario_model.predict(baseline_X)[0]

                temp_diff = pred_temp - baseline_temp
                actual_temp = float(baseline_row["Avg_Temperature_degC"])

                # ── Display results ───────────────────────────────────────
                st.markdown("---")
                st.subheader("📊 Scenario Results")

                r1, r2_col, r3 = st.columns(3)
                with r1:
                    st.metric("Country", scenario_country)
                with r2_col:
                    st.metric("Target Year", int(scenario_year))
                with r3:
                    st.metric(
                        "Temperature Change",
                        f"{temp_diff:+.2f} °C",
                        delta=f"{temp_diff:+.2f} °C",
                        delta_color="inverse",
                    )

                st.info(
                    f"**{scenario_name} — {scenario_country} ({int(scenario_year)})**\n\n"
                    f"🌡️ Observed temperature ({baseline_year}): **{actual_temp:.2f} °C**\n"
                    f"📍 Model baseline estimate ({baseline_year}): {baseline_temp:.2f} °C\n"
                    f"🎯 Scenario estimate ({int(scenario_year)}): {pred_temp:.2f} °C\n"
                    f"📈 Projected change: {temp_diff:+.2f} °C\n\n"
                    f"**Interpretation:** The model predicts a {abs(temp_diff):.2f} °C "
                    f"{'increase' if temp_diff > 0 else 'decrease'} relative to the "
                    f"{baseline_year} baseline for {scenario_country}. The small gap "
                    f"between the observed temperature ({actual_temp:.2f} °C) and the "
                    f"model estimate ({baseline_temp:.2f} °C) is normal regression error. "
                    f"Most of the projected change comes from the Year trend; "
                    f"environmental-factor adjustments contribute smaller incremental "
                    f"changes. This reflects correlations in the training data, not "
                    f"causal predictions."
                )

                # ── Comparison bar chart ──────────────────────────────────
                comparison_data = pd.DataFrame({
                    "Scenario": [f"Baseline ({baseline_year})", scenario_name],
                    "Temperature (°C)": [baseline_temp, pred_temp],
                })

                fig = px.bar(
                    comparison_data,
                    x="Scenario",
                    y="Temperature (°C)",
                    color="Scenario",
                    color_discrete_sequence=["#1565C0", "#2E7D32"],
                    text_auto=".2f",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "💡 **Reading this chart:** The left bar is the baseline estimate "
                    "for the selected country's most recent year; the right bar is your "
                    "scenario. A lower bar means the adjustments you made are associated "
                    "with cooler temperatures in the model."
                )

                # ── Feature-contribution breakdown ────────────────────────
                with st.expander("🔍 What drove this change?", expanded=True):
                    feature_cols_ordered = [
                        "Year",
                        "CO2_Emissions_tons_per_capita",
                        "Sea_Level_Rise_mm",
                        "Rainfall_mm",
                        "Population",
                        "Renewable_Energy_pct",
                        "Extreme_Weather_Events",
                        "Forest_Area_pct",
                    ]
                    contrib_records = []
                    for i, feat in enumerate(feature_cols_ordered):
                        sc_val = float(scenario_X.iloc[0][feat])
                        bl_val = float(baseline_X.iloc[0][feat])
                        coef = scenario_model.coef_[i]
                        delta_val = sc_val - bl_val
                        contribution = delta_val * coef
                        label = LABEL_MAP.get(feat, feat)
                        contrib_records.append({
                            "Feature": label,
                            "Baseline": bl_val,
                            "Scenario": sc_val,
                            "Change": delta_val,
                            "Coefficient": coef,
                            "Temp contribution (°C)": contribution,
                        })
                    contrib_df = pd.DataFrame(contrib_records)
                    st.dataframe(
                        contrib_df.style.format({
                            "Baseline": "{:.2f}",
                            "Scenario": "{:.2f}",
                            "Change": "{:+.2f}",
                            "Coefficient": "{:.6f}",
                            "Temp contribution (°C)": "{:+.4f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Each row shows how much a single feature drove the temperature "
                        "change. **Temp contribution = Change × Coefficient**. Year "
                        "typically dominates because the model captures a global warming "
                        "trend of ~0.03 °C per year."
                    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "🌿 Global Environmental Trends 2000–2024 | "
    "Data: <a href='https://www.kaggle.com/datasets/adilshamim8/temperature' "
    "style='color:gray;'>Kaggle — Adil Shamim</a> | "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
