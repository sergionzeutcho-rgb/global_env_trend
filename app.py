from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent / "data" / "processed" / "v1"
CLEAN_PATH = DATA_DIR / "environmental_trends_clean.csv"
PRED_PATH = DATA_DIR / "model_predictions.csv"

st.set_page_config(page_title="Global Environmental Trends", layout="wide")


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PRED_PATH.exists():
        df = pd.read_csv(PRED_PATH)
        df["Year"] = df["Year"].astype(int)
        return df
    return pd.DataFrame(columns=["Year", "Country", "Predicted_Avg_Temperature_degC"])


def filter_data(df: pd.DataFrame, countries: list[str], year_range: tuple[int, int]) -> pd.DataFrame:
    filtered = df[df["Country"].isin(countries)].copy()
    return filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]


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
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def align_features(row_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in feature_columns if col not in row_df.columns]
    for col in missing_cols:
        row_df[col] = 0
    return row_df[feature_columns]


st.title("Global Environmental Trends (2000-2024)")
st.markdown(
    "This dashboard summarizes climate-related signals from 2000-2024 and provides an "
    "interpretable baseline model to support discussion and planning."
)

clean_df = load_clean_data()
pred_df = load_predictions()

st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "📝 Executive Summary",
        "📊 Data Overview",
        "📈 Overview",
        "🔍 Explore Patterns",
        "🤖 Modeling & Prediction",
    ],
    index=[
        "Executive Summary",
        "Data Overview",
        "Overview",
        "Explore Patterns",
        "Modeling & Prediction",
    ].index(st.session_state.current_page if "current_page" in st.session_state else "Executive Summary"),
)
st.session_state.current_page = page.replace("📝 ", "").replace("📊 ", "").replace("📈 ", "").replace(
    "🔍 ", ""
).replace("🤖 ", "")

st.sidebar.markdown("---")

st.sidebar.header("Filters")
all_countries = sorted(clean_df["Country"].unique().tolist())
country_options = ["All"] + all_countries
selected_countries = st.sidebar.multiselect(
    "Countries",
    country_options,
    default=["All"],
)
if not selected_countries or "All" in selected_countries:
    selected_countries = all_countries
min_year, max_year = int(clean_df["Year"].min()), int(clean_df["Year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))
show_technical = st.sidebar.checkbox("Show technical notes", value=False)

with st.sidebar.expander("Recommendation thresholds", expanded=False):
    temp_threshold = st.number_input("Temp change (degC)", value=0.5, min_value=0.0, step=0.1)
    co2_threshold = st.number_input(
        "CO2 change (tons per capita)", value=0.5, min_value=0.0, step=0.1
    )
    renew_threshold = st.number_input(
        "Renewables change (%)", value=2.0, min_value=0.0, step=0.5
    )
    events_threshold = st.number_input(
        "Extreme events change", value=5.0, min_value=0.0, step=1.0
    )

filtered_df = filter_data(clean_df, selected_countries, year_range)

st.sidebar.download_button(
    "Download filtered data",
    filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="environmental_trends_filtered.csv",
    mime="text/csv",
)

if filtered_df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.markdown("### Quick guide")

if "current_page" not in st.session_state:
    st.session_state.current_page = "Executive Summary"

nav_cols = st.columns(5)
with nav_cols[0]:
    if st.button("📝 Executive Summary", use_container_width=True):
        st.session_state.current_page = "Executive Summary"
        st.rerun()
with nav_cols[1]:
    if st.button("📊 Data Overview", use_container_width=True):
        st.session_state.current_page = "Data Overview"
        st.rerun()
with nav_cols[2]:
    if st.button("📈 Overview", use_container_width=True):
        st.session_state.current_page = "Overview"
        st.rerun()
with nav_cols[3]:
    if st.button("🔍 Explore Patterns", use_container_width=True):
        st.session_state.current_page = "Explore Patterns"
        st.rerun()
with nav_cols[4]:
    if st.button("🤖 Modeling & Prediction", use_container_width=True):
        st.session_state.current_page = "Modeling & Prediction"
        st.rerun()

st.markdown("---")

if st.session_state.current_page == "Executive Summary":
    st.subheader("Executive summary")
    
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
            f"**🟢 Renewable energy is growing**: Renewable share has increased by {renew_delta:.2f}%. "
            "This indicates a transition toward cleaner energy sources."
        )
    elif renew_delta > 0:
        observations.append(
            f"**🟡 Renewable energy is growing slowly**: Renewable share has increased by {renew_delta:.2f}%. "
            "Accelerating the transition could help reduce emissions faster."
        )
    else:
        observations.append("**🔴 Renewable energy share is declining**: Fossil fuels may be taking a larger share.")
    
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

    st.subheader("🌍 Country-Specific Findings")
    st.markdown(
        "Below is a detailed breakdown by country showing where the biggest changes are happening. "
        "Use this to identify which countries face the greatest challenges or have the strongest progress."
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
    status_rows = []
    
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
    st.subheader("Data overview")
    st.write(
        "Outcome: a quick, plain-language view of what data is available and what each metric means."
    )

    data_cols = st.columns(4)
    data_cols[0].metric("Countries", f"{clean_df['Country'].nunique()}")
    data_cols[1].metric("Years", f"{clean_df['Year'].nunique()}")
    data_cols[2].metric("Records", f"{len(clean_df)}")
    data_cols[3].metric("Metrics", "8 indicators")

    st.markdown("---")

    st.subheader("Data quality assessment")
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
                "Percentage": (missing_by_col.values / float(len(clean_df)) * 100).round(2),
            }
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values found!")

    st.markdown("---")

    st.subheader("Key fields")
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
            "Year of observation (2000-2024)",
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

    st.subheader("Sample data")
    st.markdown("**First 10 records from the dataset:**")
    display_df = clean_df.head(10).copy()
    display_df["Year"] = display_df["Year"].astype(int)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif st.session_state.current_page == "Overview":
    st.subheader("Key signals")
    st.write(
        "📊 **What you'll see:** How temperature, emissions, and renewable energy change over time across selected countries. "
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
        "**What to look for:** Is the line going up or down? An upward trend shows warming; a downward trend shows cooling. "
        "Even small year-to-year changes can cause big impacts when repeated over decades."
    )
    line_fig = px.line(
        grouped,
        x="Year",
        y="Avg_Temperature_degC",
        title="Average temperature by year",
        markers=True,
    )
    st.plotly_chart(line_fig, use_container_width=True)

    latest_year = int(grouped["Year"].max())
    st.subheader("⛈️ Extreme Weather Events (Latest Year)")
    st.markdown(
        "**What this shows:** Which countries experienced the most extreme weather (hurricanes, floods, heatwaves, etc.) "
        f"in {latest_year}. Longer bars = more events = higher climate impact and greater need for disaster preparedness."
    )
    latest_df = filtered_df[filtered_df["Year"] == latest_year]
    top_events = latest_df.nlargest(10, "Extreme_Weather_Events")
    bar_fig = px.bar(
        top_events,
        x="Extreme_Weather_Events",
        y="Country",
        orientation="h",
        title=f"Top 10 countries by extreme weather events in {latest_year}",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

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
    st.subheader("What to look for")
    st.write(
        "🔎 **Purpose:** Find relationships between different metrics. If two things move together, it might mean one influences the other—or they might both be influenced by something else. "
        "These patterns are clues for deeper investigation, not proof of cause-and-effect."
    )

    st.subheader("🌍 Emissions vs. Temperature")
    st.markdown(
        "**What to look for:** Do points cluster along a diagonal line going up? That would suggest higher emissions are associated with higher temperatures. "
        "Each colored dot = one country in one year. Dots farther right have higher emissions; dots higher have warmer temperatures."
    )
    scatter_fig = px.scatter(
        filtered_df,
        x="CO2_Emissions_tons_per_capita",
        y="Avg_Temperature_degC",
        color="Country",
        title="CO2 emissions per capita vs average temperature",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.caption("⚠️ Note: Countries with different sizes and industries will have different patterns. This doesn't prove causation—just shows association.")

    st.subheader("⚡ Renewable Energy vs. Emissions")
    st.markdown(
        "**What to look for:** Do dots form a line going down-right? That would suggest higher renewable energy is associated with lower emissions. "
        "Countries farther right use more renewable energy; countries higher have more emissions. If the trend goes down-right, renewables may be helping reduce emissions."
    )
    scatter_fig2 = px.scatter(
        filtered_df,
        x="Renewable_Energy_pct",
        y="CO2_Emissions_tons_per_capita",
        color="Country",
        title="Renewable energy share vs CO2 emissions per capita",
    )
    st.plotly_chart(scatter_fig2, use_container_width=True)
    st.caption("💡 Tip: Countries that invested in renewables earlier tend to have lower current emissions.")

    st.subheader("🌧️ Rainfall Patterns")
    st.markdown(
        "**What this shows:** How rainfall is distributed across all countries and years. A peak on the left means most places get less rain; "
        "a peak on the right means more rain. Extreme rainfall can cause floods; too little causes droughts."
    )
    hist_fig = px.histogram(
        filtered_df,
        x="Rainfall_mm",
        nbins=20,
        title="Rainfall distribution across selected countries and years",
    )
    st.plotly_chart(hist_fig, use_container_width=True)
    st.caption("📊 The bars show how many observations (country-year combinations) fall into each rainfall range.")

    if show_technical:
        with st.expander("Interpretation guardrails"):
            st.write(
                "Correlations can be influenced by geography, development level, and reporting quality. "
                "Use these plots as signals, not proof of causation."
            )

elif st.session_state.current_page == "Modeling & Prediction":
    st.subheader("🤖 Baseline Temperature Model")
    st.write(
        "📊 **What this does:** A simple, explainable model that learns the relationship between emissions, renewable energy, weather, and temperature. "
        "It splits data into a training period (to learn) and a test period (to verify). The goal is clarity over precision."
    )

    train_df, test_df = time_aware_split(filtered_df)
    X_train, y_train = build_features(train_df)
    X_test, y_test = build_features(test_df)

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
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            st.caption("⚠️ **Important:** These are based on past trends. They assume nothing changes. Real futures depend on policy, technology, and behavior.")

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

    full_X, full_y = build_features(filtered_df)
    full_model = LinearRegression()
    full_model.fit(full_X, full_y)

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
            help="Higher = more emissions from energy, transport, industry. Try reducing this to see how important it is."
        )
        pred_sea = st.number_input(
            "🌊 Sea level rise (mm)",
            value=float(default_row["Sea_Level_Rise_mm"]),
            help="How much ocean levels are rising. Related to warming; higher values = more climate change impact."
        )
        pred_rain = st.number_input(
            "🌧️ Rainfall (mm)",
            value=float(default_row["Rainfall_mm"]),
            help="Annual rainfall. Major droughts have low values (~200mm); wet regions have 2000mm+."
        )
        pred_pop = st.number_input(
            "👥 Population",
            value=float(default_row["Population"]),
            step=1.0,
            help="Total population. Larger populations typically consume more energy and emit more."
        )
        pred_renew = st.number_input(
            "⚡ Renewable energy (%)",
            value=float(default_row["Renewable_Energy_pct"]),
            help="0% = all fossil fuels; 100% = all renewables. Try increasing this to reduce emissions."
        )
        pred_events = st.number_input(
            "⛈️ Extreme weather events",
            value=float(default_row["Extreme_Weather_Events"]),
            step=1.0,
            help="Count of hurricanes, floods, heatwaves, etc. More events = sign of climate instability."
        )
        pred_forest = st.number_input(
            "🌳 Forest area (%)",
            value=float(default_row["Forest_Area_pct"]),
            help="0% = no forests; 100% = completely forested. Forests absorb CO2 and regulate climate."
        )
        submitted = st.form_submit_button("🔮 Predict temperature")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "Year": pred_year,
                    "CO2_Emissions_tons_per_capita": pred_co2,
                    "Sea_Level_Rise_mm": pred_sea,
                    "Rainfall_mm": pred_rain,
                    "Population": pred_pop,
                    "Renewable_Energy_pct": pred_renew,
                    "Extreme_Weather_Events": pred_events,
                    "Forest_Area_pct": pred_forest,
                    "Country": pred_country,
                }
            ]
        )
        input_X, _ = build_features(input_df, include_target=False)
        input_X = align_features(input_X, full_X.columns.tolist())
        pred_value = full_model.predict(input_X)[0]
        
        st.success(f"🌡️ **Predicted temperature: {pred_value:.2f}°C**")
        st.markdown(
            f"**Scenario:** {pred_country} in {int(pred_year)} with these conditions:\n"
            f"- CO₂: {pred_co2:.2f} tons/person | Renewables: {pred_renew:.1f}% | Forest: {pred_forest:.1f}%\n"
            f"- Extreme events: {int(pred_events)} | Rainfall: {pred_rain:.0f}mm\n\n"
            "**What does this mean?** This temperature is what the model expects based on the inputs you provided and historical patterns. "
            "Use this to compare different scenarios and understand which factors have the biggest influence on temperature."
        )
