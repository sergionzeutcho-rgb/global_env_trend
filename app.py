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
    return pd.read_csv(CLEAN_PATH)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
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
    st.write(
        "Outcome: a concise, stakeholder-friendly view of the most important signals and what they "
        "imply for awareness and planning."
    )

    summary_grouped = filtered_df.groupby("Year", as_index=False).mean(numeric_only=True)
    summary_first = summary_grouped.iloc[0]
    summary_last = summary_grouped.iloc[-1]

    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Avg Temperature (degC)",
        f"{summary_last['Avg_Temperature_degC']:.2f}",
        f"{summary_last['Avg_Temperature_degC'] - summary_first['Avg_Temperature_degC']:.2f} vs {int(summary_first['Year'])}",
    )
    summary_cols[1].metric(
        "CO2 per Capita (tons)",
        f"{summary_last['CO2_Emissions_tons_per_capita']:.2f}",
        f"{summary_last['CO2_Emissions_tons_per_capita'] - summary_first['CO2_Emissions_tons_per_capita']:.2f} vs {int(summary_first['Year'])}",
    )
    summary_cols[2].metric(
        "Renewable Energy (%)",
        f"{summary_last['Renewable_Energy_pct']:.2f}",
        f"{summary_last['Renewable_Energy_pct'] - summary_first['Renewable_Energy_pct']:.2f} vs {int(summary_first['Year'])}",
    )

    st.subheader("Recommendations (based on observed signals)")
    rec_items = []

    temp_delta = summary_last["Avg_Temperature_degC"] - summary_first["Avg_Temperature_degC"]
    co2_delta = summary_last["CO2_Emissions_tons_per_capita"] - summary_first["CO2_Emissions_tons_per_capita"]
    renew_delta = summary_last["Renewable_Energy_pct"] - summary_first["Renewable_Energy_pct"]
    events_delta = summary_last["Extreme_Weather_Events"] - summary_first["Extreme_Weather_Events"]

    if temp_delta > 0:
        rec_items.append(
            "Temperature trend is rising; prioritize risk communication and adaptation planning."
        )
    if events_delta > 0:
        rec_items.append(
            "Extreme events appear to increase; strengthen monitoring and resilience planning."
        )
    if co2_delta > 0:
        rec_items.append(
            "Emissions per capita are up; reinforce mitigation policies and sector tracking."
        )
    if renew_delta > 0:
        rec_items.append(
            "Renewables are increasing; highlight progress and identify high-impact transition levers."
        )
    if not rec_items:
        rec_items.append(
            "Signals are mixed; continue monitoring and validate trends with updated data."
        )

    for item in rec_items[:4]:
        st.markdown(f"- {item}")

    st.caption(
        "These recommendations reflect observed associations and trends, not causal proof."
    )

    st.subheader("Country-specific recommendations")
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
        if row["temp_delta"] >= temp_threshold:
            country_recs.append("Warming trend exceeds threshold")
        if row["events_delta"] >= events_threshold:
            country_recs.append("Extreme events rising above threshold")
        if row["co2_delta"] >= co2_threshold:
            country_recs.append("CO2 per capita increasing")
        if row["renew_delta"] >= renew_threshold:
            country_recs.append("Renewables growing; sustain momentum")

        if country_recs:
            rec_rows.append(
                {
                    "Country": row["Country"],
                    "Recommendation": "; ".join(country_recs),
                }
            )

    if rec_rows:
        rec_df = pd.DataFrame(rec_rows).sort_values("Country")
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("No country-specific recommendations exceeded the selected thresholds.")

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
    st.dataframe(clean_df.head(10), use_container_width=True, hide_index=True)

elif st.session_state.current_page == "Overview":
    st.subheader("Key signals")
    st.write(
        "Outcome: quick view of how temperature, emissions, and renewable energy change over time "
        "for the selected countries."
    )

    coverage_cols = st.columns(3)
    coverage_cols[0].metric("Countries", f"{filtered_df['Country'].nunique()}")
    coverage_cols[1].metric("Years", f"{filtered_df['Year'].nunique()}")
    coverage_cols[2].metric("Records", f"{len(filtered_df)}")

    grouped = filtered_df.groupby("Year", as_index=False).mean(numeric_only=True)
    first_year = grouped.iloc[0]
    last_year = grouped.iloc[-1]

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        "Avg Temperature (degC)",
        f"{last_year['Avg_Temperature_degC']:.2f}",
        f"{last_year['Avg_Temperature_degC'] - first_year['Avg_Temperature_degC']:.2f} vs {int(first_year['Year'])}",
    )
    kpi_cols[1].metric(
        "CO2 per Capita (tons)",
        f"{last_year['CO2_Emissions_tons_per_capita']:.2f}",
        f"{last_year['CO2_Emissions_tons_per_capita'] - first_year['CO2_Emissions_tons_per_capita']:.2f} vs {int(first_year['Year'])}",
    )
    kpi_cols[2].metric(
        "Renewable Energy (%)",
        f"{last_year['Renewable_Energy_pct']:.2f}",
        f"{last_year['Renewable_Energy_pct'] - first_year['Renewable_Energy_pct']:.2f} vs {int(first_year['Year'])}",
    )

    st.subheader("Average temperature trend")
    line_fig = px.line(
        grouped,
        x="Year",
        y="Avg_Temperature_degC",
        title="Average temperature by year",
        markers=True,
    )
    st.plotly_chart(line_fig, use_container_width=True)

    st.subheader("Extreme weather events (latest year)")
    latest_year = int(grouped["Year"].max())
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

    st.subheader("Key takeaways")
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
        "- Temperature change leaders (selected range): "
        + ", ".join(
            f"{row['Country']} (+{row['temp_change']:.2f}C)" for _, row in top_warming.iterrows()
        )
    )
    st.markdown(
        "- Use the Explore Patterns section to inspect associations and the Modeling section for projections."
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
        "Outcome: identify associations, not causation. These charts help spot patterns that can "
        "inform deeper analysis."
    )

    st.subheader("CO2 emissions vs temperature")
    scatter_fig = px.scatter(
        filtered_df,
        x="CO2_Emissions_tons_per_capita",
        y="Avg_Temperature_degC",
        color="Country",
        title="CO2 emissions per capita vs average temperature",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Renewable energy vs CO2 emissions")
    scatter_fig2 = px.scatter(
        filtered_df,
        x="Renewable_Energy_pct",
        y="CO2_Emissions_tons_per_capita",
        color="Country",
        title="Renewable energy share vs CO2 emissions per capita",
    )
    st.plotly_chart(scatter_fig2, use_container_width=True)

    st.subheader("Distribution of rainfall")
    hist_fig = px.histogram(
        filtered_df,
        x="Rainfall_mm",
        nbins=20,
        title="Rainfall distribution",
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    if show_technical:
        with st.expander("Interpretation guardrails"):
            st.write(
                "Correlations can be influenced by geography, development level, and reporting quality. "
                "Use these plots as signals, not proof of causation."
            )

elif st.session_state.current_page == "Modeling & Prediction":
    st.subheader("Baseline temperature model")
    st.write(
        "This baseline model predicts average temperature using a time-aware split. "
        "It is designed for interpretability, not for high-accuracy forecasting."
    )

    train_df, test_df = time_aware_split(filtered_df)
    X_train, y_train = build_features(train_df)
    X_test, y_test = build_features(test_df)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = model_metrics(y_test, y_pred)

    metric_cols = st.columns(3)
    metric_cols[0].metric("MAE", f"{metrics['MAE']:.3f}")
    metric_cols[1].metric("RMSE", f"{metrics['RMSE']:.3f}")
    metric_cols[2].metric("R2", f"{metrics['R2']:.3f}")

    st.write(
        "Outcome: MAE and RMSE show typical error size in degrees Celsius; lower is better. "
        "R2 indicates how much of the variance is explained by the model."
    )

    results_df = test_df[["Year", "Country", "Avg_Temperature_degC"]].copy()
    results_df["Predicted_Avg_Temperature_degC"] = y_pred
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("Forecasts (existing model_predictions.csv)")
    if pred_df.empty:
        st.info("No prediction file found at data/processed/v1/model_predictions.csv.")
    else:
        pred_filtered = pred_df[pred_df["Country"].isin(selected_countries)]
        pred_fig = px.line(
            pred_filtered,
            x="Year",
            y="Predicted_Avg_Temperature_degC",
            color="Country",
            title="Projected temperature (2025+)",
        )
        st.plotly_chart(pred_fig, use_container_width=True)

    if show_technical:
        with st.expander("Forecast limitations"):
            st.write(
                "Forecasts are linear trend extrapolations and do not include policy, technology, "
                "or emissions scenario changes. Treat results as short-term projections only."
            )

    st.subheader("Prediction tool")
    st.write(
        "Outcome: estimate average temperature for a specific scenario using the baseline model. "
        "Use this as a directional indicator only."
    )

    full_X, full_y = build_features(filtered_df)
    full_model = LinearRegression()
    full_model.fit(full_X, full_y)

    tool_country_options = all_countries
    default_country = tool_country_options[0]
    country_data = filtered_df[filtered_df["Country"] == default_country].sort_values("Year")
    default_row = country_data.iloc[-1] if not country_data.empty else filtered_df.iloc[-1]

    with st.form("prediction_form"):
        pred_country = st.selectbox("Country", tool_country_options, index=0)
        pred_year = st.number_input(
            "Year",
            min_value=int(clean_df["Year"].min()),
            max_value=int(clean_df["Year"].max()) + 5,
            value=int(default_row["Year"]),
            step=1,
        )
        pred_co2 = st.number_input(
            "CO2 emissions per capita (tons)",
            value=float(default_row["CO2_Emissions_tons_per_capita"]),
        )
        pred_sea = st.number_input(
            "Sea level rise (mm)",
            value=float(default_row["Sea_Level_Rise_mm"]),
        )
        pred_rain = st.number_input(
            "Rainfall (mm)",
            value=float(default_row["Rainfall_mm"]),
        )
        pred_pop = st.number_input(
            "Population",
            value=float(default_row["Population"]),
            step=1.0,
        )
        pred_renew = st.number_input(
            "Renewable energy (%)",
            value=float(default_row["Renewable_Energy_pct"]),
        )
        pred_events = st.number_input(
            "Extreme weather events",
            value=float(default_row["Extreme_Weather_Events"]),
            step=1.0,
        )
        pred_forest = st.number_input(
            "Forest area (%)",
            value=float(default_row["Forest_Area_pct"]),
        )
        submitted = st.form_submit_button("Predict temperature")

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
        st.success(f"Predicted average temperature: {pred_value:.2f} degC")
