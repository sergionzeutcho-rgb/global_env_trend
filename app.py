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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    y = df["Avg_Temperature_degC"].copy()
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

st.sidebar.header("Filters")
all_countries = sorted(clean_df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect("Countries", all_countries, default=all_countries)
min_year, max_year = int(clean_df["Year"].min()), int(clean_df["Year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

filtered_df = filter_data(clean_df, selected_countries, year_range)

if filtered_df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

overview_tab, eda_tab, modeling_tab = st.tabs(["Overview", "EDA", "Modeling"])

with overview_tab:
    st.subheader("Key signals")
    st.write(
        "Outcome: quick view of how temperature, emissions, and renewable energy change over time "
        "for the selected countries."
    )

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

with eda_tab:
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

with modeling_tab:
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

    st.subheader("Prediction tool")
    st.write(
        "Outcome: estimate average temperature for a specific scenario using the baseline model. "
        "Use this as a directional indicator only."
    )

    full_X, full_y = build_features(filtered_df)
    full_model = LinearRegression()
    full_model.fit(full_X, full_y)

    tool_country_options = selected_countries if selected_countries else all_countries
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
        input_X, _ = build_features(input_df)
        input_X = align_features(input_X, full_X.columns.tolist())
        pred_value = full_model.predict(input_X)[0]
        st.success(f"Predicted average temperature: {pred_value:.2f} degC")
