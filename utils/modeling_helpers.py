from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
