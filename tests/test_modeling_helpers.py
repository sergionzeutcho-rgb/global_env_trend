import numpy as np
import pandas as pd

from utils.modeling_helpers import build_features, build_scenario_row, model_metrics


def test_build_features_adds_country_dummies_and_target():
    df = pd.DataFrame(
        {
            "Year": [2020, 2024],
            "Country": ["Australia", "United States"],
            "CO2_Emissions_tons_per_capita": [10.0, 14.0],
            "Sea_Level_Rise_mm": [20, 22],
            "Rainfall_mm": [1000, 1100],
            "Population": [25_000_000, 330_000_000],
            "Renewable_Energy_pct": [30.0, 20.0],
            "Extreme_Weather_Events": [10, 12],
            "Forest_Area_pct": [16.0, 33.0],
            "Avg_Temperature_degC": [14.0, 15.0],
        }
    )

    X, y = build_features(df)

    assert "Country_United States" in X.columns
    assert y is not None
    assert len(y) == 2


def test_build_scenario_row_sets_correct_country_dummy():
    training_columns = [
        "Year",
        "CO2_Emissions_tons_per_capita",
        "Sea_Level_Rise_mm",
        "Rainfall_mm",
        "Population",
        "Renewable_Energy_pct",
        "Extreme_Weather_Events",
        "Forest_Area_pct",
        "Country_Brazil",
        "Country_United States",
    ]
    data_dict = {
        "Year": 2024,
        "CO2_Emissions_tons_per_capita": 14.7,
        "Sea_Level_Rise_mm": 21,
        "Rainfall_mm": 1200,
        "Population": 330_000_000,
        "Renewable_Energy_pct": 18.5,
        "Extreme_Weather_Events": 25,
        "Forest_Area_pct": 33.0,
    }

    row = build_scenario_row(data_dict, "United States", training_columns)

    assert list(row.columns) == training_columns
    assert row.loc[0, "Country_United States"] == 1
    assert row.loc[0, "Country_Brazil"] == 0


def test_model_metrics_returns_expected_keys():
    y_true = pd.Series([10.0, 12.0, 14.0])
    y_pred = np.array([10.5, 11.5, 14.0])

    metrics = model_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] >= 0
