"""Data loading and pre-processing pipeline for FloodGuard AI."""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from .generator import generate_synthetic_flood_dataset


def load_data(
    filepath: str = None,
    n_samples: int = 6000,
    save_if_generated: bool = True
) -> pd.DataFrame:
    """
    Loads flood dataset from file if available, otherwise generates high-fidelity synthetic data.
    """
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)

    df = generate_synthetic_flood_dataset(n_samples=n_samples)

    if filepath and save_if_generated:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

    return df


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset into features and targets for training and evaluation.
    """
    feature_cols = [
        "rainfall_1h_mm",
        "rainfall_24h_mm",
        "rainfall_72h_mm",
        "river_discharge_m3s",
        "river_stage_m",
        "elevation_m",
        "slope_deg",
        "dist_to_river_m",
        "soil_moisture_pct",
        "drainage_capacity_m3s",
        "historical_flood_count",
    ]

    X = df[feature_cols]
    y = df["flood_occurred"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
