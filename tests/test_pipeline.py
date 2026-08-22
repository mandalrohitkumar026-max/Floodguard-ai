"""Comprehensive Integration & Unit Tests for FloodGuard AI Pipeline."""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate_synthetic_flood_dataset, generate_spatial_grid_data
from src.data.loader import prepare_training_data
from src.features.engineer import HydrologicFeatureEngineer
from src.models.trainer import FloodModelTrainer
from src.models.predictor import FloodPredictor
from src.risk.classifier import RiskClassifier, RiskLevel
from src.routing.evac_router import EvacuationRouter, EmergencyShelter


def test_synthetic_data_generator():
    df = generate_synthetic_flood_dataset(n_samples=200, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "rainfall_24h_mm" in df.columns
    assert "river_stage_m" in df.columns
    assert "flood_occurred" in df.columns
    assert df["flood_occurred"].isin([0, 1]).all()
    assert not df.isnull().any().any()


def test_spatial_grid_generator():
    grid = generate_spatial_grid_data(center_lat=28.6, center_lon=77.2, grid_size=7)
    assert len(grid) == 49
    assert "zone_id" in grid.columns
    assert "elevation_m" in grid.columns
    assert "dist_to_river_m" in grid.columns


def test_hydrologic_feature_engineering():
    raw_df = pd.DataFrame([{
        "rainfall_1h_mm": 20.0,
        "rainfall_24h_mm": 100.0,
        "rainfall_72h_mm": 250.0,
        "river_discharge_m3s": 800.0,
        "river_stage_m": 6.5,
        "elevation_m": 15.0,
        "slope_deg": 4.0,
        "dist_to_river_m": 400.0,
        "soil_moisture_pct": 80.0,
        "drainage_capacity_m3s": 60.0,
        "historical_flood_count": 3
    }])
    fe = HydrologicFeatureEngineer()
    transformed = fe.transform(raw_df)

    assert "rainfall_intensity_ratio" in transformed.columns
    assert "runoff_potential_index" in transformed.columns
    assert "twi_proxy" in transformed.columns
    assert "hydraulic_inundation_pressure" in transformed.columns
    assert float(transformed["runoff_potential_index"].iloc[0]) > 0


def test_model_training_and_prediction():
    df = generate_synthetic_flood_dataset(n_samples=500, random_state=42)
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    trainer = FloodModelTrainer(model_type="random_forest", random_state=42)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    assert metrics["accuracy"] >= 0.70
    assert metrics["roc_auc"] >= 0.70
    assert len(metrics["feature_importances"]) > 0

    predictor = FloodPredictor(trainer)
    sample_input = X_test.iloc[0].to_dict()
    res = predictor.predict_single(sample_input)
    assert 0.0 <= res["flood_probability"] <= 1.0
    assert res["prediction"] in [0, 1]


def test_risk_classification():
    classifier = RiskClassifier()
    assert classifier.classify_probability(0.10) == RiskLevel.LOW
    assert classifier.classify_probability(0.35) == RiskLevel.MODERATE
    assert classifier.classify_probability(0.65) == RiskLevel.HIGH
    assert classifier.classify_probability(0.92) == RiskLevel.CRITICAL

    alert = classifier.generate_alert(0.92, location_name="Sector 4")
    assert alert.risk_level == RiskLevel.CRITICAL
    assert alert.evacuation_recommended is True
    assert len(alert.actions) > 0


def test_evacuation_routing():
    grid = generate_spatial_grid_data(center_lat=28.6, center_lon=77.2, grid_size=7)
    grid["flood_probability"] = 0.10  # Mild risk

    # Set center zones to high flood
    grid.loc[grid["dist_to_river_m"] < 500, "flood_probability"] = 0.90

    router = EvacuationRouter()
    router.initialize_default_shelters(center_lat=28.6, center_lon=77.2)
    router.build_network_from_grid(grid)

    start_lat = float(grid.iloc[0]["lat"])
    start_lon = float(grid.iloc[0]["lon"])

    plan = router.find_best_evacuation_route(start_lat, start_lon)
    assert plan is not None
    assert plan.total_distance_km > 0
    assert plan.estimated_time_mins > 0
    assert len(plan.route_coordinates) >= 2
    assert plan.shelter is not None
