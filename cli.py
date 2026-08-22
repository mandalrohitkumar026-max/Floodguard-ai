"""
FloodGuard AI - Command Line Interface (CLI)
Enables quick terminal-based risk assessment and batch scenario evaluation.
"""

import argparse
import sys
import pandas as pd

from src.data.generator import generate_synthetic_flood_dataset, generate_spatial_grid_data
from src.data.loader import prepare_training_data
from src.models.trainer import FloodModelTrainer
from src.models.predictor import FloodPredictor
from src.risk.classifier import RiskClassifier


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="FloodGuard AI Command Line Risk Predictor")
    parser.add_argument("--rain1h", type=float, default=25.0, help="1-hour rainfall in mm")
    parser.add_argument("--rain24h", type=float, default=120.0, help="24-hour cumulative rainfall in mm")
    parser.add_argument("--rain72h", type=float, default=250.0, help="72-hour cumulative rainfall in mm")
    parser.add_argument("--discharge", type=float, default=950.0, help="River discharge in m3/s")
    parser.add_argument("--stage", type=float, default=6.8, help="River stage height in meters")
    parser.add_argument("--elevation", type=float, default=18.0, help="Terrain elevation in meters")
    parser.add_argument("--slope", type=float, default=3.5, help="Terrain slope in degrees")
    parser.add_argument("--dist", type=float, default=350.0, help="Distance to river in meters")
    parser.add_argument("--soil", type=float, default=80.0, help="Soil moisture saturation in percent")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "random_forest"])

    args = parser.parse_args()

    print("\n=======================================================")
    print("       FLOODGUARD AI RISK ASSESSMENT SYSTEM            ")
    print("=======================================================")
    print(f"[*] Initializing and Training {args.model.upper()} Engine...")

    df = generate_synthetic_flood_dataset(n_samples=3000, random_state=42)
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    trainer = FloodModelTrainer(model_type=args.model, random_state=42)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    predictor = FloodPredictor(trainer)
    classifier = RiskClassifier()

    input_payload = {
        "rainfall_1h_mm": args.rain1h,
        "rainfall_24h_mm": args.rain24h,
        "rainfall_72h_mm": args.rain72h,
        "river_discharge_m3s": args.discharge,
        "river_stage_m": args.stage,
        "elevation_m": args.elevation,
        "slope_deg": args.slope,
        "dist_to_river_m": args.dist,
        "soil_moisture_pct": args.soil,
        "drainage_capacity_m3s": 75.0,
        "historical_flood_count": 4
    }

    result = predictor.predict_single(input_payload)
    prob = result["flood_probability"]
    alert = classifier.generate_alert(prob, location_name="Assessment Sector")

    print(f"\n[+] Model Accuracy: {metrics['accuracy']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f}")
    print("-------------------------------------------------------")
    print(f"[*] Inundation Probability: {prob * 100:.2f}%")
    print(f"[*] Threat Level:          {alert.risk_level.value}")
    print(f"[*] Headline:              {alert.headline}")
    print(f"[*] Summary:               {alert.description}")
    print(f"[*] Evacuation Mandatory:  {'YES [CRITICAL]' if alert.evacuation_recommended else 'NO [STANDBY]'}")
    print("-------------------------------------------------------")
    print("Actionable Advisories:")
    for a in alert.actions:
        print(f"  * {a}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
