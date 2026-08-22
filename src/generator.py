"""Synthetic Hydrological & Meteorological Dataset Generator for FloodGuard AI."""

import numpy as np
import pandas as pd


def generate_synthetic_flood_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic multi-factor dataset covering:
    - Rainfall (1h, 24h, 72h accumulation)
    - River Hydrology (discharge m3/s, river water stage m, flood warning stage)
    - Topography / Geography (Elevation m, slope deg, distance to river m, soil moisture %, drainage capacity)
    - History (Past flood occurrences in 10 years)
    - Target: flood_occurred (0 or 1), flood_severity_score (0.0 to 1.0)
    """
    np.random.seed(random_state)

    # 1. Meteorological features
    # Rainfall in mm: combination of normal days and intense monsoon / storm events
    is_heavy_rain = np.random.binomial(1, 0.25, n_samples)
    rainfall_1h_mm = np.where(
        is_heavy_rain == 1,
        np.random.gamma(shape=5.0, scale=8.0, size=n_samples),   # 15 - 80 mm/hr
        np.random.exponential(scale=3.0, size=n_samples)          # 0 - 15 mm/hr
    ).round(2)

    rainfall_24h_mm = (rainfall_1h_mm * np.random.uniform(4.0, 14.0, n_samples) + np.random.normal(10, 5, n_samples)).clip(0, 450).round(2)
    rainfall_72h_mm = (rainfall_24h_mm * np.random.uniform(1.4, 2.5, n_samples) + np.random.normal(15, 8, n_samples)).clip(0, 750).round(2)

    # 2. Hydrological features
    # River base flow + upstream runoff response
    base_discharge = np.random.uniform(50, 400, n_samples)
    runoff_factor = (rainfall_72h_mm * 1.8) + (rainfall_24h_mm * 2.5)
    river_discharge_m3s = (base_discharge + runoff_factor + np.random.normal(0, 30, n_samples)).clip(30, 3500).round(2)

    # River stage height (m): Normal is 2-4m, Danger level ~7.5m, Extreme ~11m
    river_stage_m = (1.5 + (river_discharge_m3s / 300.0) ** 0.8 + np.random.normal(0, 0.2, n_samples)).clip(0.5, 14.0).round(2)

    # 3. Topographical / Spatial features
    # Elevation: Low floodplain (5m - 25m) to elevated terrain (25m - 150m)
    elevation_m = np.random.exponential(scale=35.0, size=n_samples).clip(3.0, 180.0).round(2)
    slope_deg = np.random.uniform(0.5, 30.0, n_samples).round(2)
    dist_to_river_m = np.random.exponential(scale=1200.0, size=n_samples).clip(10.0, 8000.0).round(1)
    
    # Soil moisture (%): 10% (dry) to 100% (saturated)
    soil_moisture_pct = (20.0 + (rainfall_72h_mm * 0.18) + np.random.normal(0, 5, n_samples)).clip(10.0, 100.0).round(2)

    # Drainage capacity (m3/s / mm/hr absorption equivalent)
    drainage_capacity_m3s = np.random.uniform(30.0, 150.0, n_samples).round(1)

    # 4. Historical occurrences
    historical_flood_count = np.random.poisson(lam=np.clip(8.0 - (elevation_m / 15.0) - (dist_to_river_m / 1000.0), 0.2, 8.0)).astype(int)

    # 5. Physics-grounded Synthetic Target Generation
    # Hydrodynamic vulnerability score calculation
    vuln_score = (
        (rainfall_72h_mm / 180.0) * 0.30 +
        (river_stage_m / 7.5) * 0.35 +
        (np.clip(1.0 - (elevation_m / 45.0), 0, 1)) * 0.20 +
        (np.clip(1.0 - (dist_to_river_m / 2500.0), 0, 1)) * 0.15 +
        (soil_moisture_pct / 100.0) * 0.10 -
        (drainage_capacity_m3s / 150.0) * 0.10 +
        (historical_flood_count * 0.03) +
        np.random.normal(0, 0.08, n_samples)
    )

    # Normalize to flood severity index 0.0 to 1.0
    flood_severity_score = 1.0 / (1.0 + np.exp(-4.5 * (vuln_score - 0.75)))
    flood_occurred = (flood_severity_score >= 0.50).astype(int)

    df = pd.DataFrame({
        "rainfall_1h_mm": rainfall_1h_mm,
        "rainfall_24h_mm": rainfall_24h_mm,
        "rainfall_72h_mm": rainfall_72h_mm,
        "river_discharge_m3s": river_discharge_m3s,
        "river_stage_m": river_stage_m,
        "elevation_m": elevation_m,
        "slope_deg": slope_deg,
        "dist_to_river_m": dist_to_river_m,
        "soil_moisture_pct": soil_moisture_pct,
        "drainage_capacity_m3s": drainage_capacity_m3s,
        "historical_flood_count": historical_flood_count,
        "flood_severity_score": flood_severity_score.round(4),
        "flood_occurred": flood_occurred,
    })

    return df


def generate_spatial_grid_data(
    center_lat: float = 28.6139,
    center_lon: float = 77.2090,
    grid_size: int = 15,
    cell_spacing_km: float = 0.8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a geospatial matrix of coordinates with realistic topography,
    river channel alignment, elevation gradients, and spatial infrastructure.
    """
    np.random.seed(random_state)
    km_per_lat = 111.0
    km_per_lon = 111.0 * np.cos(np.radians(center_lat))

    half_grid = grid_size // 2
    records = []

    # River centerline runs approximately south-southeast across the grid
    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            lat = center_lat + (i * cell_spacing_km) / km_per_lat
            lon = center_lon + (j * cell_spacing_km) / km_per_lon

            # River curve approximation
            river_lon_at_lat = center_lon + 0.006 * np.sin(i * 0.4) + (i * 0.003)
            dist_to_river_km = np.abs(lon - river_lon_at_lat) * km_per_lon
            dist_to_river_m = float(np.clip(dist_to_river_km * 1000.0, 20.0, 10000.0))

            # Elevation is lowest along river and increases outward
            base_elev = 12.0 + (dist_to_river_km * 7.5) + np.random.normal(0, 1.5)
            elevation_m = max(4.0, round(base_elev, 1))
            slope_deg = round(max(0.5, 1.2 + dist_to_river_km * 0.8 + np.random.uniform(0, 2)), 1)
            soil_moisture_pct = round(np.clip(85.0 - (dist_to_river_km * 8.0) + np.random.normal(0, 4), 20.0, 98.0), 1)
            historical_flood_count = int(np.clip(7 - (elevation_m // 6), 0, 10))

            zone_name = f"Zone-{chr(65 + (i + half_grid) % 26)}{j + half_grid + 1}"

            records.append({
                "zone_id": zone_name,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "elevation_m": elevation_m,
                "slope_deg": slope_deg,
                "dist_to_river_m": dist_to_river_m,
                "soil_moisture_pct": soil_moisture_pct,
                "drainage_capacity_m3s": round(np.random.uniform(40, 120), 1),
                "historical_flood_count": historical_flood_count
            })

    return pd.DataFrame(records)
