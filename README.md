# 🌊 FloodGuard AI: Real-Time Flood Prediction, Risk Assessment & Evacuation System

FloodGuard AI is an end-to-end intelligent disaster management and early-warning platform that predicts flood inundation probabilities, classifies multi-tier hazards, maps risk zones, and computes flood-aware evacuation corridors to emergency shelters.

---

## 🏗️ Architecture Pipeline

```
DATA INGESTION (Rainfall + River Gauges + Topography/DEM + Flood History)
  ↓
DATA PROCESSING & VALIDATION
  ↓
HYDROLOGIC FEATURE ENGINEERING (TWI proxy, Runoff Index, Hydraulic Pressure)
  ↓
ML MODELING (XGBoost Classifier & Random Forest Ensemble)
  ↓
CALIBRATED FLOOD PROBABILITY (0.0 to 1.0)
  ↓
RISK CLASSIFICATION (Low / Moderate / High / Critical)
  ↓
GEOSPATIAL RISK MAPPING & EARLY-WARNING DISPATCH
  ↓
SHELTER ALLOCATION & SAFE EVACUATION ROUTING (NetworkX Dijkstra/A*)
```

---

## 🚀 Key Features

1. **Multi-Source Data Ingestion & Synthesis**:
   - Ingests 1h, 24h, and 72h precipitation, river discharge rates (\(m^3/s\)), river stage heights (\(m\)), DEM elevation, slope gradients, soil moisture saturation, and historical flood frequency.
2. **Domain-Specific Hydrologic Feature Engineering**:
   - Computes flash-flood burst ratios, SCS-CN soil runoff indices, Topographic Wetness Index (TWI) approximations, and hydrodynamic pressure metrics.
3. **Machine Learning Predictive Engine**:
   - High-performance **XGBoost** and **Random Forest** models trained with cross-validation and feature importance explainability.
4. **4-Tier Risk Categorization & Automated Alerting**:
   - 🟢 **LOW** (\(< 25\%\)): Normal baseline monitoring.
   - 🟡 **MODERATE** (\(25\% - 55\%\)): Flood Watch Advisory.
   - 🟠 **HIGH** (\(55\% - 80\%\)): Flood Warning, Go-Bags & route preparation.
   - 🔴 **CRITICAL** (\(\ge 80\%\)): Imminent Flash Flood Emergency, mandatory evacuation.
5. **Interactive GIS Command Center & Folium Mapping**:
   - Dynamic map rendering flood risk sectors, river path contours, inundation heatmaps, and elevated emergency shelters.
6. **Flood-Aware Evacuation Navigator**:
   - Graph network routing (NetworkX) that dynamically applies impedance penalties and road closures to inundated or high-risk sectors, finding the safest route to elevated shelters.

---

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/test_pipeline.py -v
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Floodguard-ai/
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
├── src/
│   ├── data/
│   │   ├── generator.py           # Meteorological & spatial data generator
│   │   └── loader.py              # Data loader & preprocessor
│   ├── features/
│   │   └── engineer.py            # Hydrological feature engineering
│   ├── models/
│   │   ├── trainer.py             # Model training (XGBoost, Random Forest)
│   │   └── predictor.py           # Inference & explainability engine
│   ├── risk/
│   │   └── classifier.py          # Risk level thresholding & alert generator
│   ├── routing/
│   │   └── evac_router.py         # NetworkX flood-aware evacuation routing
│   └── visualization/
│       └── map_builder.py         # Folium geospatial map builder
└── tests/
    └── test_pipeline.py           # Automated unit and integration tests
```
