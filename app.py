"""
FloodGuard AI - Real-Time Flood Prediction, Risk Classification, Alerting & Evacuation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium

from src.data.generator import generate_synthetic_flood_dataset, generate_spatial_grid_data
from src.data.loader import prepare_training_data
from src.models.trainer import FloodModelTrainer
from src.models.predictor import FloodPredictor
from src.risk.classifier import RiskClassifier, RiskLevel
from src.routing.evac_router import EvacuationRouter, EmergencyShelter
from src.visualization.map_builder import FloodMapBuilder

# --- Page Configuration ---
st.set_page_config(
    page_title="FloodGuard AI | Flood Prediction & Evacuation System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F4C81;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-box {
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Cached Resources & Model Initializer ---
@st.cache_resource(show_spinner="Training Flood Prediction ML Models...")
def init_models():
    # Generate baseline dataset
    df = generate_synthetic_flood_dataset(n_samples=5000, random_state=42)
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    # Train XGBoost
    xgb_trainer = FloodModelTrainer(model_type="xgboost", random_state=42)
    xgb_trainer.train(X_train, y_train)
    xgb_metrics = xgb_trainer.evaluate(X_test, y_test)

    # Train Random Forest
    rf_trainer = FloodModelTrainer(model_type="random_forest", random_state=42)
    rf_trainer.train(X_train, y_train)
    rf_metrics = rf_trainer.evaluate(X_test, y_test)

    return xgb_trainer, rf_trainer, xgb_metrics, rf_metrics, df


@st.cache_data
def get_spatial_grid():
    return generate_spatial_grid_data(center_lat=28.6139, center_lon=77.2090, grid_size=13, cell_spacing_km=0.9)


# Load Models & Base Data
xgb_trainer, rf_trainer, xgb_metrics, rf_metrics, historical_df = init_models()
grid_df = get_spatial_grid().copy()

# Initialize Router & Map Builder
router = EvacuationRouter()
router.initialize_default_shelters(center_lat=28.6139, center_lon=77.2090)
map_builder = FloodMapBuilder(center_lat=28.6139, center_lon=77.2090, zoom_start=12)
classifier = RiskClassifier()

# --- Sidebar Controls ---
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
    <span style="font-size: 2.2rem;">🌊</span>
    <h2 style="margin: 0; color: #0F4C81; font-weight: 800;">FloodGuard AI</h2>
</div>
""", unsafe_allow_html=True)

# Scenario Presets
st.sidebar.subheader("🕹️ Simulation Scenarios")
scenario = st.sidebar.selectbox(
    "Select Hydrological Scenario:",
    [
        "Custom Parameters",
        "☀️ Dry Normal Conditions",
        "🌧️ Moderate Monsoon Rain",
        "⚡ Heavy Downpour & High Runoff",
        "🌊 Extreme Flash Flood & Dam Surge"
    ]
)

# Preset Values
if scenario == "☀️ Dry Normal Conditions":
    preset_rain_1h, preset_rain_24h, preset_rain_72h = 1.5, 12.0, 20.0
    preset_discharge, preset_stage = 120.0, 2.3
    preset_soil = 30.0
elif scenario == "🌧️ Moderate Monsoon Rain":
    preset_rain_1h, preset_rain_24h, preset_rain_72h = 14.0, 65.0, 140.0
    preset_discharge, preset_stage = 550.0, 5.2
    preset_soil = 65.0
elif scenario == "⚡ Heavy Downpour & High Runoff":
    preset_rain_1h, preset_rain_24h, preset_rain_72h = 38.0, 160.0, 320.0
    preset_discharge, preset_stage = 1250.0, 7.8
    preset_soil = 85.0
elif scenario == "🌊 Extreme Flash Flood & Dam Surge":
    preset_rain_1h, preset_rain_24h, preset_rain_72h = 75.0, 290.0, 540.0
    preset_discharge, preset_stage = 2600.0, 11.2
    preset_soil = 98.0
else:
    preset_rain_1h, preset_rain_24h, preset_rain_72h = 25.0, 110.0, 210.0
    preset_discharge, preset_stage = 850.0, 6.4
    preset_soil = 75.0

st.sidebar.markdown("---")
st.sidebar.subheader("🌧️ Meteorological Inputs")
rain_1h = st.sidebar.slider("1-Hour Rainfall (mm/h)", 0.0, 120.0, float(preset_rain_1h), 1.0)
rain_24h = st.sidebar.slider("24-Hour Cumulative Rain (mm)", 0.0, 450.0, float(preset_rain_24h), 5.0)
rain_72h = st.sidebar.slider("72-Hour Cumulative Rain (mm)", 0.0, 800.0, float(preset_rain_72h), 10.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌊 Hydrological Inputs")
river_discharge = st.sidebar.slider("River Discharge (m³/s)", 20.0, 3500.0, float(preset_discharge), 20.0)
river_stage = st.sidebar.slider("River Stage Height (m)", 0.5, 14.0, float(preset_stage), 0.1)
soil_moisture = st.sidebar.slider("Soil Moisture Saturation (%)", 10.0, 100.0, float(preset_soil), 1.0)
drainage_capacity = st.sidebar.slider("Storm Drainage Capacity (m³/s)", 20.0, 150.0, 80.0, 5.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
selected_model_type = st.sidebar.radio("Active ML Model:", ["XGBoost Classifier", "Random Forest Classifier"])
active_trainer = xgb_trainer if "XGBoost" in selected_model_type else rf_trainer
active_metrics = xgb_metrics if "XGBoost" in selected_model_type else rf_metrics
predictor = FloodPredictor(active_trainer)

# --- Real-Time Prediction on Grid ---
# Compute predictions for all spatial grid cells
spatial_input_df = grid_df.copy()
spatial_input_df["rainfall_1h_mm"] = rain_1h
spatial_input_df["rainfall_24h_mm"] = rain_24h
spatial_input_df["rainfall_72h_mm"] = rain_72h
spatial_input_df["river_discharge_m3s"] = river_discharge
spatial_input_df["river_stage_m"] = river_stage
spatial_input_df["soil_moisture_pct"] = np.clip(soil_moisture + (spatial_input_df["soil_moisture_pct"] - 50.0) * 0.3, 10.0, 100.0)
spatial_input_df["drainage_capacity_m3s"] = drainage_capacity

# Predict on all zones
predicted_grid_df = predictor.predict_batch(spatial_input_df)

# Compute Global Summary Probability
mean_flood_prob = float(predicted_grid_df["flood_probability"].mean())
max_flood_prob = float(predicted_grid_df["flood_probability"].max())
overall_alert = classifier.generate_alert(max_flood_prob, location_name="Metropolitan Basin")

# Update router road network with live grid risk
router.build_network_from_grid(predicted_grid_df)

# --- Header Display ---
st.markdown('<div class="main-title">🌊 FloodGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Real-Time Flood Prediction, Risk Classification, and Evacuation Guidance System</div>', unsafe_allow_html=True)

# Top Threat Banner
risk_color = overall_alert.color_hex
st.markdown(
    f"""
    <div style="background-color: {risk_color}22; border-left: 6px solid {risk_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 18px;">
        <span style="font-size: 1.2rem; font-weight: 700; color: {risk_color};">{overall_alert.headline}</span><br/>
        <span style="color: #333333; font-size: 0.95rem;">{overall_alert.description}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Top KPIs Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Max Flood Probability", f"{max_flood_prob * 100:.1f}%", delta=f"{overall_alert.risk_level.value} RISK", delta_color="inverse" if max_flood_prob > 0.5 else "normal")
with col2:
    st.metric("River Stage Height", f"{river_stage:.1f} m", delta="Warning > 7.0m" if river_stage > 7.0 else "Normal")
with col3:
    st.metric("24h Rain Accumulation", f"{rain_24h:.0f} mm", delta="Heavy" if rain_24h > 100 else "Normal")
with col4:
    critical_count = len(predicted_grid_df[predicted_grid_df["flood_probability"] >= 0.80])
    st.metric("Critical Sectors", f"{critical_count} / {len(predicted_grid_df)}", delta="Hazardous" if critical_count > 0 else "Safe")
with col5:
    st.metric("Available Shelters", f"{len(router.shelters)} Active", delta="100% Ready")

st.markdown("---")

# --- Main Application Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geospatial Risk Map & Live Command Center",
    "🏃 Evacuation Route & Shelter Allocation",
    "🤖 ML Model Intelligence & Feature Importance",
    "📊 Sector Inundation & Hydrologic Analytics"
])

# -------------------------------------------------------------
# TAB 1: GEOSPATIAL MAP & COMMAND CENTER
# -------------------------------------------------------------
with tab1:
    col_map, col_details = st.columns([7, 3])

    with col_details:
        st.subheader("🎯 Threat Gauge & Alert")

        # Gauge Chart for Probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(max_flood_prob * 100, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Peak Inundation Probability (%)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(42, 157, 143, 0.25)'},
                    {'range': [25, 55], 'color': 'rgba(255, 183, 3, 0.25)'},
                    {'range': [55, 80], 'color': 'rgba(251, 133, 0, 0.25)'},
                    {'range': [80, 100], 'color': 'rgba(230, 57, 70, 0.25)'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 📋 Recommended Actions:")
        for act in overall_alert.actions:
            st.markdown(f"• {act}")

        st.markdown("---")
        st.markdown("#### 🗺️ Map Legend:")
        st.markdown("""
        - 🔴 **Critical Risk** (≥ 80% Prob)
        - 🟠 **High Risk** (55% - 80% Prob)
        - 🟡 **Moderate Risk** (25% - 55% Prob)
        - 🟢 **Low Risk** (< 25% Prob)
        - 🏥 **Emergency Shelters** (Elevated safe ground)
        - 🌊 **River Channel** (Primary drainage)
        """)

    with col_map:
        st.subheader("🌐 Live Geospatial Risk Intelligence")
        show_heat = st.checkbox("Show Inundation Heat Density Overlay", value=True)
        folium_map = map_builder.build_map(
            grid_df=predicted_grid_df,
            shelters=router.shelters,
            show_heatmap=show_heat
        )
        st_folium(folium_map, width="100%", height=560)


# -------------------------------------------------------------
# TAB 2: EVACUATION ROUTE & SHELTER ALLOCATION
# -------------------------------------------------------------
with tab2:
    st.subheader("🏃 Real-Time Flood-Aware Evacuation Navigation")
    st.markdown("Select your starting point or zone. The routing engine calculates the fastest, safest path to verified shelters **while dynamically bypassing flooded and impassable roadways**.")

    col_nav_controls, col_nav_map = st.columns([3, 7])

    with col_nav_controls:
        st.markdown("#### 📍 Citizen Location")
        zone_options = predicted_grid_df["zone_id"].tolist()
        selected_origin_zone = st.selectbox("Select Current Location / Zone:", zone_options, index=12)

        origin_row = predicted_grid_df[predicted_grid_df["zone_id"] == selected_origin_zone].iloc[0]
        user_lat = float(origin_row["lat"])
        user_lon = float(origin_row["lon"])
        user_risk_prob = float(origin_row["flood_probability"])
        user_risk_lvl = classifier.classify_probability(user_risk_prob)

        st.info(f"**Zone Status**: {selected_origin_zone}\n\n- Flood Probability: **{user_risk_prob*100:.1f}%**\n- Risk Tier: **{user_risk_lvl.value}**\n- Elevation: **{origin_row['elevation_m']} m**")

        # Solve Evacuation Plan
        evac_plan = router.find_best_evacuation_route(user_lat, user_lon)

        if evac_plan:
            st.success(f"✅ Optimal Shelter Allocated: **{evac_plan.shelter.name}**")
            st.metric("Total Route Distance", f"{evac_plan.total_distance_km} km")
            st.metric("Estimated Evac Time", f"{evac_plan.estimated_time_mins} mins")
            st.metric("Route Safety Clearance", f"{evac_plan.safety_score} / 100")

            if evac_plan.warnings:
                for w in evac_plan.warnings:
                    st.warning(f"⚠️ {w}")
        else:
            st.error("No safe route reachable. Move to nearest elevated roof or contact emergency helicopter dispatch.")

    with col_nav_map:
        st.subheader("🗺️ Evacuation Corridor & Shelter Guidance")
        nav_map = map_builder.build_map(
            grid_df=predicted_grid_df,
            shelters=router.shelters,
            origin_coord=(user_lat, user_lon),
            evacuation_plan=evac_plan,
            show_heatmap=False
        )
        st_folium(nav_map, width="100%", height=550)

    # Shelters Table
    st.markdown("#### 🏥 Verified Emergency Shelters Status")
    shelter_table_data = []
    for s in router.shelters:
        shelter_table_data.append({
            "Shelter Name": s.name,
            "Elevation (m)": s.elevation_m,
            "Max Capacity": s.capacity,
            "Current Occupancy": s.current_occupancy,
            "Available Slots": s.available_capacity,
            "Facilities": ", ".join(s.facilities)
        })
    st.dataframe(pd.DataFrame(shelter_table_data), use_container_width=True)


# -------------------------------------------------------------
# TAB 3: ML MODEL INTELLIGENCE & FEATURE IMPORTANCE
# -------------------------------------------------------------
with tab3:
    st.subheader(f"🤖 Machine Learning Performance & Explainability ({selected_model_type})")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Model Accuracy", f"{active_metrics['accuracy'] * 100:.2f}%")
    with col_m2:
        st.metric("Precision Score", f"{active_metrics['precision'] * 100:.2f}%")
    with col_m3:
        st.metric("Recall (Sensitivity)", f"{active_metrics['recall'] * 100:.2f}%")
    with col_m4:
        st.metric("ROC-AUC Score", f"{active_metrics['roc_auc']:.4f}")

    col_chart1, col_chart2 = st.columns([6, 4])

    with col_chart1:
        st.markdown("#### 🔬 Global Feature Importance (Hydrological & Topographical Drivers)")
        importances = active_metrics.get("feature_importances", {})
        if importances:
            df_imp = pd.DataFrame({
                "Feature": list(importances.keys())[:10],
                "Importance": list(importances.values())[:10]
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 10 Flood Risk Feature Weights"
            )
            fig_imp.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🧮 Model Confusion Matrix")
        cm = active_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["No Flood (0)", "Flood Inundation (1)"],
            y=["No Flood (0)", "Flood Inundation (1)"],
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Benchmark comparison
    st.markdown("#### ⚖️ Model Comparison Benchmark")
    comp_df = pd.DataFrame([
        {
            "Model": "XGBoost Classifier",
            "Accuracy": f"{xgb_metrics['accuracy']*100:.2f}%",
            "Precision": f"{xgb_metrics['precision']*100:.2f}%",
            "Recall": f"{xgb_metrics['recall']*100:.2f}%",
            "F1-Score": f"{xgb_metrics['f1_score']*100:.2f}%",
            "ROC-AUC": f"{xgb_metrics['roc_auc']:.4f}"
        },
        {
            "Model": "Random Forest Classifier",
            "Accuracy": f"{rf_metrics['accuracy']*100:.2f}%",
            "Precision": f"{rf_metrics['precision']*100:.2f}%",
            "Recall": f"{rf_metrics['recall']*100:.2f}%",
            "F1-Score": f"{rf_metrics['f1_score']*100:.2f}%",
            "ROC-AUC": f"{rf_metrics['roc_auc']:.4f}"
        }
    ])
    st.dataframe(comp_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: SECTOR INUNDATION & HYDROLOGIC ANALYTICS
# -------------------------------------------------------------
with tab4:
    st.subheader("📊 Spatial Inundation & Sector Analytics")

    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        fig_dist = px.histogram(
            predicted_grid_df,
            x="flood_probability",
            nbins=20,
            title="Distribution of Flood Probabilities Across Sectors",
            labels={"flood_probability": "Flood Probability (0 to 1)"},
            color_discrete_sequence=["#0F4C81"]
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_stat2:
        fig_scatter = px.scatter(
            predicted_grid_df,
            x="dist_to_river_m",
            y="elevation_m",
            color="flood_probability",
            size="soil_moisture_pct",
            hover_name="zone_id",
            color_continuous_scale="Spectral_r",
            title="Elevation vs. Distance to River (Color = Flood Prob, Size = Soil Saturation)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### 📋 Sector Inundation Status Table")
    filter_tier = st.multiselect("Filter by Risk Tier:", ["CRITICAL", "HIGH", "MODERATE", "LOW"], default=["CRITICAL", "HIGH", "MODERATE", "LOW"])

    predicted_grid_df["Risk Level"] = predicted_grid_df["flood_probability"].apply(lambda p: classifier.classify_probability(p).value)
    filtered_grid = predicted_grid_df[predicted_grid_df["Risk Level"].isin(filter_tier)]

    st.dataframe(
        filtered_grid[[
            "zone_id", "Risk Level", "flood_probability", "elevation_m",
            "dist_to_river_m", "slope_deg", "soil_moisture_pct", "historical_flood_count"
        ]],
        use_container_width=True
    )
