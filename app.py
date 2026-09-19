import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import math
import joblib

ml_model = joblib.load("landslide_model.pkl")

def get_ml_prediction(lat, lon):
    input_data = pd.DataFrame(
        [[lat, lon]],
        columns=["latitude", "longitude"]
    )

    prediction = ml_model.predict(input_data)[0]

    if prediction == 1:
        return "HIGH"
    else:
        return "LOW"
    
@st.cache_data
def load_gsi_data():
    return pd.read_csv("gsi_ner_landslides.csv")
def get_historical_count(lat, lon, radius=0.1):
    nearby = gsi_data[
        (gsi_data["latitude"].between(lat - radius, lat + radius)) &
        (gsi_data["longitude"].between(lon - radius, lon + radius))
    ]
    return len(nearby)
@st.cache_data(ttl=600)
def get_live_rainfall(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=rain"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data["current"]["rain"]

    except Exception as e:
        st.error(f"Rainfall API Error: {e}")
        return None
@st.cache_data(ttl=3600)
def get_elevation(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/elevation"

        response = requests.get(
            url,
            params={
                "latitude": lat,
                "longitude": lon
            },
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        return data["elevation"][0]

    except Exception as e:
        st.error(f"Elevation API Error: {e}")
        return None
@st.cache_data(ttl=3600)
def get_auto_slope(lat, lon):
    try:
        delta = 0.001

        h1 = get_elevation(lat, lon)
        h2 = get_elevation(lat + delta, lon)

        if h1 is None or h2 is None:
            return None

        horizontal_distance = 111.0
        height_difference = abs(h2 - h1)

        slope = math.degrees(
            math.atan(height_difference / horizontal_distance)
        )

        return round(slope, 2)

    except Exception as e:
        st.error(f"Slope calculation error: {e}")
        return None
st.set_page_config(
    page_title="StrataNex",
    page_icon="⛰️",
    layout="wide"
)

gsi_data = load_gsi_data() 

def calculate_risk(rainfall, slope, historical_count):
    rainfall = rainfall if rainfall is not None else 0
    slope = slope if slope is not None else 0
    historical_count = historical_count if historical_count is not None else 0

    # Current triggering conditions
    trigger_score = 0

    # Live rainfall contribution
    if rainfall >= 50:
        trigger_score += 3
    elif rainfall >= 20:
        trigger_score += 2
    elif rainfall >= 5:
        trigger_score += 1

    # Terrain slope contribution
    if slope >= 45:
        trigger_score += 3
    elif slope >= 30:
        trigger_score += 2
    elif slope >= 15:
        trigger_score += 1

    # Historical GSI data = susceptibility/background risk
    history_score = 0

    if historical_count >= 50:
        history_score = 2
    elif historical_count >= 10:
        history_score = 1
    elif historical_count >= 1:
        history_score = 0.5

    total_score = trigger_score + history_score

    # Immediate danger requires current triggering conditions
    if trigger_score >= 4:
        return "VERY HIGH"
    elif trigger_score >= 2 and total_score >= 3:
        return "HIGH"
    elif total_score >= 2:
        return "MODERATE"
    else:
        return "LOW"

st.sidebar.header("📍 Location")

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=20.0,
    max_value=30.0,
    value=26.14,
    step=0.01
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=87.0,
    max_value=98.0,
    value=91.74,
    step=0.01
)

slope = 38  # fallback only if automatic slope is unavailable

test_rain = get_live_rainfall(latitude, longitude)

auto_slope = get_auto_slope(latitude, longitude)

if auto_slope is not None:
    st.info(f"⛰️ Automatic Slope: {auto_slope}°")
else:
    st.warning("Automatic slope data unavailable.")

    
elevation = get_elevation(latitude, longitude)

if elevation is not None:
    st.info(f"🏔️ Elevation: {elevation} m")
else:
    st.warning("Elevation data unavailable.")


if test_rain is not None:
    st.success(f"🌧️ Live Rainfall: {test_rain} mm")
else:
    st.error("Live rainfall data could not be fetched.")
    test_rain = 0
historical_count = get_historical_count(latitude, longitude)

risk_level = calculate_risk(
    test_rain,
    auto_slope if auto_slope is not None else slope,
    historical_count
)

# AI/ML prediction
ml_risk = get_ml_prediction(latitude, longitude)

st.info(f"🤖 AI/ML Prediction: {ml_risk}")

# Final Hybrid Risk: Rule-based + AI/ML
risk_rank = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "VERY HIGH": 4
}

rule_score = risk_rank.get(risk_level, 1)
ml_score = risk_rank.get(ml_risk, 1)

final_score = max(rule_score, ml_score)

final_risk = {
    1: "LOW",
    2: "MODERATE",
    3: "HIGH",
    4: "VERY HIGH"
}[final_score]

st.subheader("🚨 Final Hybrid Landslide Risk")
if final_risk == "VERY HIGH":
    st.error("🔴 VERY HIGH RISK — Immediate attention required!")
elif final_risk == "HIGH":
    st.error("🟠 HIGH RISK — Strong landslide possibility.")
elif final_risk == "MODERATE":
    st.warning("🟡 MODERATE RISK — Monitor conditions carefully.")
else:
    st.success("🟢 LOW RISK — Conditions currently stable.")
st.title("⛰️ StrataNex")

st.subheader(
    "AI-Based Early Warning and Landslide Risk Monitoring System in NER"
)

st.caption("Smart India Hackathon 2026 | Disaster Management")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Risk Level", final_risk)

with col2:
    if test_rain is not None:
        st.metric("Rainfall", f"{test_rain} mm")
    else:
        st.metric("Rainfall", "Data unavailable")

with col3:
    st.metric("Slope", f"{auto_slope if auto_slope is not None else slope}°")

with col4:
    alert_count = {
        "LOW": 0,
        "MODERATE": 1,
        "HIGH": 2,
        "VERY HIGH": 3
    }.get(risk_level, 0)

    st.metric("Active Alerts", alert_count)

st.divider()

st.header("🚨 Early Warning")

if final_risk == "VERY HIGH":
    st.error("🔴 VERY HIGH RISK: Immediate landslide warning. Avoid vulnerable areas.")

elif final_risk == "HIGH":
    st.error("🟠 HIGH RISK: Landslide conditions are dangerous. Stay alert.")

elif final_risk == "MODERATE":
    st.warning("🟡 MODERATE RISK: Conditions require monitoring.")

else:
    st.success("🟢 LOW RISK: No significant landslide risk detected.")

st.header("🛰️ Monitoring System")

st.info(
    "StrataNex analyses rainfall, terrain, historical landslides "
    "and environmental parameters to estimate landslide risk."
)

st.header("System Modules")

c1, c2 = st.columns(2)

with c1:
    st.write("🌧️ Real-Time Rainfall Monitoring")
    st.write("🗺️ GIS Landslide Risk Map")
    st.write("🤖 AI/ML Risk Prediction")
    st.write("🚨 Early Warning System")

with c2:
    st.write("🛣️ Road Connectivity Monitoring")
    st.write("📱 Field Reporting")
    st.write("🧭 Safe Route Recommendation")
    st.write("📡 Low-Network Support")

    st.divider()
st.subheader("🔎 Filter Landslide Data")

states = ["All"] + sorted(gsi_data["state"].dropna().unique().tolist())

selected_state = st.selectbox(
    "Select North-East State",
    states
)

if selected_state == "All":
    filtered_gsi_data = gsi_data
else:
    filtered_gsi_data = gsi_data[gsi_data["state"] == selected_state]

st.caption(
    f"Showing {len(filtered_gsi_data):,} historical landslide records"
)
st.header("🗺️ North-East India Landslide Risk Map")

if not filtered_gsi_data.empty:
    map_lat = filtered_gsi_data["latitude"].mean()
    map_lon = filtered_gsi_data["longitude"].mean()

    if selected_state == "All":
        map_zoom = 6
    else:
        map_zoom = 8
else:
    map_lat = 26.2
    map_lon = 92.9
    map_zoom = 6
m = folium.Map(
    location=[map_lat, map_lon],
    zoom_start=map_zoom,
    tiles="OpenStreetMap"
)

marker_cluster = MarkerCluster(
    name="GSI Historical Landslides"
).add_to(m)

# Real historical landslide locations extracted from GSI inventory
gsi_map_data = filtered_gsi_data.dropna(
    subset=["latitude", "longitude"]
)

for _, row in gsi_map_data.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        popup=f"""
        <b>Historical Landslide</b><br>
        State: {row['state']}<br>
        Source: GSI Landslide Inventory
        """,
        tooltip=f"{row['state']} - Historical Landslide",
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.65
    ).add_to(marker_cluster)

# Selected monitoring location
folium.Marker(
    location=[latitude, longitude],
    popup=f"""
    <b>StrataNex Monitoring Location</b><br>
    Latitude: {latitude}<br>
    Longitude: {longitude}<br>
    Risk Level: {risk_level}<br>
    Rainfall: {test_rain} mm
    """,
    tooltip=f"Current Risk: {risk_level}",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(m)

# Risk zone colour based on current risk level
risk_colors = {
    "LOW": "green",
    "MODERATE": "orange",
    "HIGH": "red",
    "VERY HIGH": "darkred"
}

risk_color = risk_colors.get(risk_level, "blue")


risk_radius = {
    "LOW": 3000,
    "MODERATE": 5000,
    "HIGH": 8000,
    "VERY HIGH": 12000
}.get(risk_level, 5000)


folium.Circle(
    location=[latitude, longitude],
    radius=risk_radius,
    color=risk_color,
    fill=True,
    fill_color=risk_color,
    fill_opacity=0.55, weight=4,
    popup=f"{risk_level} Risk Zone",
    tooltip=f"{risk_level} Risk Zone"
).add_to(m)
st_folium(
    m,
    height=550,
    use_container_width=True
)
# Current Landslide Risk Status
st.subheader("⚠️ Current Landslide Risk Status")

col1, col2, col3 = st.columns(3)

col1.metric("Risk Level", final_risk)
col2.metric("Rainfall", f"{test_rain} mm")
col3.metric("Risk Zone Radius", f"{risk_radius/1000:.0f} km")

if final_risk == "LOW":
    st.success("🟢 LOW RISK - Conditions are currently stable.")
elif final_risk == "MODERATE":
    st.warning("🟡 MODERATE RISK - Stay alert and monitor conditions.")
elif final_risk == "HIGH":
    st.error("🟠 HIGH RISK - Avoid vulnerable slopes and stay prepared.")
else:
    st.error("🔴 VERY HIGH RISK - Immediate precaution recommended.")
st.caption(
    "Prototype demonstration only. Risk markers are sample data, "
    "not real-time landslide predictions."
)
