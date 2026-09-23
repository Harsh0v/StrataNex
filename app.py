import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from streamlit_folium import st_folium, folium_static
import pandas as pd
import math
import joblib
from dotenv import load_dotenv
import os
load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")
import openrouteservice

@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("landslide_model.pkl")
    except Exception:
        return None

ml_model = load_ml_model()

def get_ml_prediction(lat, lon):
    input_data = pd.DataFrame(
        [[lat, lon]],
        columns=["latitude", "longitude"]
    )

    if ml_model is None:
        return "UNAVAILABLE"

    try:
        prediction = ml_model.predict(input_data)[0]
        return "HIGH" if prediction == 1 else "LOW"
    except Exception:
        return "UNAVAILABLE"
    
@st.cache_data
def load_gsi_data():
    try:
        df = pd.read_csv("gsi_ner_landslides.csv")
        required = {"state", "latitude", "longitude", "source", "raw_record"}
        if not required.issubset(df.columns):
            return pd.DataFrame(columns=list(required))
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        return df.dropna(subset=["latitude", "longitude"]).copy()
    except Exception:
        return pd.DataFrame(columns=["state", "latitude", "longitude", "source", "raw_record"])
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

        response = requests.get(url, timeout=30)
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

        # delta=0.001 degrees latitude is about 111 metres. Elevation is metres.
        horizontal_distance_m = 111.0
        height_difference_m = abs(h2 - h1)

        slope = math.degrees(
            math.atan(height_difference_m / horizontal_distance_m)
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

# =========================================================
# STRATANEX PROFESSIONAL DARK UI
# =========================================================
# =========================================================
# STRATANEX HERO HEADER
# =========================================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1f33 55%, #102b3f 100%);
    color: #f8fafc;
}

/* Hero */
.stratanex-hero {
    background: linear-gradient(135deg, #0b1728 0%, #102d46 55%, #123b4d 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 22px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.30);
}

.hero-label {
    color: #38bdf8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-title span {
    color: #38bdf8;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 20px;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.hero-badges span {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 700;
    color: #e2e8f0;
}

.hero-footer {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid rgba(148,163,184,0.20);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #081525;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid #1e3a5f;
    padding: 15px;
    border-radius: 14px;
}



/* ==========================================================
   STRATANEX FINAL SUBMISSION UI
   Visual-only layer. Application/routing/ML/GSI logic unchanged.
   ========================================================== */

/* Remove Streamlit's top white/header strip and excess top spacing */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu {
    display: none !important;
}
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
      radial-gradient(circle at 78% 18%, rgba(3,105,161,.18), transparent 30%),
      linear-gradient(135deg, #06111f 0%, #071a2d 48%, #082744 100%) !important;
}
.stApp {
    color: #f8fafc !important;
}

/* Main content typography */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}
p, li, label, [data-testid="stCaptionContainer"] {
    color: #e2e8f0 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    opacity: 1 !important;
}

/* Sidebar becomes a proper dark navigation/location rail */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07182a 0%, #08233c 100%) !important;
    border-right: 1px solid rgba(56,189,248,.18) !important;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* Inputs */
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    color: #7dd3fc !important;
    font-weight: 700 !important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
}
div[data-baseweb="input"] input {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 650 !important;
}
[data-testid="stNumberInput"] button {
    color: #0f172a !important;
}

/* Buttons: visible, blue, presentation-ready */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    min-height: 48px !important;
    border-radius: 10px !important;
    border: 1px solid #38bdf8 !important;
    background: linear-gradient(135deg, #1687ff 0%, #006ee6 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 24px rgba(0,110,230,.28) !important;
}
.stButton > button *,
[data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #2997ff 0%, #0878ef 100%) !important;
    border-color: #7dd3fc !important;
    transform: translateY(-1px);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(8,35,60,.96), rgba(7,27,47,.96)) !important;
    border: 1px solid rgba(14,165,233,.72) !important;
    border-radius: 14px !important;
    padding: 1rem 1.05rem !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.20) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    color: #f8fafc !important;
    opacity: 1 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 650 !important;
}
[data-testid="stMetricValue"] {
    font-weight: 850 !important;
    font-size: clamp(1.45rem, 2.15vw, 2.15rem) !important;
    line-height: 1.12 !important;
}

/* Keep metric values readable instead of cutting them with ... */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
    overflow-wrap: anywhere !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(125,211,252,.40) !important;
    box-shadow: 0 8px 22px rgba(0,0,0,.14) !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(7,31,54,.72) !important;
    border: 1px solid rgba(56,189,248,.28) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Maps/components sit in clean cards */
iframe {
    border-radius: 12px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(56,189,248,.25) !important;
}

/* Hero/card readability */
.hero-card {
    border: 1px solid rgba(56,189,248,.34) !important;
    box-shadow: 0 18px 45px rgba(0,0,0,.22) !important;
}
.hero-subtitle, .hero-footer {
    color: #dbeafe !important;
}
.hero-badges span {
    color: #f8fafc !important;
}

/* Tables/data */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Links */
.stApp a {
    color: #7dd3fc !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .block-container {
        padding-left: .8rem !important;
        padding-right: .8rem !important;
        padding-top: .6rem !important;
    }
    h1 { font-size: 1.85rem !important; }
    h2 { font-size: 1.45rem !important; }
    [data-testid="stMetric"] {
        padding: .75rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

risk_color = {
    "LOW": "#35e58b",
    "MODERATE": "#ffd54a",
    "HIGH": "#ff8a3d",
    "VERY HIGH": "#ff4b4b"
}.get("LOW", "#35e58b")

hero_html = f"""
<div class="stratanex-hero">
<div class="hero-label">AI-POWERED LANDSLIDE INTELLIGENCE</div>
<div class="hero-title">🏔️ Strata<span>Nex</span></div>
<div class="hero-subtitle">AI-Based Early Warning &amp; Landslide Risk Monitoring System<br>for North-East India</div>
<div class="hero-badges">
<span>● LIVE MONITORING</span>
<span>🤖 AI/ML ENABLED</span>
<span>🛰️ GSI DATA</span>
<span>⚠️ HYBRID RISK ENGINE</span>
</div>
<div class="hero-footer">Smart India Hackathon 2026 • Disaster Management</div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

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


rainfall_available = test_rain is not None
if rainfall_available:
    st.success(f"🌧️ Live Rainfall: {test_rain} mm")
else:
    st.warning("Live rainfall data unavailable. Risk confidence is reduced; rainfall is not treated as a confirmed zero.")
    test_rain = 0
historical_count = get_historical_count(latitude, longitude)

risk_level = calculate_risk(
    test_rain,
    auto_slope if auto_slope is not None else slope,
    historical_count
)

ml_risk = get_ml_prediction(latitude, longitude)
st.info(f"🤖 AI/ML Model Prediction: {ml_risk}")

risk_priority = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "VERY HIGH": 4
}

if ml_risk in risk_priority and risk_priority[ml_risk] > risk_priority.get(risk_level, 1):
    final_risk = ml_risk
else:
    final_risk = risk_level

data_sources_ok = sum([rainfall_available, auto_slope is not None, elevation is not None, not gsi_data.empty, ml_model is not None])
data_confidence = "HIGH" if data_sources_ok >= 5 else ("MODERATE" if data_sources_ok >= 3 else "LOW")

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

# Fast client-side cluster: avoids constructing thousands of Python Folium
# marker objects on every Streamlit rerun. This makes SafeRoute clicks much faster.
gsi_map_data = filtered_gsi_data.dropna(subset=["latitude", "longitude"])
# Performance mode: the all-NER overview does not need to send every one of the
# ~10k inventory points to the browser. A deterministic representative sample
# keeps the map responsive; selecting a state still shows all records for it.
if selected_state == "All" and len(gsi_map_data) > 2500:
    display_gsi_data = gsi_map_data.sample(n=2500, random_state=42)
    st.caption("⚡ Fast map mode: displaying 2,500 representative inventory points. Select a state to view all its records.")
else:
    display_gsi_data = gsi_map_data
fast_points = display_gsi_data[["latitude", "longitude"]].astype(float).values.tolist()
if fast_points:
    FastMarkerCluster(
        data=fast_points,
        name="GSI Historical Landslides"
    ).add_to(m)

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
# Display-only map: folium_static avoids the event bridge used by st_folium,
# which reduces reruns and noticeably improves initial dashboard loading.
folium_static(m, height=550, width=1000)
# Current Landslide Risk Status
st.subheader("⚠️ Current Landslide Risk Status")

# =========================================================
# PREMIUM LIVE METRICS
# =========================================================

risk_emoji = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🟠",
    "VERY HIGH": "🔴"
}.get(final_risk, "⚪")

display_slope = auto_slope if auto_slope is not None else slope

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "⚠️ HYBRID RISK",
        f"{risk_emoji} {final_risk}"
    )

with m2:
    st.metric(
        "🌧️ LIVE RAINFALL",
        f"{test_rain:.1f} mm"
    )

with m3:
    st.metric(
        "⛰️ TERRAIN SLOPE",
        f"{display_slope:.2f}°"
    )

with m4:
    st.metric(
        "📍 RISK ZONE",
        f"{risk_radius / 1000:.0f} km"
    )

st.caption(f"Data confidence: **{data_confidence}** • Available sources: {data_sources_ok}/5")
    
if final_risk == "LOW":
    st.success("🟢 LOW RISK - Conditions are currently stable.")
elif final_risk == "MODERATE":
    st.warning("🟡 MODERATE RISK - Stay alert and monitor conditions.")
elif final_risk == "HIGH":
    st.error("🟠 HIGH RISK - Avoid vulnerable slopes and stay prepared.")
else:
    st.error("🔴 VERY HIGH RISK - Immediate precaution recommended.")
st.caption(
    "Decision-support screening: live weather/elevation services and GSI historical inventory are combined with the local model. "
    "This is not an official landslide warning, road-closure feed, or evacuation order."
)

# =========================================================
# AI RISK ANALYTICS
# =========================================================

st.markdown("---")
st.subheader("🧠 AI Risk Intelligence")

a1, a2, a3 = st.columns(3)

with a1:
    st.metric("🤖 ML Prediction", ml_risk)

with a2:
    st.metric("📚 Historical Events", historical_count)

with a3:
    st.metric("🏔️ Elevation", f"{elevation:.0f} m" if elevation is not None else "N/A")

st.subheader("📊 Risk Analysis")

st.write(
    f"""
    **Current Hybrid Risk:** {final_risk}

    • AI/ML Prediction: **{ml_risk}**  
    • Live Rainfall: **{test_rain:.1f} mm**  
    • Terrain Slope: **{display_slope:.2f}°**  
    • Historical Landslides Nearby: **{historical_count}**  
    • Elevation: **{f'{elevation:.0f} m' if elevation is not None else 'Unavailable'}**
    """
)

if final_risk == "VERY HIGH":
    st.error("🚨 Immediate action recommended. Avoid vulnerable slopes and use safer routes.")
elif final_risk == "HIGH":
    st.warning("⚠️ High landslide susceptibility. Avoid steep slopes and monitor alerts.")
elif final_risk == "MODERATE":
    st.warning("🟡 Conditions require continuous monitoring.")
else:
    st.success("🟢 Current conditions indicate comparatively low landslide risk.")

st.subheader("🛡️ Smart Safety Recommendation")

if final_risk == "VERY HIGH":
    st.error("Evacuate vulnerable slopes • Avoid landslide-prone roads • Follow official emergency instructions")
elif final_risk == "HIGH":
    st.warning("Avoid steep slopes • Limit unnecessary travel • Keep emergency supplies ready")
elif final_risk == "MODERATE":
    st.info("Monitor rainfall • Stay alert for warnings • Avoid unstable slopes during heavy rain")
else:
    st.success("Conditions are stable • Continue normal monitoring")   

# =========================================================
# SAFE ROUTE RECOMMENDATION
# =========================================================

st.markdown("---")
st.subheader("🛣️ Safe Route Recommendation")

if final_risk in ["HIGH", "VERY HIGH"]:
    st.error(
        "⚠️ Current location is in a high-risk condition. "
        "Avoid nearby vulnerable slopes and use an alternative route."
    )

elif final_risk == "MODERATE":
    st.warning(
        "🟡 Moderate landslide risk detected. "
        "Travel carefully and avoid steep or historically affected zones."
    )

else:
    st.success(
        "🟢 Route conditions are currently comparatively safe. "
        "Continue monitoring live landslide alerts."
    )

st.info(
    "🛰️ StrataNex SafeRoute uses rainfall, terrain slope, "
    "historical landslide records and AI risk level to recommend safer travel."
)

# =========================================================
# SAFEROUTE MAP
# =========================================================

st.subheader("🗺️ StrataNex SafeRoute Map")

route_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=13,
    tiles="OpenStreetMap"
)

# Current location
folium.Marker(
    [latitude, longitude],
    popup="Current Location",
    tooltip="📍 Current Location",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(route_map)

# Risk zone around current location
route_color = {
    "LOW": "green",
    "MODERATE": "orange",
    "HIGH": "red",
    "VERY HIGH": "darkred"
}.get(final_risk, "blue")

folium.Circle(
    location=[latitude, longitude],
    radius=risk_radius,
    color=route_color,
    fill=True,
    fill_color=route_color,
    fill_opacity=0.20,
    tooltip=f"{final_risk} Risk Zone"
).add_to(route_map)

# This overview map is display-only; static rendering is faster and prevents
# map pan/zoom events from triggering unnecessary Streamlit reruns.
folium_static(route_map, height=450, width=1000)

# =========================================================
# SAFEROUTE NAVIGATION
# =========================================================
# =========================================================
# SAFEROUTE NAVIGATION + GSI ROUTE-RISK ANALYSIS
# =========================================================

st.markdown("---")
st.subheader("🛣️ SafeRoute Navigation")
st.write("Enter destination coordinates. StrataNex will calculate the road route and check it against GSI historical landslide locations.")

destination_lat = st.number_input("Destination Latitude", value=float(latitude), format="%.5f", key="destination_lat")
destination_lon = st.number_input("Destination Longitude", value=float(longitude), format="%.5f", key="destination_lon")

analyse_route = st.button("🛡️ Analyse Safe Route", use_container_width=True)

if "safe_route_result" not in st.session_state:
    st.session_state.safe_route_result = None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def analyse_gsi_route(route_points, corridor_km=0.50):
    """Find historical GSI landslides close to a route corridor.

    Uses a bounding-box prefilter and sampled route vertices to keep the
    calculation responsive even when the inventory contains many records.
    """
    if not route_points or gsi_data.empty:
        return [], 0.0

    # Sample at most ~250 route vertices for responsive distance checks.
    step = max(1, len(route_points) // 250)
    sampled = route_points[::step]
    if sampled[-1] != route_points[-1]:
        sampled.append(route_points[-1])

    lats = [p[0] for p in sampled]
    lons = [p[1] for p in sampled]
    pad_lat = corridor_km / 111.0
    mean_lat = sum(lats) / len(lats)
    pad_lon = corridor_km / max(20.0, 111.0 * abs(math.cos(math.radians(mean_lat))))

    candidates = gsi_data[
        gsi_data["latitude"].between(min(lats) - pad_lat, max(lats) + pad_lat)
        & gsi_data["longitude"].between(min(lons) - pad_lon, max(lons) + pad_lon)
    ]

    nearby = []
    closest = None
    for row in candidates.itertuples(index=False):
        lat = float(row.latitude)
        lon = float(row.longitude)
        min_d = min(haversine_km(lat, lon, rp[0], rp[1]) for rp in sampled)
        if closest is None or min_d < closest:
            closest = min_d
        if min_d <= corridor_km:
            nearby.append({
                "latitude": lat,
                "longitude": lon,
                "distance_km": min_d,
                "state": getattr(row, "state", ""),
                "source": getattr(row, "source", "GSI Landslide Inventory"),
                "raw_record": str(getattr(row, "raw_record", "")),
            })

    nearby.sort(key=lambda x: x["distance_km"])
    return nearby, (closest if closest is not None else float("inf"))


def route_risk_level(nearby_count, closest_km, distance_km, current_risk):
    """Transparent screening score; not a substitute for an official warning."""
    score = 0
    if nearby_count >= 10:
        score += 3
    elif nearby_count >= 4:
        score += 2
    elif nearby_count >= 1:
        score += 1

    if closest_km <= 0.10:
        score += 3
    elif closest_km <= 0.25:
        score += 2
    elif closest_km <= 0.50:
        score += 1

    # Add the live/current hybrid condition at the start point.
    score += {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY HIGH": 3}.get(current_risk, 0)

    # Longer exposed routes get a small extra screening penalty.
    if distance_km >= 50:
        score += 1

    if score >= 6:
        return "VERY HIGH", score
    if score >= 4:
        return "HIGH", score
    if score >= 2:
        return "MODERATE", score
    return "LOW", score


@st.cache_data(ttl=900, show_spinner=False)
def get_ors_route_cached(start_lat, start_lon, end_lat, end_lon):
    """Fetch a road route once and reuse it for identical coordinates for 15 min."""
    client = openrouteservice.Client(key=ORS_API_KEY, timeout=12)
    coordinates = [
        [round(float(start_lon), 5), round(float(start_lat), 5)],
        [round(float(end_lon), 5), round(float(end_lat), 5)],
    ]
    return client.directions(
        coordinates=coordinates,
        profile="driving-car",
        format="geojson",
    )


if analyse_route:
    if not ORS_API_KEY:
        st.error("OpenRouteService API key not found. Check ORS_API_KEY in your local .env file.")
        st.session_state.safe_route_result = None
    elif abs(destination_lat - latitude) < 1e-7 and abs(destination_lon - longitude) < 1e-7:
        st.warning("Destination is the same as the start location. Enter a different destination.")
        st.session_state.safe_route_result = None
    else:
        try:
            with st.spinner("Calculating road route..."):
                route = get_ors_route_cached(
                    float(latitude), float(longitude),
                    float(destination_lat), float(destination_lon)
                )
            feature = route["features"][0]
            route_coordinates = feature["geometry"]["coordinates"]
            route_points = [[coord[1], coord[0]] for coord in route_coordinates]
            summary = feature["properties"]["summary"]
            distance_km = float(summary["distance"]) / 1000.0
            duration_min = float(summary["duration"]) / 60.0

            nearby_landslides, closest_km = analyse_gsi_route(route_points, corridor_km=0.50)
            route_risk, risk_score = route_risk_level(
                len(nearby_landslides), closest_km, distance_km, final_risk
            )

            st.session_state.safe_route_result = {
                "start": [float(latitude), float(longitude)],
                "destination": [float(destination_lat), float(destination_lon)],
                "route_points": route_points,
                "distance_km": distance_km,
                "duration_min": duration_min,
                "nearby_landslides": nearby_landslides,
                "closest_landslide_km": closest_km,
                "route_risk": route_risk,
                "risk_score": risk_score,
                "corridor_km": 0.50,
            }
        except Exception as e:
            st.session_state.safe_route_result = None
            st.error(f"Route calculation failed: {e}")

# Render from session state so Folium reruns do not erase the result.
result = st.session_state.safe_route_result
if result is not None:
    st.subheader("🛣️ Analysed SafeRoute")

    route_risk = result.get("route_risk", "LOW")
    map_route_color = {
        "LOW": "green", "MODERATE": "orange", "HIGH": "red", "VERY HIGH": "darkred"
    }.get(route_risk, "blue")

    navigation_map = folium.Map(location=result["start"], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(result["start"], tooltip="Start Location", popup="Current Location",
                  icon=folium.Icon(color="blue")).add_to(navigation_map)
    folium.Marker(result["destination"], tooltip="Destination", popup="Destination",
                  icon=folium.Icon(color="green")).add_to(navigation_map)
    folium.PolyLine(result["route_points"], color=map_route_color, weight=7, opacity=0.9,
                    tooltip=f"Route risk: {route_risk}").add_to(navigation_map)

    # Show the closest historical GSI points. Limit markers to keep the map fast.
    for item in result.get("nearby_landslides", [])[:100]:
        record = item["raw_record"].replace("<", "&lt;").replace(">", "&gt;")
        if len(record) > 180:
            record = record[:177] + "..."
        popup = (
            f"<b>GSI historical landslide</b><br>"
            f"Distance from route: {item['distance_km']:.2f} km<br>"
            f"State: {item['state']}<br>Source: {item['source']}<br>{record}"
        )
        folium.CircleMarker(
            [item["latitude"], item["longitude"]], radius=5, color="red",
            fill=True, fill_opacity=0.85, tooltip="Historical landslide near route",
            popup=folium.Popup(popup, max_width=320)
        ).add_to(navigation_map)

    if result["route_points"]:
        lats = [p[0] for p in result["route_points"]]
        lons = [p[1] for p in result["route_points"]]
        navigation_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    st_folium(navigation_map, height=500, use_container_width=True, key="real_navigation_route")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route Distance", f"{result['distance_km']:.2f} km")
    c2.metric("Estimated Time", f"{result['duration_min']:.0f} min")
    c3.metric("GSI Points ≤500 m", str(len(result.get("nearby_landslides", []))))
    closest = result.get("closest_landslide_km", float("inf"))
    c4.metric("Closest GSI Record", "None nearby" if math.isinf(closest) else f"{closest:.2f} km")

    if route_risk == "LOW":
        st.success("🟢 Route screening: LOW historical-landslide exposure in the checked 500 m corridor.")
    elif route_risk == "MODERATE":
        st.warning("🟡 Route screening: MODERATE exposure. Use extra caution and check current official warnings.")
    else:
        st.error(f"🔴 Route screening: {route_risk}. Historical landslide records and/or current conditions indicate elevated exposure.")

    st.caption(
        "Risk screening combines the current StrataNex start-point risk with proximity to GSI historical landslide records. "
        "Historical proximity does not prove that a road is currently unsafe, and this is not an official evacuation or road-closure advisory."
    )

    if result.get("nearby_landslides"):
        with st.expander("📍 View nearest GSI landslide records"):
            table_rows = []
            for item in result["nearby_landslides"][:25]:
                table_rows.append({
                    "Distance from route (km)": round(item["distance_km"], 3),
                    "Latitude": item["latitude"],
                    "Longitude": item["longitude"],
                    "State": item["state"],
                    "Source": item["source"],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.info("Before travel, verify official warnings and road closures. StrataNex is a decision-support screening tool and does not replace GSI/NDMA/SDMA/district-authority instructions.")


# ---------- Final submission usability layer ----------
st.markdown("---")
st.subheader("🛡️ StrataNex Operational Summary")

_op1, _op2, _op3 = st.columns(3)
with _op1:
    st.metric("System Mode", "Decision Support")
with _op2:
    st.metric("Historical Layer", "GSI Inventory")
with _op3:
    st.metric("Routing Layer", "OpenStreetMap / ORS")

with st.expander("ℹ️ How to interpret StrataNex results"):
    st.markdown("""
**Risk prediction** combines the data available to the application for screening purposes.

**SafeRoute screening** checks the calculated road route against nearby historical GSI landslide records. A nearby historical record indicates exposure that deserves attention; it does **not** prove that the road is currently blocked or unsafe.

**Before travel or evacuation**, verify current warnings and road-closure information from GSI, NDMA, SDMA and the relevant district/local authorities.

**Data limitations:** internet-dependent weather/routing services can temporarily be unavailable, and historical inventories do not represent every possible future landslide.
""")

st.caption(
    "StrataNex • AI-assisted landslide risk monitoring and route-screening system for North-East India"
)

