# 🏔️ StrataNex

## AI-Based Early Warning and Landslide Risk Monitoring System for North-East India

**Smart India Hackathon 2026**  
**Problem Statement ID:** SIH26001  
**Theme:** Disaster Management  
**Category:** Software  
**Team:** StrataNex (Next-gen subsurface analytics)

---

## 🌐 Live Demo

**StrataNex Web Application:**  
https://stratanex.streamlit.app

**GitHub Repository:**  
https://github.com/Harsh0v/StrataNex

---

## 📌 About the Project

StrataNex is an AI-assisted landslide risk monitoring and decision-support system designed for North-East India.

The system combines historical landslide information, terrain parameters, rainfall information and AI/ML-assisted risk screening to provide an easy-to-understand landslide risk dashboard.

It also includes **SafeRoute**, which analyses a road route and checks its proximity to historical GSI landslide locations.

---

## 🚀 Main Capabilities

- 📊 Interactive Streamlit monitoring dashboard
- 🗺️ GIS-based landslide risk visualisation
- 🤖 AI/ML-assisted landslide risk screening
- 🌧️ Live rainfall information
- ⛰️ Automatic elevation and slope information
- 📍 Location-based risk analysis
- 🗃️ Historical GSI landslide inventory integration
- ⚠️ Hybrid risk assessment
- 🧠 AI Risk Intelligence
- 🛡️ Smart safety recommendations
- 🛣️ OpenRouteService road routing
- 📌 SafeRoute screening against nearby historical landslide records
- 📱 Web-based interface accessible from desktop and mobile browsers

---

## 🗺️ SafeRoute

SafeRoute provides route-level landslide exposure screening.

The user enters destination coordinates and StrataNex:

1. Calculates the road route.
2. Displays the route on an interactive map.
3. Checks historical GSI landslide records near the route.
4. Screens records within approximately 500 metres of the route.
5. Displays route distance and estimated travel time.
6. Reports nearby historical landslide points.
7. Provides a route-screening risk indication.

SafeRoute is intended as a **decision-support feature** and does not declare a road officially safe or closed.

---

## 📊 Data and Risk Inputs

The current prototype uses:

- Historical GSI landslide inventory
- Latitude and longitude
- Terrain slope
- Elevation
- Rainfall information
- AI/ML model prediction
- Historical landslide proximity
- Road-route information

The project currently contains **9,663 historical GSI landslide records** for analysis and visualisation.

---

## 🧠 AI/ML Risk Screening

StrataNex uses a trained machine-learning model together with environmental and historical information to assist landslide risk screening.

The dashboard combines available indicators to generate an understandable risk output for decision-support purposes.

---

## 🛠️ Technology Stack

- **Python**
- **Streamlit**
- **Machine Learning**
- **Pandas / NumPy**
- **Folium**
- **GIS / Geospatial Processing**
- **OpenStreetMap**
- **OpenRouteService (ORS)**
- **Open-Meteo**
- **GSI Landslide Inventory**
- **Git & GitHub**
- **Streamlit Community Cloud**

---

## 📂 Important Project Files

- `app.py` — Main Streamlit application
- `gsi_ner_landslides.csv` — Historical landslide dataset
- `landslide_model.pkl` — Trained ML model
- `pilot_segments.csv` — Pilot segment data
- `requirements.txt` — Python dependencies
- `train_model.py` — Model training script
- `README.md` — Project documentation

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Harsh0v/StrataNex.git
cd StrataNex
