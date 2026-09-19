# 🏔️ StrataNex

## AI-Based Early Warning and Landslide Risk Monitoring System in NER

StrataNex is a prototype landslide risk monitoring and early-warning system designed for the North-Eastern Region (NER) of India. It combines historical landslide inventory, live rainfall information, terrain parameters and a prototype machine-learning model to support location-based landslide risk assessment.

## 🌐 Live Prototype

👉 https://stratanex.streamlit.app/

## 🚀 Key Features

- Live rainfall data integration
- Automatic elevation estimation
- Automatic terrain slope estimation
- Historical landslide analysis
- GSI-based historical landslide inventory integration
- AI/ML-based prototype risk prediction
- Hybrid landslide risk assessment
- LOW, MODERATE, HIGH and VERY HIGH risk levels
- Early-warning alerts
- Interactive North-East India landslide risk map
- Historical landslide marker clustering
- Location-based monitoring
- Dynamic risk-zone visualisation
- State-wise historical landslide filtering

## 🤖 Hybrid Risk Assessment

StrataNex combines:

- Live rainfall conditions
- Terrain slope
- Historical landslide occurrence
- Location-based prototype ML prediction

The outputs are combined to generate a final hybrid landslide risk level.

> **Note:** The machine-learning component is a prototype model for demonstration and research purposes. Its validation results should not be interpreted as operational real-world landslide forecasting accuracy.

## 🛠️ Technology Stack

- Python
- Streamlit
- Scikit-learn
- Random Forest
- Pandas
- Folium
- Streamlit-Folium
- Open-Meteo API
- Open-Elevation API
- GitHub
- Streamlit Community Cloud

## 📊 Data Sources

- Geological Survey of India (GSI) landslide inventory
- Open-Meteo for live rainfall/weather data
- Open-Elevation for elevation data

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
