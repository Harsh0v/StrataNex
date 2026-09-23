# StrataNex

AI-assisted landslide risk monitoring and SafeRoute screening system for North-East India.

## Main capabilities
- Streamlit monitoring dashboard
- GSI historical landslide inventory visualisation
- AI/ML-assisted landslide risk screening
- Rainfall, elevation and slope context
- OpenRouteService road routing
- SafeRoute screening against nearby historical GSI landslide records
- Optimised map loading and route caching

## Run locally
1. Install Python 3.10+.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a `.env` file from `.env.example` and add your own OpenRouteService API key.
4. Keep the required project data/model files beside `app.py`.
5. Run:
   `streamlit run app.py`

## Required local runtime assets
The final working installation uses:
- `gsi_ner_landslides.csv`
- `landslide_model.pkl`
- `pilot_segments.csv` (when referenced by your build)

These local data/model files are intentionally not fabricated in this package. Copy the tested versions from your working StrataNex project folder into the submission folder before running on a new computer.

## Security
Never commit or submit `.env`, API keys, tokens, `.venv`, or `__pycache__`.
The ORS key previously used during development should be rotated before public deployment.

## Important limitation
StrataNex is a decision-support / early-warning project. Historical GSI proximity does not prove that a road is currently blocked or unsafe. For real travel or evacuation decisions, verify current official warnings and road-closure information from the relevant authorities.
