# Lebanon Wildfire-Risk Predictor

Predicts next-day wildfire risk for every ~5 km grid cell across Lebanon and shows it on a live map.

## Status
🚧 Work in progress, Day 1 of 18: project setup complete.

## Data sources
- NASA FIRMS: satellite fire detections
- Open-Meteo: daily weather
- NDVI: vegetation greenness
- Elevation and terrain

## Stack
Python 3.12 · pandas · geopandas · LightGBM · MLflow · Optuna · SHAP · FastAPI · Docker · Streamlit

## Setup
1. `py -3.12 -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your NASA FIRMS MAP_KEY
5. `python src\check_setup.py`