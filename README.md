# Lebanon Wildfire-Risk Predictor

Predicts next-day wildfire risk for every ~5 km grid cell across Lebanon and shows it on a live map.

## Project Status

🚧 **In progress — Day 3 of 18 complete**

| Day | Task | Status |
|----|------|--------|
| 1 | Setup (repo, venv, structure) | ✅ Done |
| 2 | Fire data (NASA FIRMS) | ✅ Done |
| 3 | Lebanon grid (~5 km cells) | ✅ Done |
| 4 | Weather data (Open-Meteo) | ⏳ Next |
| 5 | Vegetation (NDVI) + terrain | ⬜ |
| 6 | Exploratory data analysis | ⬜ |
| 7 | Feature engineering | ⬜ |
| 8 | Baseline model + MLflow | ⬜ |
| 9 | LightGBM model | ⬜ |
| 10 | Spatial/time validation | ⬜ |
| 11 | Optuna tuning | ⬜ |
| 12 | SHAP explainability | ⬜ |
| 13 | FastAPI service | ⬜ |
| 14 | Docker | ⬜ |
| 15 | Daily pipeline | ⬜ |
| 16 | Streamlit live map | ⬜ |
| 17 | Deploy | ⬜ |
| 18 | README + showcase | ⬜ |


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

## Data

### Fire detections — NASA FIRMS
- Satellite: VIIRS S-NPP (375 m resolution)
- Archive (`VIIRS_SNPP_SP`) from Jan 2012, near-real-time (`VIIRS_SNPP_NRT`) for recent months
- Area: Lebanon bounding box (35.10–36.63°E, 33.05–34.70°N)
- Cleaning: kept vegetation fires only, removed low-confidence and duplicate detections
- Result: 13,447 detections on 3,270 fire days (2012–2026)
- Fire season peaks June–October

Scripts:
- `src/fetch_firms.py` — downloads FIRMS data (needs `FIRMS_MAP_KEY` in `.env`)
- `src/clean_firms.py` — filters and saves to `data/raw/firms_lebanon.parquet`
- `src/check_firms.py` — prints summary checks

## What's Built So Far

### Fire data (Day 2)
- Downloads NASA FIRMS VIIRS detections (375 m) over Lebanon, 2012 → today
- Keeps vegetation fires only, drops low-confidence detections and duplicates

### Lebanon grid (Day 3)
- Splits Lebanon into **488 cells of 5 × 5 km**, built in a meter-based projection (UTM 36N)
- Each cell stores `land_frac`, the share of the cell inside Lebanon (coastal and border cells are partial)
- Each cell gets one lat/lon point placed on land, used to fetch its weather
- **9,527 fire detections** matched to their cells; 3,920 outside Lebanon removed
- **Persistent hotspot filter:** spots detected in ≥ 6 years *and* ≥ 8 calendar months are flagged as likely industrial heat sources (e.g. coastal power plants and cement works), not wildfires. That's 959 detections, kept but flagged.
- Fire is rare: about **0.5% of cell-days** have a fire, so later models are judged with PR-AUC and recall, not accuracy

| Script | What it does |
|--------|--------------|
| `src/fetch_firms.py` | Downloads raw FIRMS fire detections |
| `src/clean_firms.py` | Cleans detections and saves them as Parquet |
| `src/download_boundary.py` | Downloads Lebanon's border (geoBoundaries) |
| `src/grid.py` | Builds the 5 km grid, land fractions and cell points |
| `src/join_fires.py` | Assigns fires to cells and flags persistent hotspots |