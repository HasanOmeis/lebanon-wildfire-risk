# Lebanon Wildfire-Risk Predictor

Predicts next-day wildfire risk for every ~5 km grid cell across Lebanon and shows it on a live map.

## Project Status

🚧 **In progress — Day 4 of 18 (weather download running)**

| Day | Task | Status |
|----|------|--------|
| 1 | Setup (repo, venv, structure) | ✅ Done |
| 2 | Fire data (NASA FIRMS) | ✅ Done |
| 3 | Lebanon grid (~5 km cells) | ✅ Done |
| 4 | Weather data (Open-Meteo) | ⏳ In progress |
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

## Data Sources
- **NASA FIRMS**: satellite fire detections (VIIRS, 375 m)
- **geoBoundaries**: Lebanon's country border
- **Open-Meteo**: daily historical weather
- **NDVI**: vegetation greenness *(Day 5)*
- **Elevation and terrain** *(Day 5)*

## Stack
Python 3.12 · pandas · geopandas · LightGBM · MLflow · Optuna · SHAP · FastAPI · Docker · Streamlit

## Setup
1. `py -3.12 -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your NASA FIRMS MAP_KEY
5. `python src\check_setup.py`

## What's Built So Far

### Fire data (Day 2)
- Satellite: VIIRS S-NPP (375 m resolution)
- Archive (`VIIRS_SNPP_SP`) from Jan 2012, near-real-time (`VIIRS_SNPP_NRT`) for recent months
- Area: Lebanon bounding box (35.10–36.63°E, 33.05–34.70°N)
- Cleaning: kept vegetation fires only, removed low-confidence and duplicate detections
- Result: **13,447 detections** on 3,270 fire days (2012–2026)
- Fire season peaks June–October

### Lebanon grid (Day 3)
- Splits Lebanon into **488 cells of 5 × 5 km**, built in a meter-based projection (UTM 36N)
- Each cell stores `land_frac`, the share of the cell inside Lebanon (coastal and border cells are partial)
- Each cell gets one lat/lon point placed on land, used to fetch its weather
- **9,527 fire detections** matched to their cells; 3,920 outside Lebanon removed (mostly Syrian farmland inside the bounding box)
- **Persistent hotspot filter:** spots detected in ≥ 6 years *and* ≥ 8 calendar months are flagged as likely industrial heat sources (e.g. coastal power plants and cement works), not wildfires. That's 959 detections, kept but flagged.
- Fire is rare: about **0.5% of cell-days** have a fire, so later models are judged with PR-AUC and recall, not accuracy

### Weather data (Day 4)
- Source: Open-Meteo Historical Weather API, daily values from **2012-01-25 to 2026-10-29** (same range as the fire data)
- 7 fire-weather variables: max temperature, min relative humidity, precipitation, max wind speed, max wind gusts, FAO evapotranspiration, max vapour pressure deficit
- **Key decision: a 0.1° weather grid instead of 488 cells.** Open-Meteo's historical model has a grid point about every 10 km, so neighbouring 5 km cells get the same weather anyway. Each cell is snapped to its nearest 0.1° point, giving **126 weather points** (~3.9 cells each). The data quality is the same, and the API cost drops from ~131k to ~34k calls.
- **Resumable download:** 90 batches (15 years × 6 groups of 21 points), each saved as soon as it arrives. Every run skips finished batches, stops below the 10,000 calls/day free limit, paces requests under the hourly limit, and retries failures. The full download takes ~4 daily runs.
- Output: `data/processed/weather_daily.parquet`, one row per cell per day (~2.6M rows when complete)
- Sanity check: hot, dry summers (Jul–Aug max ≈ 30°C, near-zero rain Jun–Sep), wet winters, vapour pressure deficit peaking in the fire season, and much colder winters at high inland cells than on the coast

## Scripts

| Script | What it does |
|--------|--------------|
| `src/check_setup.py` | Checks that the environment and packages work |
| `src/fetch_firms.py` | Downloads raw FIRMS fire detections (needs `FIRMS_MAP_KEY` in `.env`) |
| `src/clean_firms.py` | Cleans detections and saves `data/raw/firms_lebanon.parquet` |
| `src/check_firms.py` | Prints summary checks on the cleaned fire data |
| `src/download_boundary.py` | Downloads Lebanon's border (geoBoundaries) |
| `src/grid.py` | Builds the 5 km grid, land fractions and cell points |
| `src/join_fires.py` | Assigns fires to cells and flags persistent hotspots |
| `src/check_inputs.py` | Checks the grid points and fire date range before the weather download |
| `src/weather.py` | Builds the 0.1° weather grid and downloads all weather batches (run once per day until complete) |
| `src/combine_weather.py` | Merges the batches into one weather table per cell per day |
| `src/check_weather.py` | Monthly climate summary + coast vs inland check chart |

## Pipeline Order

1. `python src\fetch_firms.py`
2. `python src\clean_firms.py`
3. `python src\download_boundary.py`
4. `python src\grid.py`
5. `python src\join_fires.py`
6. `python src\weather.py` *(repeat daily until "All batches downloaded.")*
7. `python src\combine_weather.py`
8. `python src\check_weather.py`