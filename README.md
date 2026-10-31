# Lebanon Wildfire-Risk Predictor

Predicts next-day wildfire risk for every ~5 km grid cell across Lebanon and shows it on a live map.

## Project Status

🚧 **In progress — Day 5 of 18 complete (Day 4 weather download still running)**

| Day | Task | Status |
|----|------|--------|
| 1 | Setup (repo, venv, structure) | ✅ Done |
| 2 | Fire data (NASA FIRMS) | ✅ Done |
| 3 | Lebanon grid (~5 km cells) | ✅ Done |
| 4 | Weather data (Open-Meteo) | ⏳ Download in progress |
| 5 | Vegetation (NDVI) + terrain | ✅ Done |
| 6 | Exploratory data analysis | ⏳ Next |
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
- **MODIS NDVI** (Terra + Aqua, 250 m, 16-day composites) via Microsoft Planetary Computer: vegetation greenness
- **Copernicus DEM** (90 m) via Microsoft Planetary Computer: elevation and terrain

## Stack
Python 3.12 · pandas · geopandas · xarray · rioxarray · odc-stac · LightGBM · MLflow · Optuna · SHAP · FastAPI · Docker · Streamlit

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

### Vegetation + terrain (Day 5)

**Vegetation (NDVI)**
- Source: MODIS 250 m 16-day NDVI from **both Terra and Aqua**, searched through the Planetary Computer STAC catalog: **663 composites, Nov 2011 – Sep 2026**
- Terra and Aqua composites are offset by 8 days, so combining them gives a **fresh NDVI value every 8 days** instead of every 16
- Lebanon sits on the border of two MODIS tiles (h20v05, h21v05); both are stitched and reprojected onto one fixed 250 m UTM grid
- **Fast per-cell averaging:** grid cells (clipped to Lebanon's border) are rasterized onto the pixel grid once, so each date's per-cell mean is a single `numpy.bincount` instead of 488 polygon clips
- Reprocessed duplicate images are removed (newest version kept per date and tile)
- **Parallel, resumable download:** a thread pool processes several dates at once; each date is saved as its own file and finished dates are skipped on rerun, so crashes and expired download links cost nothing
- **Leakage guard:** every value gets an `available_date` = composite end date + 3 days (NASA's processing lag), so the model never uses vegetation data before it actually existed
- Data quality: only **0.9% missing** (clouds); 3 tiny coastal sliver cells never covered; **catalog gap from 2025-06-26 to 2025-11-17** (most of the 2025 fire season). It is handled for now with forward-filling plus an "NDVI age" feature, and will be filled from NASA AppEEARS before validation.
- Seasonal cycle: NDVI peaks in **April (~0.41)** and is lowest in **August–September (~0.30)**, right when fires peak

**Terrain**
- Source: Copernicus DEM 90 m, reprojected to UTM 36N (meters), so slope math is correct
- Slope and aspect computed from elevation gradients; aspect turned into **northness** (+1 north-facing, −1 south-facing) and **eastness**
- Per-cell features: mean / max / std elevation, mean / max slope, mean northness, mean eastness, and **`south_steep_frac`**: the share of a cell that is both steep (> 10°) and south-facing. These are the sunniest, driest, fastest-burning slopes.
- Sanity check: highest cell max ≈ 3,080 m (Qurnat as Sawda, 3,088 m); slopes mostly face west, toward the sea, as expected for Mount Lebanon

**Outputs:** `data/processed/ndvi.parquet` (one row per cell per composite), `data/processed/terrain.parquet` (one row per cell)

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
| `src/ndvi.py` | Downloads MODIS NDVI for every date and saves per-cell means (parallel, resumable) |
| `src/combine_ndvi.py` | Combines the NDVI files, adds `available_date` and runs date/gap checks |
| `src/terrain.py` | Downloads the 90 m DEM, computes slope/aspect and saves per-cell terrain features |
| `src/check_day5.py` | Maps of NDVI, elevation and steep south-facing share + NDVI seasonal cycle |

## Pipeline Order

1. `python src\fetch_firms.py`
2. `python src\clean_firms.py`
3. `python src\download_boundary.py`
4. `python src\grid.py`
5. `python src\join_fires.py`
6. `python src\weather.py` *(repeat daily until "All batches downloaded.")*
7. `python src\combine_weather.py`
8. `python src\check_weather.py`
9. `python src\ndvi.py` *(rerun until no dates fail)*
10. `python src\combine_ndvi.py`
11. `python src\terrain.py`
12. `python src\check_day5.py`