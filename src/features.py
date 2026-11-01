"""Build the model-ready feature table for the Lebanon wildfire-risk model.  # what this file does

One row = one cell on one day (t).  # the unit of the table
Label = fire_tomorrow: 1 if the cell has a fire on day t+1, else 0.  # what the model predicts

LEAKAGE RULE: a feature for day t may only use data available by the end of day t.  # the rule every step follows
"""  # end of the module description

import json  # save the feature-name list as a small text file
from pathlib import Path  # clean file paths that work on Windows

import geopandas as gpd  # read the grid geometry (needed for neighbors)
import numpy as np  # math helpers (sin/cos, logs)
import pandas as pd  # all the table work



DATA = Path("data/processed")  # folder with all cleaned tables
FIRES_PATH = DATA / "fires_with_cells.parquet"  # Day 3: fire detections with cell_id
GRID_PATH = DATA / "lebanon_grid.gpkg"  # Day 3: the 488 cells with shapes
TERRAIN_PATH = DATA / "terrain.parquet"  # Day 5: per-cell terrain stats
NDVI_PATH = DATA / "ndvi.parquet"  # Day 6: merged NDVI composites
WEATHER_PATH = DATA / "weather_daily.parquet"  # Day 4/6: daily weather per cell
OUT_PATH = DATA / "features.parquet"  # today's output: the model-ready table
FEATURE_LIST_PATH = DATA / "feature_columns.json"  # the exact feature names, shared with training and the API
ID_COLS = ["cell_id", "date"]  # identify a row, but are not features
TARGET = "fire_tomorrow"  # the label
START_DATE = pd.Timestamp("2012-01-25")  # first day with fire data (VIIRS starts here)
FIRE_WINDOWS = [7, 30, 365]  # look-back windows (days) for fire-history counts
WEATHER_WINDOWS = [3, 7]  # short windows for heat, dryness and wind
RAIN_WINDOWS = [7, 30, 90]  # longer windows for rainfall totals


def load_fire_days():  # collapse fire detections into one row per cell per fire day
    fires = pd.read_parquet(FIRES_PATH)  # all detections with cell_id and the hotspot flag
    last_date = pd.to_datetime(fires["acq_date"]).max().normalize()  # last day covered by the fire data (taken before filtering)
    fires = fires[~fires["hotspot"]].copy()  # drop persistent industrial heat sources (Day 3 rule)
    fires["date"] = pd.to_datetime(fires["acq_date"]).dt.normalize()  # date only, no time part
    daily = (  # aggregate detections per cell per day
        fires.groupby(["cell_id", "date"])  # one group = one cell on one day
        .agg(n_detections=("frp", "size"), frp_sum=("frp", "sum"))  # how many detections, and total fire power
        .reset_index()  # cell_id and date back to normal columns
    )  # end of aggregation
    return daily, last_date  # the fire days, plus the end of the panel


def build_panel(daily, cell_ids, end_date):  # every cell x every day, with today's fire and tomorrow's label
    dates = pd.date_range(START_DATE, end_date, freq="D")  # every calendar day from the start to the last fire-data day
    idx = pd.MultiIndex.from_product([cell_ids, dates], names=["cell_id", "date"])  # all cell-day combinations
    panel = pd.DataFrame(index=idx).reset_index()  # turn the combinations into a table with two columns
    panel = panel.merge(daily, on=["cell_id", "date"], how="left")  # attach fire info; days without fire get NaN
    panel[["n_detections", "frp_sum"]] = panel[["n_detections", "frp_sum"]].fillna(0)  # no detections means 0, not missing
    panel["fire_today"] = (panel["n_detections"] > 0).astype("int8")  # 1 if the cell burned on day t
    panel = panel.sort_values(["cell_id", "date"]).reset_index(drop=True)  # each cell's days in time order (required for shift)
    panel["fire_tomorrow"] = panel.groupby("cell_id")["fire_today"].shift(-1)  # the label: the same cell's fire on day t+1
    return panel  # the base table every other step adds columns to


def add_fire_history(panel):  # past-fire features per cell, using day t and earlier only
    g = panel.groupby("cell_id")["fire_today"]  # each cell's daily fire series (panel is already sorted by cell, date)
    for w in FIRE_WINDOWS:  # 7, 30 and 365 days
        panel[f"fires_last_{w}d"] = g.transform(lambda s: s.rolling(w, min_periods=1).sum()).astype("float32")  # fire days in the last w days, including today

    fire_dates = panel["date"].where(panel["fire_today"] == 1)  # the date on fire days, NaT on other days
    last_fire = fire_dates.groupby(panel["cell_id"]).ffill()  # carry each cell's latest fire date forward in time
    days_since = (panel["date"] - last_fire).dt.days  # days between today and the last fire (0 if burning today)
    panel["days_since_fire"] = days_since.fillna(3650).clip(upper=3650).astype("float32")  # never burned yet = 10 years, and cap everything at 10 years

    cum_fires = g.cumsum()  # running total of fire days per cell, up to and including today
    days_seen = panel.groupby("cell_id").cumcount() + 1  # how many days of history the cell has so far
    panel["cell_fire_rate"] = (cum_fires / days_seen).astype("float32")  # the cell's long-run fire frequency, from the past only
    return panel  # the panel with 5 new columns

def neighbor_matrix(grid, cell_ids):  # 488 x 488 matrix: 1 where two cells touch (sides or corners)
    shapes = grid[["cell_id", "geometry"]].copy()  # only the ID and the shape
    shapes["geometry"] = shapes.geometry.buffer(10)  # grow each cell by 10 m so tiny rounding gaps don't hide a shared edge
    pairs = gpd.sjoin(shapes, shapes, predicate="intersects")  # every pair of cells that overlap after the buffer
    pairs = pairs[pairs["cell_id_left"] != pairs["cell_id_right"]]  # a cell is not its own neighbor
    pos = {c: i for i, c in enumerate(cell_ids)}  # cell_id -> row/column position in the matrix
    A = np.zeros((len(cell_ids), len(cell_ids)), dtype="float32")  # start with no neighbors
    A[pairs["cell_id_left"].map(pos).to_numpy(), pairs["cell_id_right"].map(pos).to_numpy()] = 1  # mark each neighbor pair
    return A  # the adjacency matrix


def add_neighbor_fires(panel, A, cell_ids):  # how many neighboring cells burned today and in the last 7 days
    wide = panel.pivot(index="date", columns="cell_id", values="fire_today")[cell_ids]  # days x cells table of 0/1 fires
    nbr = wide.to_numpy(dtype="float32") @ A.T  # matrix multiply: for each day and cell, sum fire_today over its neighbors
    nbr_today = pd.DataFrame(nbr, index=wide.index, columns=pd.Index(cell_ids, name="cell_id"))  # back to a labeled table
    nbr_7d = nbr_today.rolling(7, min_periods=1).sum()  # neighbor fire-days over the last 7 days (looks back only)
    for name, df in [("nbr_fires_today", nbr_today), ("nbr_fires_last_7d", nbr_7d)]:  # handle both tables the same way
        long = df.stack().rename(name).reset_index()  # wide table -> one row per date and cell
        panel = panel.merge(long, on=["date", "cell_id"], how="left")  # attach to the panel (keeps the panel's row order)
        panel[name] = panel[name].astype("float32")  # smaller type to save memory
    n_nbrs = pd.Series(A.sum(axis=1), index=cell_ids, name="n_neighbors")  # how many neighbors each cell has (border cells have fewer)
    panel["n_neighbors"] = panel["cell_id"].map(n_nbrs).astype("int8")  # attach to every row
    return panel  # the panel with 3 new columns


def add_weather(panel):  # today's weather plus rolling windows, all looking back from day t
    rename = {  # long Open-Meteo names -> short feature names
        "temperature_2m_max": "tmax",  # daily max temperature (°C)
        "relative_humidity_2m_min": "rh_min",  # driest humidity of the day (%)
        "precipitation_sum": "precip",  # rain for the day (mm)
        "wind_speed_10m_max": "wind_max",  # strongest wind (km/h)
        "wind_gusts_10m_max": "gust_max",  # strongest gust (km/h)
        "et0_fao_evapotranspiration": "et0",  # water the air pulls from the ground (mm)
        "vapour_pressure_deficit_max": "vpd_max",  # how "thirsty" the air is (kPa)
    }  # end of the name map
    weather = pd.read_parquet(WEATHER_PATH, columns=["cell_id", "date", *rename]).rename(columns=rename)  # load only what we need, with short names
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()  # same date format as the panel
    weather[list(rename.values())] = weather[list(rename.values())].astype("float32")  # smaller numbers to save memory
    panel = panel.merge(weather, on=["cell_id", "date"], how="left")  # attach weather; days not downloaded yet stay NaN

    g = panel.groupby("cell_id")  # each cell's days in time order
    for w in WEATHER_WINDOWS:  # 3 and 7 days
        panel[f"tmax_mean_{w}d"] = g["tmax"].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")  # average heat
        panel[f"rh_min_mean_{w}d"] = g["rh_min"].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")  # average dryness of the air
        panel[f"vpd_max_mean_{w}d"] = g["vpd_max"].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")  # average air thirst
        panel[f"wind_max_{w}d"] = g["wind_max"].transform(lambda s: s.rolling(w, min_periods=1).max()).astype("float32")  # strongest wind in the window
    for w in RAIN_WINDOWS:  # 7, 30 and 90 days
        panel[f"precip_sum_{w}d"] = g["precip"].transform(lambda s: s.rolling(w, min_periods=1).sum()).astype("float32")  # total rain in the window
    panel["et0_sum_30d"] = g["et0"].transform(lambda s: s.rolling(30, min_periods=1).sum()).astype("float32")  # water lost by the land over a month

    dry = panel["precip"] < 1  # a "dry day" = under 1 mm of rain
    block = (~dry).astype("int32").groupby(panel["cell_id"]).cumsum()  # a new block starts every time it rains
    streak = dry.astype("int32").groupby([panel["cell_id"], block]).cumsum()  # count dry days inside each block
    panel["dry_streak"] = streak.where(panel["precip"].notna()).astype("float32")  # NaN where there's no weather yet, not a fake 0
    return panel  # the panel with 20 new weather columns


def add_weather_anomalies(panel):  # this week's weather minus the cell's normal for this month, from PAST years only
    base = {  # daily variable used for the normal -> feature compared against it
        "tmax": "tmax_mean_7d",  # hotter than usual this week?
        "rh_min": "rh_min_mean_7d",  # drier air than usual this week?
        "vpd_max": "vpd_max_mean_7d",  # thirstier air than usual this week?
        "et0": "et0",  # the land losing more water than usual today?
    }  # end of the map
    year = panel["date"].dt.year.rename("year")  # calendar year of each row
    month = panel["date"].dt.month.rename("month")  # calendar month of each row
    monthly = panel.groupby(["cell_id", year, month])[list(base)].mean().reset_index()  # one average per cell, year and month
    monthly = monthly.sort_values(["cell_id", "month", "year"])  # for each cell-month, the years in time order
    normals = monthly.groupby(["cell_id", "month"])[list(base)].transform(lambda s: s.shift().expanding().mean())  # average of PREVIOUS years only (shift skips the current year)
    normals.columns = [f"{c}_normal" for c in base]  # e.g. tmax -> tmax_normal
    monthly = pd.concat([monthly[["cell_id", "year", "month"]], normals], axis=1)  # keys + normals side by side
    keys = pd.DataFrame({"cell_id": panel["cell_id"], "year": year, "month": month})  # the key of every panel row
    norm = keys.merge(monthly, on=["cell_id", "year", "month"], how="left")  # each row's normal, in the panel's row order
    for var, feat in base.items():  # build one anomaly per variable
        panel[f"{var}_anom"] = (panel[feat] - norm[f"{var}_normal"].to_numpy()).astype("float32")  # positive = above normal, negative = below
    return panel  # the panel with 4 new columns



def add_ndvi(panel):  # latest published vegetation index, its age, change and anomaly
    ndvi = pd.read_parquet(NDVI_PATH, columns=["cell_id", "ndvi", "valid_frac", "available_date"])  # use Day 5's availability date
    ndvi = ndvi[ndvi["ndvi"].notna() & (ndvi["valid_frac"] >= 0.2)].copy()  # drop cloudy composites (under 20% usable pixels)
    ndvi["avail_date"] = pd.to_datetime(ndvi["available_date"]).dt.normalize().astype("datetime64[ns]")  # end date + 3 days, set on Day 5
    obs = ndvi.groupby(["cell_id", "avail_date"], as_index=False)["ndvi"].mean()  # Terra + Aqua on the same day -> one value
    obs = obs.sort_values(["cell_id", "avail_date"])  # each cell's composites in time order
    obs["month"] = obs["avail_date"].dt.month  # month of each composite
    obs["ndvi_clim"] = obs.groupby(["cell_id", "month"])["ndvi"].transform(lambda s: s.shift().expanding().mean())  # this cell's average for this month in PAST years only
    
    panel["date"] = panel["date"].astype("datetime64[ns]")  # same date precision as avail_date, which merge_asof requires
    panel = panel.sort_values("date")  # merge_asof needs both tables sorted by the time key
    obs = obs.sort_values("avail_date")  # same for the NDVI table
    panel = pd.merge_asof(  # for each panel row, take the latest composite available on or before that day
        panel,  # left table: every cell-day
        obs[["cell_id", "avail_date", "ndvi", "ndvi_clim"]],  # right table: published composites
        left_on="date",  # the day we're predicting from
        right_on="avail_date",  # when the composite became available
        by="cell_id",  # match within the same cell only
        direction="backward",  # only look into the past, never the future
    )  # end of the as-of join
    panel = panel.sort_values(["cell_id", "date"]).reset_index(drop=True)  # back to cell, date order (other steps rely on it)

    panel["ndvi_age_days"] = (panel["date"] - panel["avail_date"]).dt.days.astype("float32")  # how old the NDVI value is (large in the 2025 gap)
    panel["ndvi_anomaly"] = (panel["ndvi"] - panel["ndvi_clim"]).astype("float32")  # greener (+) or browner (-) than this cell usually is in this month
    panel["ndvi_change_30d"] = (panel["ndvi"] - panel.groupby("cell_id")["ndvi"].shift(30)).astype("float32")  # how fast vegetation is drying out
    panel["ndvi"] = panel["ndvi"].astype("float32")  # smaller type
    return panel.drop(columns=["avail_date", "ndvi_clim"])  # remove helper columns, keep 4 NDVI features

def add_static_and_calendar(panel, grid):  # per-cell terrain + land share, and season features
    terrain = pd.read_parquet(TERRAIN_PATH)  # 8 terrain columns per cell
    terrain["elev_std"] = terrain["elev_std"].fillna(terrain["elev_std"].median())  # fill the one missing value with a typical one
    static = terrain.merge(grid[["cell_id", "land_frac"]], on="cell_id", how="left")  # add the share of each cell that is land
    float_cols = [c for c in static.columns if c != "cell_id"]  # every static feature
    static[float_cols] = static[float_cols].astype("float32")  # smaller type
    panel = panel.merge(static, on="cell_id", how="left")  # copy the cell's static values onto every day

    doy = panel["date"].dt.day_of_year  # day of year, 1 to 366 (new pandas name)
    panel["month"] = panel["date"].dt.month.astype("int8")  # calendar month 1-12
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")  # position on the yearly circle (part 1)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")  # position on the yearly circle (part 2)
    panel["day_of_week"] = panel["date"].dt.day_of_week.astype("int8")  # 0 = Monday ... 6 = Sunday (new pandas name)
    return panel  # the panel with 13 new columns

def save_features(panel):  # order the columns, save the table and the feature list
    feature_cols = [c for c in panel.columns if c not in ID_COLS + [TARGET]]  # everything except IDs and the label
    panel = panel[ID_COLS + feature_cols + [TARGET]]  # clean order: IDs, features, label
    panel.to_parquet(OUT_PATH, index=False)  # save the model-ready table
    FEATURE_LIST_PATH.write_text(json.dumps(feature_cols, indent=2))  # save the names, one per line
    size_mb = OUT_PATH.stat().st_size / 1e6  # file size in megabytes
    print(f"Saved {OUT_PATH} ({size_mb:.0f} MB) with {len(feature_cols)} features")  # confirm
    print(f"Saved {FEATURE_LIST_PATH}")  # confirm
    return feature_cols  # handy for the checks below





def main():  # runs every step in order
    grid = gpd.read_file(GRID_PATH)  # the 488 cells
    cell_ids = sorted(grid["cell_id"].unique())  # list of cell IDs in order
    daily, last_date = load_fire_days()  # fire days without hotspots, and the last data day
    panel = build_panel(daily, cell_ids, last_date)  # full cell x day table with the label
    panel = add_fire_history(panel)  # recent and long-run fire history per cell
    A = neighbor_matrix(grid, cell_ids)  # who touches whom
    panel = add_neighbor_fires(panel, A, cell_ids)  # neighbor fire counts
    panel = add_weather(panel)  # weather and dryness features
    panel = add_weather_anomalies(panel)  # weather compared with each cell's past monthly normal
    panel = add_ndvi(panel)  # vegetation features
    panel = add_static_and_calendar(panel, grid)  # terrain, land share and calendar    
    print("Panel shape:", panel.shape)  # rows, columns
    print("Dates:", panel["date"].min().date(), "->", panel["date"].max().date())  # time coverage
    print("Fire cell-days:", int(panel["fire_today"].sum()))  # how many cell-days burned (without hotspots)
    print(f"Positive rate (fire_tomorrow): {panel['fire_tomorrow'].mean():.4%}")  # class imbalance of the label
    print("Missing labels:", int(panel["fire_tomorrow"].isna().sum()))  # should equal 488 (each cell's last day has no tomorrow)
    feature_cols = save_features(panel)  # write features.parquet and feature_columns.json
    missing = panel[feature_cols].isna().mean().sort_values(ascending=False)  # share of missing values per feature
    print("Features with missing values (top 5):")  # header for the next print
    print(missing[missing > 0].head().round(3).to_dict())  # mostly weather (until the download finishes) and a little NDVI
    


if __name__ == "__main__":  # only runs when you call: python src/features.py
    main()  # start the build
   