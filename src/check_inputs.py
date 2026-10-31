from pathlib import Path  # Path builds file paths that work on any OS
import pandas as pd  # pandas loads the CSV and Parquet files

CENTROIDS_PATH = Path("data/processed/grid_points.csv")  # one lat/lon point per grid cell (Day 3)
FIRES_PATH = Path("data/processed/fires_with_cells.parquet")  # fire detections with cell_id (Day 3)

if __name__ == "__main__":  # only run this block when the file is executed directly
    cells = pd.read_csv(CENTROIDS_PATH)  # load the cell points
    print("Cells:", len(cells))  # should be 488
    print("Columns:", list(cells.columns))  # should include cell_id, lat, lon
    print(cells.head())  # preview the first 5 cells

    fires = pd.read_parquet(FIRES_PATH)  # load the fires
    fires["acq_date"] = pd.to_datetime(fires["acq_date"])  # make sure the date column is a real date type
    print("Fires:", len(fires))  # number of detections inside Lebanon
    print("First fire date:", fires["acq_date"].min().date())  # earliest day we need weather for
    print("Last fire date:", fires["acq_date"].max().date())  # latest day we need weather for
    n_days = (fires["acq_date"].max() - fires["acq_date"].min()).days + 1  # total days in the range, counting both ends
    print("Days in range:", n_days)  # number of days of weather per cell
    print("Rows to expect:", n_days * len(cells))  # cells × days = size of the final weather table