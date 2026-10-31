from pathlib import Path  # Path builds file paths that work on any OS
import pandas as pd  # pandas loads and summarizes the tables
import matplotlib.pyplot as plt  # draws the check charts

WEATHER_PATH = Path("data/processed/weather_daily.parquet")  # combined weather from step 7
CELLS_PATH = Path("data/processed/grid_points.csv")  # cell lat/lon from Day 3
FIG_PATH = Path("reports/figures/weather_check.png")  # where the chart image is saved


if __name__ == "__main__":  # only run this block when the file is executed directly
    weather = pd.read_parquet(WEATHER_PATH)  # load the weather table
    weather["month"] = weather["date"].dt.month  # month number, 1 to 12

    cols = ["temperature_2m_max", "relative_humidity_2m_min", "precipitation_sum", "vapour_pressure_deficit_max"]  # key fire-weather columns
    monthly = weather.groupby("month")[cols].mean().round(1)  # average of each column for each month
    print(monthly)  # expect Jul–Aug hottest and driest, Dec–Feb wettest

    cells = pd.read_csv(CELLS_PATH)  # load cell coordinates
    cells = cells[cells["cell_id"].isin(weather["cell_id"])]  # keep only cells we have weather for
    west_id = cells.loc[cells["lon"].idxmin(), "cell_id"]  # westernmost cell (on the coast)
    east_id = cells.loc[cells["lon"].idxmax(), "cell_id"]  # easternmost cell (inland)
    print("West cell:", west_id, "| East cell:", east_id)  # show which cells are being compared

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # two charts side by side
    for cid, label in [(west_id, "West (coast)"), (east_id, "East (inland)")]:  # loop over the two cells
        one = weather[weather["cell_id"] == cid]  # this cell's rows
        temp = one.groupby("month")["temperature_2m_max"].mean()  # its average max temperature per month
        axes[0].plot(temp.index, temp.values, marker="o", label=label)  # one line per cell
    axes[0].set_title("Avg max temperature by month")  # left chart title
    axes[0].set_xlabel("Month")  # x-axis label
    axes[0].set_ylabel("°C")  # y-axis label
    axes[0].legend()  # show which line is which

    axes[1].bar(monthly.index, monthly["precipitation_sum"])  # average daily rain per month, all cells
    axes[1].set_title("Avg daily rain by month (all cells)")  # right chart title
    axes[1].set_xlabel("Month")  # x-axis label
    axes[1].set_ylabel("mm")  # y-axis label

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)  # create reports/figures if it doesn't exist
    fig.tight_layout()  # stop labels from overlapping
    fig.savefig(FIG_PATH, dpi=120)  # save the chart for the README later
    print(f"Saved {FIG_PATH}")  # confirm the save
    plt.show()  # open the chart window