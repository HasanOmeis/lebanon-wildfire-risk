import geopandas as gpd  # read the grid
import matplotlib.pyplot as plt  # plotting
import pandas as pd  # read the tables

grid = gpd.read_file("data/processed/lebanon_grid.gpkg")  # the 488 cells with their shapes
terrain = pd.read_parquet("data/processed/terrain.parquet")  # per-cell terrain table from Step 9
ndvi = pd.read_parquet("data/processed/ndvi.parquet")  # per-cell NDVI over time from Step 6

ndvi_mean = ndvi.groupby("cell_id")["ndvi"].mean().rename("ndvi_mean").reset_index()  # average greenness of each cell over all 15 years
cells = grid.merge(terrain, on="cell_id").merge(ndvi_mean, on="cell_id")  # attach terrain and NDVI to the cell shapes

fig, axes = plt.subplots(1, 3, figsize=(15, 7))  # three maps side by side
cells.plot(column="ndvi_mean", cmap="RdYlGn", legend=True, ax=axes[0], missing_kwds={"color": "lightgrey"})  # greenness; grey = no data
axes[0].set_title("Average NDVI 2012–2026")  # title 1
cells.plot(column="elev_mean", cmap="terrain", legend=True, ax=axes[1])  # average height per cell
axes[1].set_title("Mean elevation (m)")  # title 2
cells.plot(column="south_steep_frac", cmap="OrRd", legend=True, ax=axes[2])  # share of steep south-facing slopes
axes[2].set_title("Steep south-facing share")  # title 3
for ax in axes:  # for every map
    ax.set_axis_off()  # hide the meter coordinates for a cleaner look
plt.tight_layout()  # avoid overlaps
plt.show()  # open the figure

monthly = ndvi.groupby(ndvi["start_date"].dt.month)["ndvi"].mean()  # average NDVI per calendar month across all cells and years
print(monthly.round(3))  # expect a peak around March–April and a low around August–September