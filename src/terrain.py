from pathlib import Path  # clean file paths

import geopandas as gpd  # read the grid
import matplotlib.pyplot as plt  # quick visual check
import odc.stac  # loads satellite tiles onto a pixel grid
import planetary_computer  # signs download links
import pystac_client  # searches the catalog
import numpy as np  # array math for the gradients
import rioxarray  # opens and saves GeoTIFFs, and adds .rio tools to arrays
from odc.geo.geobox import GeoBox  # describes a pixel grid
import pandas as pd  # build the per-cell table
from rasterio.features import rasterize  # paint cell polygons onto pixels


GRID_PATH = Path("data/processed/lebanon_grid.gpkg")  # Day 3 grid
DEM_PATH = Path("data/raw/dem_90m.tif")  # where the elevation file will be saved
BOUNDARY_PATH = Path("data/raw/lebanon_boundary.geojson")  # Lebanon outline, to drop sea and Syria pixels
OUT_PATH = Path("data/processed/terrain.parquet")  # final per-cell terrain table
CRS = "EPSG:32636"  # UTM zone 36N, meters, same as the grid
RES = 90  # 90 m pixels, small download and plenty for 5 km cells
LEBANON_BBOX = [35.1, 33.05, 36.65, 34.7]  # [min lon, min lat, max lon, max lat]
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"  # catalog address


def download_dem():  # fetch, stitch, reproject and save the elevation data
    grid = gpd.read_file(GRID_PATH)  # load the grid to know the area to cover
    geobox = GeoBox.from_bbox(tuple(grid.total_bounds), crs=CRS, resolution=RES)  # 90 m pixel grid over all cells

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)  # connect and sign links
    items = list(catalog.search(collections=["cop-dem-glo-90"], bbox=LEBANON_BBOX).items())  # Copernicus DEM 90 m tiles touching Lebanon
    print("DEM tiles found:", len(items))  # each tile is 1 x 1 degree, expect about 4

    ds = odc.stac.load(  # download and merge the tiles onto our grid
        items,  # all tiles
        bands=["data"],  # the elevation layer
        geobox=geobox,  # our exact 90 m UTM grid
        groupby="solar_day",  # all tiles share one date, so they merge into one image
        resampling="bilinear",  # smooth interpolation, right for continuous values like height
    )  # end of load
    dem = ds["data"].isel(time=0)  # drop the time dimension -> plain 2D map of meters above sea level

    DEM_PATH.parent.mkdir(parents=True, exist_ok=True)  # make sure data/raw exists
    dem.rio.to_raster(DEM_PATH)  # save as GeoTIFF with its projection info
    return dem  # hand it back for checking

def slope_aspect(dem):  # turn heights into steepness and facing direction
    z = dem.values.astype("float64")  # heights as a plain numpy array, float64 for precise math
    dz_drow, dz_dcol = np.gradient(z, RES)  # height change per meter down the rows and across the columns (pixels are 90 m apart)
    dz_dnorth = -dz_drow  # rows go from north to south, so flip the sign to get change per meter going north
    dz_deast = dz_dcol  # columns already go west to east

    steep = np.hypot(dz_deast, dz_dnorth)  # size of the height change in any direction (rise over run)
    slope = np.degrees(np.arctan(steep))  # convert rise/run into an angle in degrees (0 = flat)

    with np.errstate(invalid="ignore", divide="ignore"):  # flat pixels divide 0 by 0; silence that warning
        northness = np.where(steep > 0, -dz_dnorth / steep, 0.0)  # downhill direction's north part: +1 faces north, -1 faces south, 0 if flat
        eastness = np.where(steep > 0, -dz_deast / steep, 0.0)  # downhill direction's east part: +1 faces east, -1 faces west, 0 if flat

    return (  # wrap each result back into a map with the same coordinates as the DEM
        dem.copy(data=slope),  # slope map in degrees
        dem.copy(data=northness),  # northness map
        dem.copy(data=eastness),  # eastness map
    )  # end of return

def cell_terrain_stats(dem, slope, northness, eastness):  # summarize the terrain maps into one row per cell
    grid = gpd.read_file(GRID_PATH)  # the 488 cells
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(grid.crs)  # outline in the same projection
    land_cells = gpd.clip(grid, boundary)  # keep only the part of each cell inside Lebanon

    cell_raster = rasterize(  # paint each 90 m pixel with its cell ID
        shapes=zip(land_cells.geometry, land_cells["cell_id"]),  # (shape, value) pairs
        out_shape=dem.shape,  # same rows and columns as the DEM
        transform=dem.rio.transform(),  # same position on the map as the DEM
        fill=-1,  # outside Lebanon = -1
        dtype="int32",  # whole numbers
    )  # end of rasterize
    inside = cell_raster >= 0  # True for pixels inside a Lebanon cell

    pixels = pd.DataFrame({  # one row per pixel inside Lebanon
        "cell_id": cell_raster[inside],  # which cell
        "elev": dem.values[inside],  # height in meters
        "slope": slope.values[inside],  # steepness in degrees
        "northness": northness.values[inside],  # facing north (+1) or south (-1)
        "eastness": eastness.values[inside],  # facing east (+1) or west (-1)
    })  # end of table
    pixels["south_steep"] = (pixels["northness"] < -0.5) & (pixels["slope"] > 10)  # True for steep, mostly south-facing pixels

    stats = pixels.groupby("cell_id").agg(  # summarize per cell
        elev_mean=("elev", "mean"),  # average height
        elev_max=("elev", "max"),  # highest point
        elev_std=("elev", "std"),  # how rugged: big spread of heights = mountainous
        slope_mean=("slope", "mean"),  # average steepness
        slope_max=("slope", "max"),  # steepest pixel
        northness_mean=("northness", "mean"),  # overall facing north or south
        eastness_mean=("eastness", "mean"),  # overall facing east or west
        south_steep_frac=("south_steep", "mean"),  # share of the cell that is steep and south-facing (True counts as 1)
    ).reset_index()  # turn cell_id back into a normal column

    return grid[["cell_id"]].merge(stats, on="cell_id", how="left")  # keep all 488 cells; cells with no pixels get NaN




if __name__ == "__main__":  # runs only when this file is run directly
    if DEM_PATH.exists():  # we already downloaded it in Step 7
        dem = rioxarray.open_rasterio(DEM_PATH).squeeze("band", drop=True)  # open the saved file and drop the single 'band' dimension
    else:  # first run on a new machine
        dem = download_dem()  # download it

    slope, northness, eastness = slope_aspect(dem)  # compute the three terrain maps
    print("Slope min / mean / max:", round(float(slope.min()), 1), round(float(slope.mean()), 1), round(float(slope.max()), 1))  # degrees
    print("Northness min / max:", round(float(northness.min()), 2), round(float(northness.max()), 2))  # should be -1 and 1

    terrain = cell_terrain_stats(dem, slope, northness, eastness)  # build the per-cell table
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # make sure data/processed exists
    terrain.to_parquet(OUT_PATH, index=False)  # save it
    print("Terrain table shape:", terrain.shape)  # expect (488, 9)
    print("Cells with no terrain data:", int(terrain["elev_mean"].isna().sum()))  # tiny slivers, maybe 0 at 90 m
    print(terrain.describe().round(2).T)  # quick summary of every column (T = flip so columns become rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))  # two maps side by side
    slope.plot(ax=axes[0], cmap="magma", vmax=40)  # left: steepness, capped at 40° so details show
    axes[0].set_title("Slope (degrees)")  # left title
    northness.plot(ax=axes[1], cmap="coolwarm_r", vmin=-1, vmax=1)  # right: blue = north-facing, red = south-facing
    axes[1].set_title("Northness (+1 north, -1 south)")  # right title
    plt.tight_layout()  # avoid overlapping labels
    plt.show()  # open the plots