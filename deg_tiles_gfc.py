# Theo Kerr

# Imports/env settings
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import os
console = Console()
gdal.UseExceptions()


# =========================
# Define custom functions
# =========================
class InvalidChangeYears(Exception):
    pass

def get_raster_info(raster_path):
    """Opens a raster at the given path and returns dataset and info.

    Args:
        raster_path (str): path to a raster dataset.

    Returns:
        tuple: returns a GDAL raster and info dictionary.
    """
    ds = gdal.Open(raster_path)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    transform = ds.GetGeoTransform()
    projection = ds.GetSpatialRef()
    info = {"xsize": xsize,
            "ysize": ysize,
            "transform": transform,
            "projection": projection,        
            "xmin": transform[0], 
            "ymax": transform[3],
            "xmax": transform[0] + transform[1] * xsize,
            "ymin": transform[3] + transform[5] * ysize,
            "pixel_size": transform[1]}
    
    return ds, info

def rasterize(shapefile, output_tiff, info):
    console.log(f"Rasterizing {os.path.basename(output_tiff)}...", style="dim cyan") 
    # Initialize the raster
    output = gdal.GetDriverByName("GTiff").Create(output_tiff, info["xsize"], info["ysize"], 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=100"])
    band1 = output.GetRasterBand(1)
    band1.SetNoDataValue(255)
    band2 = output.GetRasterBand(2)
    band2.SetNoDataValue(255)
    output.SetGeoTransform(info["transform"])
    output.SetProjection(info["projection"].ExportToWkt())

    # Create options and rasterize year_start into band 1
    b1_rasterize_options = gdal.RasterizeOptions(format="GTiff",
                                                callback=gdal.TermProgress_nocb,
                                                bands=[1],
                                                attribute="year_start")
    gdal.Rasterize(output_tiff, shapefile, options=b1_rasterize_options)
    
    # Create options and rasterize year_end into band 2
    b2_rasterize_options = gdal.RasterizeOptions(format="GTiff",
                                        callback=gdal.TermProgress_nocb,
                                        bands=[2],
                                        attribute="year_end")
    gdal.Rasterize(output_tiff, shapefile, options=b2_rasterize_options)

    console.log(f"\u2713 Rasterized {os.path.basename(shapefile)}", style="dim green")
    
    return output_tiff

def mask_gfc(gfc_band, year_start_band, year_end_band, output_band, x, y, cols, rows):
    # Read in arrays at current location
    gfc_array = gfc_band.ReadAsArray(x, y, cols, rows).astype(np.uint8)
    year_start = year_start_band.ReadAsArray(x, y, cols, rows).astype(np.uint8)
    year_end = year_end_band.ReadAsArray(x, y, cols, rows).astype(np.uint8)
    
    # Case 1: Both NaN → natural change, extract modal year
    natural_null = np.isnan(year_start) & np.isnan(year_end)
    if natural_null.any():
        gfc_natural_null = gfc_array[natural_null & (gfc_array != 0)]
        if gfc_natural_null.size > 0:
            vals, counts = np.unique(gfc_natural_null, return_counts=True)
            changeyear = int(vals[np.argmax(counts)])
            condition_mask = natural_null & (gfc_array == changeyear)
            change_array = np.where(condition_mask, gfc_array, change_array)

    # Case 2: Both == 100 → manmade, extract all change
    manmade_all = (year_start == 100) & (year_end == 100)
    if manmade_all.any():
        change_min = np.nanmin(gfc_array[manmade_all])
        change_max = np.nanmax(gfc_array[manmade_all])
        condition_mask = manmade_all & (gfc_array >= change_min) & (gfc_array <= change_max)
        change_array = np.where(condition_mask, 100 + gfc_array, change_array)

    # Case 3: Both > 100 → manmade, specific year range
    manmade_years = (year_start > 100) & (year_end > 100)
    if manmade_years.any():
        year_start_adj = year_start - 100
        year_end_adj = year_end - 100
        condition_mask = manmade_years & (gfc_array >= year_start_adj) & (gfc_array <= year_end_adj)
        change_array = np.where(condition_mask, gfc_array, change_array)

    # Case 4: Natural change with specific year range
    natural_years = ~natural_null & ~manmade_all & ~manmade_years
    if natural_years.any():
        condition_mask = natural_years & (gfc_array >= year_start) & (gfc_array <= year_end)
        change_array = np.where(condition_mask, 100 + gfc_array, change_array)

    # Write to output band
    output_band.WriteArray(change_array, x, y)
    
    change_array = None

def mask_gfc_by_block(training_raster_path, quadrant_path, output_dir):
    # Load gfc quadrant raster
    gfc_raster, gfc_info = get_raster_info(quadrant_path)
    gfc_band = gfc_raster.GetRasterBand(1)
    
    # Load training raster
    training_raster, _ = get_raster_info(training_raster_path)
    year_start_band = training_raster.GetRasterBand(1)
    year_end_band = training_raster.GetRasterBand(2)
    
    # Set block size
    x_block_size = 256
    y_block_size = 160
    
    # Create new raster
    output_tiff = os.path.join(output_dir, "output_gfc.tif")
    output = gdal.GetDriverByName("GTiff").Create(output_tiff, gfc_info["xsize"], gfc_info["ysize"], 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=100"])
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(255)
    output.SetGeoTransform(gfc_info["transform"])
    output.SetProjection(gfc_info["projection"].ExportToWkt())
    
    # Mask gfc
    total_blocks = (gfc_info["xsize"] // x_block_size + 1) * (gfc_info["ysize"] // y_block_size + 1)
    with Progress(SpinnerColumn(),
              "[progress.description]{task.description}",
              MofNCompleteColumn(),
              TimeElapsedColumn(),
              transient=True) as progress:
        task = progress.add_task("Masking GFC...", total=total_blocks)
        for y in range(0, gfc_info["ysize"] + 1, y_block_size):
            rows = min(y_block_size, gfc_info["ysize"] - y)  # Handles edge case for remaining rows
            for x in range(0, gfc_info["xsize"] + 1, x_block_size):
                cols = min(x_block_size, gfc_info["xsize"] - x)  # Handles edge case for remaining cols
                mask_gfc(gfc_band, year_start_band, year_end_band, output_band, x, y, cols, rows)
                progress.update(task, advance=1)
    
    console.print("\u2713 Masked GFC", style="dim green")
    output_band = None
    output = None
    gfc_raster = None
    training_raster = None
    
    return output_tiff

# =========================
# Extract GFC change
# =========================
def extract_training(training_shp, gfc_quadrants_dir, output_dir):
    for quadrant in sorted(set(os.listdir(gfc_quadrants_dir))):
        # Step 1: Rasterize training_shp to bounds of current quadrant
        quadrant_path = os.path.join(gfc_quadrants_dir, quadrant)
        _, quadrant_info = get_raster_info(quadrant_path)
        output_tiff = os.path.join(output_dir, f"{quadrant}_training.tif")
        training_raster = rasterize(training_shp, output_tiff, quadrant_info)
    
        # # Step 2: Mask GFC by block
        # output_raster = mask_gfc_by_block(training_raster, quadrant_path, output_dir)
    
        break
    
# =========================
# Create the training data
# =========================
def main():
    console.rule("[bold cyan]CREATING GFC TRAINING", align="left")

    # Load in training data files
    output_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/deg_tiles_output_training_v2"
    training_shp = "/gpfs/glad1/Theo/Shapefiles/GFC/gfc_training.shp"
    gfc_quadrants_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_quadrants"
    os.makedirs(output_dir, exist_ok=True)

    # Extract change
    extract_training(training_shp, gfc_quadrants_dir, output_dir)
    
    console.print(f"\n\u2713 All tiles written to {output_dir}\n", style="bold green")
    
if __name__ == "__main__":
    main()