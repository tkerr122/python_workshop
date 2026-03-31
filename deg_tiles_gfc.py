# Theo Kerr

# Imports/env settings
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import os, shutil, math
console = Console()
gdal.UseExceptions()


# =========================
# Define custom functions
# =========================
class InvalidChangeYears(Exception):
    pass

def get_raster_info(raster_path):
    """Opens a raster at the given path.

    Args:
        raster_path (str): path to a raster dataset.

    Returns:
        tuple: returns a GDAL raster, number of columns, number of rows, the geotransform, and the projection.
    """
    ds = gdal.Open(raster_path)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    transform = ds.GetGeoTransform()
    projection = ds.GetSpatialRef()
    
    return ds, xsize, ysize, transform, projection

def crop_raster(raster_path, output_folder, cutline, current_feature=None):
    """Uses the GDAL Warp function to crop a raster to the given cutline. 

    Args:
        raster_path (str): path to a raster dataset.
        output_folder (str): folder for the output dataset.
        cutline (str): path to a GeoJSON cutline file.
        current_feature (str): name for output file.
        Defaults to None.

    Returns:
        str: path to output warped raster.
    """
    # Set warp options
    if isinstance(raster_path, list) and len(raster_path) != 1:
        raster_basename = os.path.splitext(os.path.basename(raster_path[0]))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_{current_feature}_cropped_merged.tif"
    elif isinstance(raster_path, list) and len(raster_path) == 1:
        raster_basename = os.path.splitext(os.path.basename(raster_path[0]))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_{current_feature}_cropped.tif"
    else: 
        raster_basename = os.path.splitext(os.path.basename(raster_path))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_{current_feature}_cropped.tif"

    # Crop the raster, if it hasn't already been done
    if os.path.isfile(dst_ds) == False:
        warp_options = gdal.WarpOptions(format="GTiff", 
                                        cutlineDSName=cutline, 
                                        cropToCutline=True,
                                        multithread=True,
                                        warpMemoryLimit=2000,
                                        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=100"],
                                        warpOptions=["NUM_THREADS=100"])
        gdal.Warp(dst_ds, raster_path, options=warp_options)
    else:
        console.print(f"\u2192 '{os.path.basename(dst_ds)}' exists, path saved", style="dim yellow", highlight=False)
        
    return dst_ds

def get_tiles(x_min, x_max, y_min, y_max):
    """Takes given max and mins and finds which GFC tiles they intersect with, returns a list of the tiles.

    Args:
        x_min (float): Minimum x value
        x_max (float): Maximum x value
        y_min (float): Minimum y value
        y_max (float): Maximum y value

    Returns:
        list: list of tiles
    """
    tile_names = []
    
    lat_start = math.ceil(y_max / 10) * 10 # Northmost tile
    lat_end = math.ceil(y_min / 10) * 10 # Southmost tile
    
    lon_start = math.floor(x_min / 10) * 10 # Westmost tile (because longitude in western hemisphere is negative)
    lon_end = math.floor(x_max / 10) * 10 # Eastmost tile
    
    for lat in range(lat_start, lat_end - 1, -10): # Negative step to move downwards from maximum latitude
        for lon in range(lon_start, lon_end + 1, 10): # Because longitude is already negative, no need for negative step
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
    
            lat_str = f"{abs(lat):02d}{lat_dir}"
            lon_str = f"{abs(lon):03d}{lon_dir}"
            
            tile_name = f"Hansen_GFC-2024-v1.12_lossyear_{lat_str}_{lon_str}.tif"
            tile_names.append(tile_name)
            
    tile_names = sorted(set(tile_names))
    
    return tile_names

def mask_gfc(gfc_array, changetype, year_start=None, year_end=None):
    """Takes in an array for GFC and depending on user input either extracts the change based on the most frequent pixel 
    value for year of the change, or uses year_start and year_end to extract values in that range.

    Args:
        gfc_array (np.array): array for GFC
        year_start (int, optional): 2 digit start year for the change. Defaults to None.
        year_end (int, optional): 2 digit end year for the change. Defaults to None.

    Returns:
        np.array: array with GFC change
    """
    if changetype == "natural":
        if np.isnan(year_start) and np.isnan(year_end):
            vals, counts = np.unique(gfc_array[gfc_array != 0], return_counts=True)
            mode = np.argwhere(counts == np.max(counts))
            changeyear = vals[mode].flatten().tolist()
            changeyear = int(changeyear[0])
            
            condition_mask = (gfc_array == changeyear)
            change_array = np.where(condition_mask, gfc_array, 0)
            
        else:
            condition_mask = (gfc_array >= year_start) & (gfc_array <= year_end)
            change_array = np.where(condition_mask, gfc_array, 0)
            
    elif changetype == "manmade":
        if np.isnan(year_start) and np.isnan(year_end):
            change_min = np.nanmin(gfc_array)
            change_max = np.nanmax(gfc_array)
            
            condition_mask = (gfc_array >= change_min) & (gfc_array <= change_max)
            change_array = np.where(condition_mask, 100 + gfc_array, 0)
            
        else:
            condition_mask = (gfc_array >= year_start) & (gfc_array <= year_end)
            change_array = np.where(condition_mask, 100 + gfc_array, 0)

    else:
        raise ValueError("Changetype must be either 'natural' or 'manmade'")
    
    return change_array

def split_by_quadrant(gfc_quadrant_dir, training_dir, output_dir):
    """Opens large gfc raster and gets bounding box to use as a cutline to warp
    & merge the training rasters.

    Args:
        gfc_quadrant_dir (str): path to large gfc raster directory.
        training_dir (str): path to training raster directory.
        output_dir (str): path to output directory.
    """
    quadrants = os.listdir(gfc_quadrant_dir)
    console.print(f"Splitting training into {len(quadrants)} quadrants...", style="dim orange")
    for quadrant in quadrants:
        # Get quadrant footprint
        quadrant_path = os.path.join(gfc_quadrant_dir, quadrant)
        ds = gdal.Open(quadrant_path)
        gt = ds.GetGeoTransform()
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + gt[1] * ds.RasterXSize
        ymin = ymax + gt[5] * ds.RasterYSize
        footprint = f"POLYGON (({xmin} {ymin}, {xmin} {ymax}, {xmax} {ymax}, {xmax} {ymin}, {xmin} {ymin}))"
        
        ds = None
        
        # Write out footprint
        footprint_path = os.path.join(output_dir, "footprint.geojson")
        footprint_geom = ogr.CreateGeometryFromWkt(footprint)
        footprint_ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(footprint_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        footprint_layer = footprint_ds.CreateLayer("footprint", srs, geom_type=ogr.wkbPolygon)
        feature = ogr.Feature(footprint_layer.GetLayerDefn())
        feature.SetGeometry(footprint_geom)
        footprint_layer.CreateFeature(feature)
        
        feature = None
        footprint_layer = None
        footprint_ds = None
        
        # Warp the training files to the quadrant
        training = [os.path.join(training_dir, f) for f in os.listdir(training_dir)]
        dst_ds = os.path.join(output_dir, f"{os.path.splitext(quadrant)[0]}_training.tif")
        warp_options = gdal.WarpOptions(format="GTiff", 
                                        cutlineDSName=footprint_path, 
                                        cropToCutline=True,
                                        multithread=True,
                                        warpMemoryLimit=2000,
                                        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=100"],
                                        warpOptions=["NUM_THREADS=100"],
                                        callback=gdal.TermProgress_nocb)
        gdal.Warp(dst_ds, training, options=warp_options)
        
        # Remove temp footprint
        os.remove(footprint_path)
        
    console.print(f"\u2713 Split training into {len(quadrants)} quadrants", style="dim green")
            

# =========================
# Extract GFC change
# =========================
def extract_training(training_shp, output_dir, gfc_tiles_dir, changetype):
    """Takes input training shapefile and gfc tiles and extracts the change present in the shapefile.

    Args:
        training_shp (str): path to change shapefile
        output_dir (str): path to output directory
        gfc_tiles_dir (str): path to gfc tile directory
        changetype (str): either 'natural' or 'manmade' change
    """
    # Extract raster training for change
    with Progress(SpinnerColumn(),
                "[progress.description]{task.description}",
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True) as progress:
        task = progress.add_task(f"Extracting {changetype} training", total=len(training_shp))    
                
        for polygon in training_shp.itertuples(index=True):
            # Write out the polygon for future rasterizing
            current_feature = f"{changetype}_feature{polygon[0]}"
            output_path = os.path.join(output_dir, f"{current_feature}_temp_geojson.geojson")
            year_start = polygon.year_start
            year_end = polygon.year_end
            
            # Check for errors
            if np.isnan(year_start) != np.isnan(year_end):
                raise InvalidChangeYears(f"Change years for feature{current_feature} must both be Null, or both integers. \nYear_start: {year_start}\nYear_end: {year_end}")
            
            gpd.GeoDataFrame([{"geometry": polygon.geometry}], 
                            crs=training_shp.crs).to_file(output_path, driver="GeoJSON")
            
            # Open the feature and get extents
            feature = ogr.Open(output_path)
            layer = feature.GetLayer()
            x_min, x_max, y_min, y_max = layer.GetExtent()

            # Find tiles
            tile_names = get_tiles(x_min, x_max, y_min, y_max)
            gfc_tile_path = []
            for tile in tile_names:
                tile_path = os.path.join(gfc_tiles_dir, tile)
                
                if os.path.isfile(tile_path) == False:
                    print(f"\n\"{tile}\" doesn't exist, skipping")
                    continue
                
                gfc_tile_path.append(tile_path)
            
            # Warp tiles to current feature
            gfc_cropped_path = crop_raster(gfc_tile_path, output_dir, output_path, current_feature)
            
            # Read in gfc array, extract change array
            gfc_cropped, gfc_xsize, gfc_ysize, gfc_geotransform, gfc_srs = get_raster_info(gfc_cropped_path)
            gfc_8bit = gfc_cropped.GetRasterBand(1).ReadAsArray(0, 0, gfc_xsize, gfc_ysize).astype(np.uint8)
            gfc_change = mask_gfc(gfc_8bit, changetype, year_start, year_end)
            
            gfc_cropped = None
            gfc_8bit = None
            
            # Create blank output raster
            output_tiff = os.path.join(output_dir, f"{current_feature}.tif")
            output = gdal.GetDriverByName("GTiff").Create(output_tiff, gfc_xsize, gfc_ysize, 1, gdal.GDT_Byte, 
                                                        options=["COMPRESS=LZW", "BIGTIFF=YES"])
            output_band = output.GetRasterBand(1)
            output_band.SetNoDataValue(255)
            output.SetGeoTransform(gfc_geotransform)
            output.SetProjection(gfc_srs.ExportToWkt())
            
            # Write out gfc change array, cleanup
            output_band.WriteArray(gfc_change)
            
            gfc_change = None
            output_band = None
            output = None
            
            os.remove(output_path)
            os.remove(gfc_cropped_path)
            
            progress.update(task, advance=1)
        
    console.print(f"\u2713 Extracted {len(training_shp)} features for {changetype} training", style="dim green")

# =========================
# Create the training data
# =========================
def main():
    console.print("\nCREATING GFC TRAINING", style="bold cyan")

    # Load in training data files
    output_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/deg_tiles_output_training"
    training_dir = os.path.join(output_dir, "training_rasters")
    natural_training = gpd.read_file("/gpfs/glad1/Theo/Data/Global_Forest_Change/Natural_change/Natural_change.shp")
    manmade_training = gpd.read_file("/gpfs/glad1/Theo/Data/Global_Forest_Change/Manmade_change/Manmade_change.shp")
    gfc_tiles_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_tiles"
    os.makedirs(training_dir, exist_ok=True)

    # # Extract natural and manmade change
    # extract_training(natural_training, training_dir, gfc_tiles_dir, changetype="natural")
    # extract_training(manmade_training, training_dir, gfc_tiles_dir, changetype="manmade")
    
    # Split into quadrants
    gfc_quadrant_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_quadrants"
    split_by_quadrant(gfc_quadrant_dir, training_dir, output_dir)
    
    console.print(f"\n\u2713 All tiles written to {output_dir}\n", style="bold green")
    
if __name__ == "__main__":
    main()