# Theo Kerr

# Imports/env settings
from get_planet_tiles import get_planet_tiles
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr
from tqdm import tqdm
import os, shutil, math
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

def crop_raster(raster_path, output_folder, cutline, current_feature):
    """Uses the GDAL Warp function to reproject the given raster to given crs and pixel size, and crop to the given cutline. 

    Args:
        raster_path (str): path to a raster dataset.
        output_folder (str): folder for the output dataset.
        crs (str): string for a crs, in the format "EPSG:3857" for example.
        pixel_size (float): desired pixel size, in destination crs units.
        cutline (str): path to a GeoJSON cutline file.

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
        gdal.Warp(dst_ds, 
                  raster_path, 
                  format="GTiff", 
                  cutlineDSName=cutline, 
                  cropToCutline=True, 
                  warpOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
        
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
            change_array = np.where(condition_mask, gfc_array, 0)
            
        else:
            condition_mask = (gfc_array >= year_start) & (gfc_array <= year_end)
            change_array = np.where(condition_mask, gfc_array, 0)

    else:
        raise ValueError("Changetype must be either 'natural' or 'manmade'")
    
    return change_array

def split_raster(raster_path, planet_tiles, planet_tile_list, crs):
    # Set up variables
    tiles_to_keep = []
    output_dir = os.path.dirname(raster_path)
    feature_name = f"{os.path.splitext(raster_path)[0]}"
    progress_bar = tqdm(total=len(planet_tile_list), desc="Splitting progress", unit="tile", leave=False)
    
    # Loop through planet tiles
    for tile in planet_tile_list:
        # Build warp options
        current_tile = f'"location" = \'{tile}\''
        warp_options = gdal.WarpOptions(format="GTiff", 
                                        dstSRS=crs,
                                        cutlineDSName=planet_tiles,
                                        cutlineWhere=current_tile,
                                        cropToCutline=True,
                                        warpOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
        
        # Warp the raster for current tile
        dst_ds = os.path.join(output_dir, f"{feature_name}_{tile}_training.tif")
        gdal.Warp(dst_ds, raster_path, options=warp_options)
        
        progress_bar.update(1)
        
        # Remove blank raster
        ds = gdal.Open(dst_ds)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        ds = None
        
        blank_condition =  np.isin(array, [0, 255])
        if np.all(blank_condition):
            os.remove(dst_ds)
            continue
        
        # Add tile to list
        tiles_to_keep.append(tile)

    # Sort list and remove duplicates before returning
    tiles_to_keep = sorted(set(tiles_to_keep))
        
    progress_bar.close()
    
    return tiles_to_keep


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
    # Create list for planet tiles
    tile_list = []
    
    # Create progress bar
    print()
    progress_bar = tqdm(total=len(training_shp), desc=f"Progress for {changetype}", unit="feature")

    # Extract raster training for change
    for polygon in training_shp.itertuples(index=True):
        # Write out the polygon for future rasterizing
        current_feature = f"feature{polygon[0]}"
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
        
        # Get planet tiles
        planet_tiles_path = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
        tiles = get_planet_tiles(output_tiff)
        
        # Split the output raster into tiles
        tiles = split_raster(output_tiff, planet_tiles_path, tiles, gfc_srs)
        tile_list.extend(tiles)
        
        # Remove large output raster after tiling
        os.remove(output_tiff)
        
        progress_bar.update(1)
        
    progress_bar.close() 

    # Sort the tile list and remove duplicates before returning it
    tile_list = sorted(set(tile_list))
    
    return tile_list

# =========================
# Create the training data
# =========================
def main():
    columns = shutil.get_terminal_size().columns
    print()
    print("=".center(columns, "="))
    print("CREATING GFC TRAINING")

    # Load in training data files
    output_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/Output_training_v2"
    natural_training = gpd.read_file("/gpfs/glad1/Theo/Data/Global_Forest_Change/Natural_change/Natural_change.shp")
    manmade_training = gpd.read_file("/gpfs/glad1/Theo/Data/Global_Forest_Change/Manmade_change/Manmade_change.shp")
    gfc_tiles_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_tiles"

    # Extract natural change
    natural_output_dir = os.path.join(output_dir, "natural_training")
    os.makedirs(natural_output_dir, exist_ok=True)
    natural_tiles = extract_training(natural_training, natural_output_dir, gfc_tiles_dir, changetype="natural")

    # Extract manmade change
    manmade_output_dir = os.path.join(output_dir, "manmade_training")
    os.makedirs(manmade_output_dir, exist_ok=True)
    manmade_tiles = extract_training(manmade_training, manmade_output_dir, gfc_tiles_dir, changetype="manmade")

    # Write out planet tile lists
    natural_tiles_txt = os.path.join(output_dir, "natural_planet_tiles.txt")
    manmade_tiles_txt = os.path.join(output_dir, "manmade_planet_tiles.txt")
    
    with open(natural_tiles_txt, "w") as f:
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in natural_tiles)
    
    with open(manmade_tiles_txt, "w") as f:
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in manmade_tiles)
        
    print()
    print("=".center(columns, "="))
    print()
    
if __name__ == "__main__":
    main()