# Theo Kerr

# Global imports/env settings
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import os, shutil, argparse
gdal.UseExceptions()

# Local imports
from get_planet_tiles import get_planet_tiles

"""This script is a command-line utility to split a given raster or folder of rasters
into tiles matching the planet tile scheme.
================================================
-p option: path to the raster or folder of rasters to be processed
-od option: path to the output directory for the tiles
-crs: crs for the output tiled rasters
"""

# ========================
# Tile training
# ========================
def tile_training(raster_path, output_dir, crs="EPSG:3857"):
    # Set up variables
    tiles_to_keep = []
    
    # Get planet tiles
    planet_tile_list = get_planet_tiles(raster_path)
    planet_tiles = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
    
    # Loop through planet tiles
    progress_bar = tqdm(total=len(planet_tile_list), desc="Splitting progress", unit="tile")
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
        dst_ds = os.path.join(output_dir, f"{tile}.tif")
        
        gdal.Warp(dst_ds, raster_path, options=warp_options)
        
        progress_bar.update(1)
        
        # Remove blank raster
        ds = gdal.Open(dst_ds)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        ds = None
        band = None
        
        blank_condition =  np.isin(array, [0, 255])
        if np.all(blank_condition):
            os.remove(dst_ds)
            continue
        
        # Add tile to list
        tiles_to_keep.append(tile)

    # Sort list and remove duplicates before returning
    tiles_to_keep = sorted(set(tiles_to_keep))
    
    # Cleanup
    progress_bar.close()
    
    return tiles_to_keep

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for tiling a raster or folder of rasters")
    parser.add_argument("-p", "--training-path", type=str, help="Path to raster(s)", required=True)
    parser.add_argument("-od", "--output-dir", type=str, help="Path to output folder", required=True)
    parser.add_argument("-crs", type=int, help="EPSG code for desired crs", default=3857)
    
    # Parse args
    args = parser.parse_args()
    
    # Set up variables    
    training_path = args.training_path
    output_dir = args.output_dir
    crs = f"EPSG:{args.crs}"
    planet_tile_list = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Start message
    columns = shutil.get_terminal_size().columns
    print()
    print("=".center(columns, "="))
    print(f"TILING TRAINING FOR {os.path.basename(training_path)}")
    
    # Tile training in folder/file
    if os.path.isdir(training_path):
        # Get training
        rasters = os.listdir(training_path)
        
        # Tile each raster
        for raster in rasters:
            # Set up path
            path = os.path.join(training_path, raster)
            
            # Tile raster
            tile_list = tile_training(path, output_dir, crs)
            planet_tile_list.extend(tile_list)

        # Write out planet tiles list
        tiles_txt = os.path.join(output_dir, "planet_tiles.txt")
        planet_tile_list = sorted(set(planet_tile_list))
        
        with open(tiles_txt, "w") as f:
            f.write("location\n")
            f.writelines(f"{tile}\n"for tile in planet_tile_list)
    
    else:
        tile_list = tile_training(training_path, output_dir, crs)
        tiles_txt = os.path.join(output_dir, "planet_tiles.txt")
        
        with open(tiles_txt, "w") as f:
            f.write("location\n")
            f.writelines(f"{tile}\n"for tile in planet_tile_list)
            
    print()
    print("=".center(columns, "="))
    print()

if __name__ == "__main__":
    main()