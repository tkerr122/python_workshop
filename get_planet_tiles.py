# Theo Kerr on 11/13/2025
# For use on linux cluster "gdalenv" conda env

# Imports/env settings 
from osgeo import gdal, ogr
from tqdm import tqdm
import argparse, os, time
gdal.UseExceptions()

"""This script is a command-line utility to find which planet tiles a given raster or folder of rasters
intersect with.
================================================
-p option: path to the raster or folder of rasters to be processed
-t option: path to the output TXT file which will have the planet tiles
"""

# =========================
# Define custom functions
# =========================
def get_planet_tiles(raster_path):
    # Get raster footprint
    footprint_ds = gdal.Footprint("", raster_path, format="Memory", dstSRS="EPSG:3857")
    footprint_layer = footprint_ds.GetLayer(0)
    footprint_feature = footprint_layer.GetNextFeature()
    footprint_geom = footprint_feature.GetGeometryRef().Clone()
        
    # Open planet tiles and filter by footprint
    planet_tiles_path = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
    tiles_ds = ogr.Open(planet_tiles_path)
    tiles_layer = tiles_ds.GetLayer(0)
    tiles_layer.SetSpatialFilter(footprint_geom)
    
    # Return the tiles
    tiles = sorted(set(f.GetField("location") for f in tiles_layer))

    return tiles

def main():
    # Setup
    print("\nGetting planet tiles...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for getting the planet tiles of a raster")
    parser.add_argument("-p", "--raster-path", type=str, help="Path to raster or folder of rasters to be processed")
    parser.add_argument("-t", "--txt-path", type=str, help="Path to txt for output planet tiles")

    # Parse arguments
    args = parser.parse_args()

    # Set up variables
    raster_path = args.raster_path
    txt_path = args.txt_path

    # Loop through the folder and get the planet tiles for each raster
    if os.path.isdir(raster_path):
        rasters = os.listdir(raster_path)
        
        # Create progress bar
        print()
        progress_bar = tqdm(total=len(rasters), desc="Progress", unit="raster")
        
        # Get tiles
        tiles_list = []
        for raster in rasters:
            # Set up path
            full_path = os.path.join(raster_path, raster)
            
            # Get planet tiles
            tiles = get_planet_tiles(full_path)
            tiles_list.extend(tiles)
            
            progress_bar.update(1)
            
        progress_bar.close()
    
    else:
        tiles_list = get_planet_tiles(raster_path)
        
    # Write to txt file
    with open(txt_path, "w") as f:
        tiles_list = sorted(set(tiles_list))
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in tiles_list)
    
    print("done")
    
if __name__ == "__main__":
    main()