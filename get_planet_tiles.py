# Theo Kerr on 11/13/2025
# For use on linux cluster "gdalenv" conda env

# Imports/env settings 
from osgeo import gdal, ogr
from tqdm import tqdm
import argparse, os
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
    footprint_path = f"{os.path.splitext(raster_path)[0]}_footprint.geojson"
    tiles_path = f"{os.path.splitext(raster_path)[0]}_tiles_split.geojson"
    
    # Get raster footprint
    gdal.Footprint(footprint_path, raster_path, format="GeoJSON", dstSRS="EPSG:3857")
    
    # Intersect the footprint and planet tiles
    planet_tiles_path = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
    translate_options = gdal.VectorTranslateOptions(format="GeoJSON", 
                                                    clipSrc=footprint_path, 
                                                    selectFields=["location"])
    gdal.VectorTranslate(tiles_path, planet_tiles_path, options=translate_options)
    
    # Return the tiles
    tiles_ds = ogr.Open(tiles_path)
    tiles_layer = tiles_ds.GetLayer(0)
    tiles = [tile.GetField("location") for tile in tiles_layer]
    tiles = sorted(set(tiles))
    
    # Cleanup
    os.remove(footprint_path)
    os.remove(tiles_path)
    
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

    # Loop through the folder and get the planet tiles for each CHM
    if os.path.isdir(raster_path):
        chms = os.listdir(raster_path)
        
        # Create progress bar
        print()
        progress_bar = tqdm(total=len(chms), desc="Progress", unit="CHM")
        
        # Get tiles
        tiles_list = []
        for chm in chms:
            # Set up path
            chm_path = os.path.join(raster_path, chm)
            
            # Get planet tiles
            tiles = get_planet_tiles(chm_path)
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