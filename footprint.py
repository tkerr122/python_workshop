# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import os, argparse, shutil
gdal.UseExceptions()

"""This script is a command-line utility to create footprints for a folder of rasters
================================================
-p option: path to the folder of rasters to be processed
-od option: path to the output directory
"""

def get_footprint(raster_path, output_dir):
    # Get footprint
    name = os.path.splitext(os.path.basename(raster_path))[0]
    footprint_path = os.path.join(output_dir, f"{name}_footprint.geojson")
    gdal.Footprint(footprint_path, raster_path, format="GeoJSON", dstSRS="EPSG:3857")

def merge_footprints(footprint_dir, output_file):
    gdfs = []
    
    # Load all footprints
    for file in os.listdir(footprint_dir):
        filepath = os.path.join(footprint_dir, file)
        gdf = gpd.read_file(filepath)
        gdf["filename"] = file.rsplit("_footprint.geojson")[0]
        gdfs.append(gdf)
        
    # Write footprints to file
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    merged.to_file(output_file, driver='GPKG')
    
def main():
    # Setup
    print("\nGetting footprints...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for getting the footprints for a folder of rasters")
    parser.add_argument("-p", "--raster-dir", type=str, required=True, help="Path to folder of rasters to be processed")
    parser.add_argument("-od", "--output-dir", type=str, required=True, help="Path to output footprint folder")

    # Parse arguments
    args = parser.parse_args()
    
    # Set up variables
    raster_dir = args.raster_dir
    output_dir = args.output_dir
    temp = os.path.join(output_dir, "temp")
    os.makedirs(temp, exist_ok=True)
        
    # Get footprints
    rasters = os.listdir(raster_dir)
    progress_bar = tqdm(total=len(rasters), desc="Progress", unit="Raster")
    for raster in rasters:
        raster_path = os.path.join(raster_dir, raster)
        get_footprint(raster_path, temp)
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Merge footprints
    output_file = os.path.join(output_dir, f"{os.path.basename(raster_dir)}_footprints.gpkg")
    merge_footprints(temp, output_file)
    
    # Remove footprints dir
    shutil.rmtree(output_dir)
        
if __name__ == "__main__":
    main()