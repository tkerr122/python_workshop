# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import os
gdal.UseExceptions()

def get_footprint(raster_path, output_dir):
    # Get footprint
    name = os.path.splitext(os.path.basename(raster_path))[0]
    footprint_path = os.path.join(output_dir, f"{name}_footprint.geojson")
    gdal.Footprint(footprint_path, raster_path, format="GeoJSON", dstSRS="EPSG:3857")

def merge_footprints(footprint_dir, output_dir):
    gdfs = []
    
    # Load all footprints
    for file in os.listdir(footprint_dir):
        filepath = os.path.join(footprint_dir, file)
        gdf = gpd.read_file(filepath)
        gdf["filename"] = file
        gdfs.append(gdf)
        
    # Write footprints to file
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    merged_path = os.path.join(output_dir, "footprints.gpkg")
    merged.to_file(merged_path, driver='GPKG')
    
def main():
    # Set up filepaths
    input_chm_dir = "/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/AZ_BlackRock_unfiltered_CHM"
    footprint_dir = "/gpfs/glad1/Theo/Data/Lidar/AZ_BlackRock_footprints"
    output_dir = "/gpfs/glad1/Theo/Data/Lidar"
    os.makedirs(footprint_dir, exist_ok=True)
        
    # Get footprints
    rasters = os.listdir(input_chm_dir)
    progress_bar = tqdm(total=len(rasters), desc="Progress", unit="Raster")
    for raster in rasters:
        raster_path = os.path.join(input_chm_dir, raster)
        get_footprint(raster_path, footprint_dir)
        
        progress_bar.update(1)
        
    progress_bar.close()
    
    # Merge footprints
    merge_footprints(footprint_dir, output_dir)
    
    # Remove footprints dir
    os.rmdir(footprint_dir)
        
if __name__ == "__main__":
    main()