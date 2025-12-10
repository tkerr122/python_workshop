# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
import os, argparse
gdal.UseExceptions()

def create_slope(dem_path, output_dir):
    # Create slope raster
    name = os.path.splitext(os.path.basename(dem_path))[0]
    slope_path = os.path.join(output_dir, f"{name}_slope.tif")
    slope_options = gdal.DEMProcessingOptions(format="GTiff",
                                              computeEdges=True,
                                              slopeFormat="degree")
    gdal.DEMProcessing(slope_path, dem_path, processing="slope", options=slope_options)
    
def main():
    # Setup
    print("\nCreating slope rasters...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script to create slope rasters from a DEM or folder of DEMs")
    parser.add_argument("-p", "--dem-path", type=str, help="Path to DEM or DEM folder")
    parser.add_argument("-od", "--output-dir", type=str, help="Path to output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up variables    
    dem_path = args.dem_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
        
    # Loop through DEM folder and create slope rasters
    if os.path.isdir(dem_path):
        dems = os.listdir(dem_path)
        
        # Create progress bar
        print()
        progress_bar = tqdm(total=len(dems), desc="Progress", unit="DEM")
        
        # Create slope rasters
        for dem in dems:
            # Set up path
            path = os.path.join(dem_path, dem)
            
            # Create slope
            create_slope(path, output_dir)
            
            progress_bar.update(1)
            
        progress_bar.close()
    
    else:
        create_slope(dem_path, output_dir)    
    
if __name__ == "__main__":
    main()