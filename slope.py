# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
import os, shutil
gdal.UseExceptions()

def create_slope(dem_path, output_dir):
    # Create slope raster
    name = os.path.splitext(os.path.basename(dem_path))[0]
    slope_path = os.path.join(output_dir, f"{name}_slope.tif")
    slope_options = gdal.DEMProcessingOptions(format="GTiff",
                                              computeEdges=True,
                                              slopeFormat="degree")
    gdal.DEMProcessing(slope_path, dem_path, processing="slope", options=slope_options)
    
def merge_slope(slope_dir):
    # Create filenames, build VRT
    slope_rasters = [os.path.join(slope_dir, f) for f in os.listdir(slope_dir)]
    vrt_path = f"{slope_dir}.vrt"
    print(f"Building VRT for {os.path.basename(slope_dir)}:")
    gdal.BuildVRT(vrt_path, slope_rasters, callback=gdal.TermProgress_nocb)
    
    # Translate VRT into GeoTIFF
    merged_path = f"{slope_dir}.tif"
    print(f"Translating VRT to {merged_path}:")
    gdal.Translate(merged_path, vrt_path, format="GTiff", creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"], callback=gdal.TermProgress_nocb)
    
    # Cleanup
    shutil.rmtree(slope_dir)
    os.remove(vrt_path)
    
def main():
    # Setup
    print("\nCreating slope rasters...")
    
    # Set up variables    
    dem_path = "/gpfs/glad1/Theo/Data/Lidar/DTMs/AZ_BlackRock_DTM"
    slope_dir = "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope/AZ_BlackRock_slope"
    os.makedirs(slope_dir, exist_ok=True)
        
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
            create_slope(path, slope_dir)
            
            progress_bar.update(1)
            
        progress_bar.close()
    
        # Merge the slope rasters
        merge_slope(slope_dir)
    
    else:
        create_slope(dem_path, slope_dir)    
        merge_slope(slope_dir)
    
if __name__ == "__main__":
    main()