# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import grey_dilation
import numpy as np
import os, shutil
gdal.UseExceptions()

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
    
    return merged_path

def dilate_slope(merged_slope_path):
    # Load slope raster
    print("Smoothing slope...")
    
    ds, xsize, ysize, transform, projection = get_raster_info(merged_slope_path)
    band = ds.GetRasterBand(1)
    slope_array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    
    # Apply dilation
    smoothed_slope = grey_dilation(slope_array, size=(3,3))
    
    # Create blank output raster
    output_tiff = f"{os.path.splitext(merged_slope_path)[0]}_smoothed.tif"
    output = gdal.GetDriverByName("GTiff").Create(output_tiff, xsize, ysize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)
    output.SetGeoTransform(transform)
    output.SetProjection(projection.ExportToWkt())
    
    # Write out smoothed slope raster
    output_band.WriteArray(smoothed_slope)
    
    # Cleanup
    os.remove(merged_slope_path)
    os.rename(output_tiff, merged_slope_path)
    smoothed_slope = None
    output_band = None
    output = None
    
    
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
        merged_path = merge_slope(slope_dir)
        
        # Smnooth the slope raster
        dilate_slope(merged_path)
    
    else:
        create_slope(dem_path, slope_dir)    
        merged_path = merge_slope(slope_dir)
        dilate_slope(merged_path)
    
    print("Done")
    
if __name__ == "__main__":
    main()