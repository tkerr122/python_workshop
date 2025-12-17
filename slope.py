# Imports/env settings 
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import grey_dilation
import os, shutil, argparse
gdal.UseExceptions()

"""I have written this script to be a command-line utility for creating a single merged slope raster
from a DEM, or folder of DEMs. 
====================================================================================================
-p option: path to DEM, or folder of DEMs.
-od option: path to output folder for the slope raster
"""


# =========================
# Define custom functions
# =========================
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
    """Creates a slope raster from a given DEM.

    Args:
        dem_path (str): Path to DEM.
        output_dir (str): Path to output directory.
    
    Returns:
        str: Path to the slope raster
    """
    # Create slope raster
    name = os.path.splitext(os.path.basename(dem_path))[0]
    slope_path = os.path.join(output_dir, f"{name}_slope.tif")
    slope_options = gdal.DEMProcessingOptions(format="GTiff",
                                              computeEdges=True,
                                              slopeFormat="degree")
    gdal.DEMProcessing(slope_path, dem_path, processing="slope", options=slope_options)
    
    return slope_path

def dilate_slope(slope_path, output_dir):
    """Uses the SciPy Grey Dilation tool to emphasize high slope values in the given slope raster

    Args:
        slope_path (str): Path to slope raster to dilate.
    """
    # Load slope raster
    ds, xsize, ysize, transform, projection = get_raster_info(slope_path)
    band = ds.GetRasterBand(1)
    slope_array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    
    # Apply dilation
    smoothed_slope = grey_dilation(slope_array, size=(3,3))
    
    # Create blank output raster
    output_tiff = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(slope_path))[0]}_smoothed.tif")
    output = gdal.GetDriverByName("GTiff").Create(output_tiff, xsize, ysize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)
    output.SetGeoTransform(transform)
    output.SetProjection(projection.ExportToWkt())
    
    # Write out smoothed slope raster
    output_band.WriteArray(smoothed_slope)
    
    # Cleanup
    smoothed_slope = None
    output_band = None
    output = None
    
def merge_slope(slope_dir, output_tiff):
    """Uses GDAL BuildVRT and GDAL Translate tools to merge all slope rasters in a given folder.

    Args:
        slope_dir (str): Path to slope folder to merge.

    Returns:
        str: Path to merged slope raster.
    """
    # Create filenames, build VRT
    slope_rasters = [os.path.join(slope_dir, f) for f in os.listdir(slope_dir)]
    vrt_path = f"{os.path.dirname(slope_dir)}.vrt"
    print(f"Building VRT for {os.path.basename(slope_dir)}:")
    gdal.BuildVRT(vrt_path, slope_rasters, callback=gdal.TermProgress_nocb)
    
    # Translate VRT into GeoTIFF
    print(f"Translating VRT to {output_tiff}:")
    gdal.Translate(output_tiff, vrt_path, format="GTiff", creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"], callback=gdal.TermProgress_nocb)
    
    # Cleanup
    os.remove(vrt_path)
    

# =========================
# Create slope
# =========================
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for creating slope raster from folder of DTMs")
    parser.add_argument("-p", "--dem-path", type=str, help="Path to DEM or folder of DEMs", required=True)
    parser.add_argument("-od", "--output-dir", type=str, help="Path to output slope folder", required=True)
    
    # Parse args
    args = parser.parse_args()
    
    # Set up variables    
    dem_path = args.dem_path
    slope_dir = args.output_dir
    os.makedirs(slope_dir, exist_ok=True)
    
    # Start message
    print(f"\nCREATING SLOPE RASTERS FOR {os.path.basename(dem_path)}")
    
    # Loop through DEM folder and create slope rasters
    if os.path.isdir(dem_path):
        # Create subfolders
        raw_slope_dir = os.path.join(slope_dir, "raw_slope")
        dilated_slope_dir = os.path.join(slope_dir, "dilated_slope")
        os.makedirs(raw_slope_dir, exist_ok=True)
        os.makedirs(dilated_slope_dir, exist_ok=True)
        
        # Get dems
        dems = os.listdir(dem_path)
        
        # Create progress bar
        print()
        progress_bar = tqdm(total=len(dems), desc="Progress", unit="DEM")
        
        # Create slope rasters
        for dem in dems:
            # Set up path
            path = os.path.join(dem_path, dem)
            
            # Create slope
            slope_path = create_slope(path, raw_slope_dir)
            
            # Dilate slope
            dilate_slope(slope_path, dilated_slope_dir)
            
            progress_bar.update(1)
            
        progress_bar.close()
    
        # Merge the slope rasters
        merge_slope(dilated_slope_dir, f"{slope_dir}.tif")
        
        # Cleanup
        shutil.rmtree(slope_dir)
    
    else:
        # Create slope
        slope_path = create_slope(dem_path, slope_dir)
        dilate_slope(slope_path, slope_dir)
    
    print("Done")
    
if __name__ == "__main__":
    main()