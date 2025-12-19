# Theo Kerr on 12/02/2025

# Global imports/env settings
from osgeo import gdal, ogr
import geopandas as gpd
import os, shutil
gdal.UseExceptions()

# Local imports
from tile_training import tile_training

# =========================
# Define custom functions
# =========================
def rasterize(input_file, output_tiff, pixel_size, burn_value):
    # Open dataset
    dataset = ogr.Open(input_file)
    layer = dataset.GetLayer() 
    
    # Define raster properties
    x_min, x_max, y_min, y_max = layer.GetExtent()
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName("GTiff").Create(output_tiff, x_res, y_res, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    
    # Set projection
    srs = layer.GetSpatialRef()
    target_ds.SetProjection(srs.ExportToWkt())
    
    # Set band nodata value
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(255)

    # Rasterize dataset
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[burn_value])
    
    # Fill nodata
    arr = band.ReadAsArray()
    arr = (arr == 1).astype("uint8")
    band.WriteArray(arr)
    
    print(f"Rasterized {os.path.basename(input_file)}...")
    
    band = None
    target_ds = None
    dataset = None

    
# =========================
# Extract the training
# =========================
def extract_training(training_shp, output_tiff, crs, pixel_size, buffer_size):
    # Read in the training
    training = gpd.read_file(training_shp)
    training = training.to_crs(crs)
    training_temp = f"{os.path.splitext(output_tiff)[0]}.geojson"
    
    # Buffer the training
    training_buffered = training.buffer(buffer_size, cap_style="square")
    
    # Reproject
    training_buffered.to_file(training_temp, driver="GeoJSON")
    
    # Rasterize the training
    rasterize(training_temp, output_tiff, pixel_size, burn_value=1)
        
    # Cleanup
    os.remove(training_temp)
    

# =========================
# Create the training data
# =========================
def main():
    columns = shutil.get_terminal_size().columns
    print()
    print("=".center(columns, "="))
    print("CREATING PASTURES TRAINING")
    
    # Set up variables
    crs = "EPSG:3857"
    pixel_size = 4.77731426716
    
    # Load in training data files
    output_dir = "/gpfs/glad1/Exch/Antoine/Pastures_training/Output_training_v2"
    os.makedirs(output_dir, exist_ok=True)
    pasture_path = "/gpfs/glad1/Exch/Antoine/Pastures_training/linear_features/linear_features_065W_13S_px512_r4_c4.shp"
    
    # Extract pastures training
    pasture_tiff = os.path.join(output_dir, "Pasture.tif")
    extract_training(pasture_path, pasture_tiff, crs, pixel_size, buffer_size=10)
    
    # Tile the training
    output_tiled = os.path.join(output_dir, "tiled")
    os.makedirs(output_tiled, exist_ok=True)
    planet_tile_list = tile_training(pasture_tiff, output_tiled, crs)
    
    # Write out planet tile list
    tiles_txt = os.path.join(output_dir, "planet_tiles.txt")
    
    with open(tiles_txt, "w") as f:
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in planet_tile_list)

    print()
    print("=".center(columns, "="))
    print()
    
if __name__ == "__main__":
    main()