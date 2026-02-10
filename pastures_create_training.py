# Theo Kerr on 12/02/2025

# Global imports/env settings
from osgeo import gdal, ogr
from datetime import datetime
import geopandas as gpd
import os, shutil
gdal.UseExceptions()

# Local imports
from tile_training import tile_training

# =========================
# Define custom functions
# =========================
def rasterize(input_file, output_tiff, pixel_size, burn_value):
    # Rasterize the input file
    rasterize_options = gdal.RasterizeOptions(format="GTiff",
                                              burnValues=burn_value,
                                              xRes=pixel_size,
                                              yRes=pixel_size)
    gdal.Rasterize(output_tiff, input_file, options=rasterize_options)
    
    print(f"Rasterized {os.path.basename(input_file)}...")
    
    
# =========================
# Extract the training
# =========================
def extract_training(training_shp, output_tiff, crs, pixel_size, buffer_size):
    # Read in the training
    training = gpd.read_file(training_shp)
    training = training.to_crs(crs)
    training_temp = f"{os.path.splitext(output_tiff)[0]}.geojson"
    
    # Buffer the training
    training_buffered = training.buffer(buffer_size, cap_style="flat")
    
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
    output_dir = "/gpfs/glad1/Exch/Antoine/Pastures_training/Output_training"
    
    # Move older folders to "Previous versions" folder
    for file in os.listdir(output_dir):
        path = os.path.join(output_dir, file)
        
        if os.path.isdir(path) and file != "Previous_versions":
            shutil.move(path, os.path.join(output_dir, "Previous_versions"))
            print(f"Moved {file} to \"Previous_versions\"...")
    
    # Load in training data files
    date = datetime.now().strftime("%b%d_%Y").lower()
    current_output_dir = os.path.join(output_dir, f"Training_{date}")
    os.makedirs(current_output_dir, exist_ok=True)
    # antoine_pastures = "/gpfs/glad1/Exch/Antoine/Pastures_training/linear_features/linear_features_01082026/linear_features_01082026.shp"
    theo_pastures = "/gpfs/glad1/Theo/Shapefiles/Pastures_training/Pastures.shp"
    
    # Extract pastures training
    # antoine_pasture_tiff = os.path.join(current_output_dir, "Antoine_pasture.tif")
    theo_pasture_tiff = os.path.join(current_output_dir, "Theo_pasture.tif")
    # extract_training(antoine_pastures, antoine_pasture_tiff, crs, pixel_size, buffer_size=10)
    extract_training(theo_pastures, theo_pasture_tiff, crs, pixel_size, buffer_size=10)
    
    # Tile the training
    output_tiled = os.path.join(current_output_dir, "tiled")
    os.makedirs(output_tiled, exist_ok=True)
    # antoine_tile_list = tile_training(antoine_pasture_tiff, output_tiled, crs)
    theo_tile_list = tile_training(theo_pasture_tiff, output_tiled, crs)
    
    # Write out planet tile list
    planet_tile_list = []
    # planet_tile_list.extend(antoine_tile_list)
    planet_tile_list.extend(theo_tile_list)
    planet_tile_list = sorted(set(planet_tile_list))
    
    tiles_txt = os.path.join(current_output_dir, "planet_tiles.txt")
    
    with open(tiles_txt, "w") as f:
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in planet_tile_list)
        
    # Cleanup
    # os.remove(antoine_pasture_tiff)
    os.remove(theo_pasture_tiff)

    print()
    print("=".center(columns, "="))
    print()
    
if __name__ == "__main__":
    main()