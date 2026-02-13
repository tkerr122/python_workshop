# Theo Kerr on 12/02/2025

# Global imports/env settings
from osgeo import gdal
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
    
def load_files(input_dir):
    # Loop through input directory and create list of files
    files = os.listdir(input_dir)
    filepaths = []
    
    for file in files:
        if file.lower().endswith(".geojson"):
            filepath = os.path.join(input_dir, file)
            filepaths.append(filepath)
            
    filepaths = sorted(filepaths)
        
    return filepaths
    
def create_training(input_file, output_dir, crs, pixel_size, buffer_size):
    # Define output tiff name
    output_tiff = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.tif")
    
    # Read in the training
    training = gpd.read_file(input_file)
    training = training.to_crs(crs)
    training_temp = f"{os.path.splitext(output_tiff)[0]}.geojson"
    
    # Buffer the training
    training_buffered = training.buffer(buffer_size, cap_style="flat")
    training_buffered.to_file(training_temp, driver="GeoJSON")
    
    # Rasterize the training
    rasterize(training_temp, output_tiff, pixel_size, burn_value=1)
        
    # Tile the training
    tile_list = tile_training(output_tiff, output_dir, crs)
    
    # Cleanup
    os.remove(training_temp)
    os.remove(output_tiff)
    
    return tile_list
    

# =========================
# Create the training data
# =========================
def main():
    columns = shutil.get_terminal_size().columns
    print()
    print("=".center(columns, "="))
    print("CREATING PASTURES TRAINING")
    print()
    
    # Set up variables
    crs = "EPSG:3857"
    pixel_size = 4.77731426716
    buffer_size = 10
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
    antoine_pastures = load_files("/gpfs/glad1/Exch/Antoine/Pastures_training/linear_features/Antoine_pastures")
    theo_pastures = load_files("/gpfs/glad1/Theo/Shapefiles/Pastures_training")
    
    # Create the training
    planet_tile_list = []
    for item in antoine_pastures:
        tile_list = create_training(item, current_output_dir, crs, pixel_size, buffer_size)
        planet_tile_list.extend(tile_list)
        print()
    
    for item in theo_pastures:
        tile_list = create_training(item, current_output_dir, crs, pixel_size, buffer_size)
        planet_tile_list.extend(tile_list)
        print()
    
    # Write out planet tile list
    planet_tile_list = sorted(set(planet_tile_list))
    tiles_txt = os.path.join(current_output_dir, "planet_tiles.txt")
    with open(tiles_txt, "w") as f:
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in planet_tile_list)
        
    print()
    print("=".center(columns, "="))
    print()
    
if __name__ == "__main__":
    main()