# Theo Kerr on 12/02/2025

# Imports/env settings
from osgeo import gdal, ogr
import geopandas as gpd
import os, shutil
gdal.UseExceptions()


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
    
def get_planet_tiles(raster_path, planet_tiles_path):
    # Get raster info
    ds = gdal.Open(raster_path)
    projection = ds.GetSpatialRef().ExportToWkt()
    
    # Get raster footprint
    footprint_path = f"{os.path.splitext(raster_path)[0]}_footprint.geojson"
    cutline_path = f"{os.path.splitext(raster_path)[0]}_cutline.geojson"
    
    # Get raster footprint
    gdal.Footprint(footprint_path, raster_path, format="GeoJSON", dstSRS=projection)
    
    # Intersect the footprint and planet tiles
    translate_options = gdal.VectorTranslateOptions(format="GeoJSON", clipSrc=footprint_path, selectFields=["location"])
    gdal.VectorTranslate(cutline_path, planet_tiles_path, options=translate_options)
    
    # Return the tiles
    tiles_ds = ogr.Open(cutline_path)
    tiles_layer = tiles_ds.GetLayer(0)
    tiles = [tile.GetField("location") for tile in tiles_layer]
    
    # Cleanup
    os.remove(footprint_path)
    os.remove(cutline_path)
    
    return tiles

def split_raster(raster_path, planet_tiles, planet_tile_list, crs, pixel_size):
    output_dir = f"{os.path.splitext(raster_path)[0]}_split"
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through planet tiles
    print(f"Splitting {os.path.basename(raster_path)}...")
    for tile in planet_tile_list:
        # Build warp options
        current_tile = f'"location" = \'{tile}\''
        warp_options = gdal.WarpOptions(format="GTiff", 
                                        dstSRS=crs,
                                        xRes=pixel_size,
                                        yRes=pixel_size,
                                        cutlineDSName=planet_tiles,
                                        cutlineWhere=current_tile,
                                        cropToCutline=True,
                                        warpOptions=["COMPRESS=LZW", "BIGTIFF=YES"],
                                        callback=gdal.TermProgress_nocb)
        
        # Warp the raster for current tile
        dst_ds = os.path.join(output_dir, f"{tile}_training.tiff")
        gdal.Warp(dst_ds, raster_path, options=warp_options)

    
# =========================
# Rasterize training
# =========================
def extract_training(training_shp, output_tiff, crs, pixel_size, txt_path, buffer_size):
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
    
    # Get planet tiles
    planet_tiles = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
    tiles = get_planet_tiles(output_tiff, planet_tiles)
    
    # Write out planet tile list
    with open(txt_path, "w") as f:
        tiles = sorted(set(tiles))
        f.write("location\n")
        f.writelines(f"{tile}\n"for tile in tiles)
        
    # Split training raster into tiles
    split_raster(output_tiff, planet_tiles, tiles, crs, pixel_size)
        
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
    output_dir = "/gpfs/glad1/Exch/Antoine/Pastures_training/Output_training/test"
    os.makedirs(output_dir, exist_ok=True)
    pasture_path = "/gpfs/glad1/Exch/Antoine/Pastures_training/working/pasture_roads/pasture_roads.shp"
    
    # Extract pastures training
    pasture_txt = os.path.join(output_dir, "Pasture_planet_tiles.txt")
    pasture_tiff = os.path.join(output_dir, "Pasture.tif")
    extract_training(pasture_path, pasture_tiff, crs, pixel_size, pasture_txt, buffer_size=10)
    
    print()
    print("=".center(columns, "="))
    print()
    
if __name__ == "__main__":
    main()