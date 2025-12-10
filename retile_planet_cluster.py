# Theo Kerr on 8/26/2024


# Imports
import pandas as pd
import geopandas as gpd
import os

# Build function to split a dictionary into chunks, to speed up processing
def split_dict(dictionary, batch_size):
    keys = list(dictionary.keys())
    for i in range(0, len(keys), batch_size):
        yield {k: dictionary[k] for k in keys[i:i + batch_size]}

# Build function to retile planet tiles to specified number of tiles
def retile_planet(input_layer: str, output_path: str, dimensions: int, batch_size: int) -> None:
    # Load input layer
    planet_tiles = gpd.read_file(input_layer)
    planet_tiles = planet_tiles.sort_values(by="location")
    
    # Create a set to track processed tiles
    processed_tiles = set()

    # Create a dictionary to quickly look up tiles by their location
    tile_dict = {tile['location']: tile['geometry'] for index, tile in planet_tiles.iterrows()}
    
    # Batch process tiles, if they haven't been processed yet, combine with neighboring E, SE, and S tiles
    batch_counter = 0 
    for batch in split_dict(tile_dict, batch_size):
        # Set up dictionary for retiles
        retiles = {}
                
        for loc, geom in batch.items():
            # Set up location variables
            lon = int(loc[0:4])
            lat = int(loc[6:10])
            lon_str = loc[4:6]
            lat_str = loc[10]
            
            # Find tiles to process
            tiles = []
            for x in range(0, dimensions):
                for y in range(0, dimensions):
                    tile_loc = f"{(lon + x):04}{lon_str}{(lat + y):04}{lat_str}"
                    tiles.append(tile_loc)
        
            # Determine if the tile has been processed or not
            if all(loc not in processed_tiles for loc in tiles):
                # Create a list to store the retile
                retile_list = []
                                                
                # Append tiles tiles to retile list if they exist
                for tile in tiles:
                    if tile in tile_dict:
                        tile_buffer = tile_dict[tile].buffer(100, cap_style='flat')
                        retile_list.append(tile_buffer)
                        
                # Merge all the tiles into a retile
                retile = gpd.GeoSeries(retile_list, crs="EPSG:3857").unary_union
                retile_buffer = retile.buffer(-100, cap_style='flat')
                
                # Convert to geodataframe, append to retiles gdf
                retiles[loc] = retile_buffer
                
                # Append the tiles to processed tiles list
                processed_tiles.update(tiles)
                
        # After creating new tiles for the batch, convert to geodataframe and save temp output to disk
        locations = list(retiles.keys())
        geometry = list(retiles.values())
        retiles_gdf = gpd.GeoDataFrame({'location': locations, 'geometry': geometry}, crs="EPSG:3857")
        temp_output = os.path.join(output_path, f"output_batch_{batch_counter}.geojson")
        retiles_gdf.to_file(temp_output)
        
        batch_counter += 1
        print(f"Batch {batch_counter} done...")
        
    # Merge the temporary outputs into 1 shapefile
    batch_files = [os.path.join(output_path, f"output_batch_{i}.geojson") for i in range(batch_counter)]
    batch_gdfs = [gpd.read_file(file) for file in batch_files]
    final_gdf = gpd.GeoDataFrame(pd.concat(batch_gdfs, ignore_index=True), crs="EPSG:3857")
    
    # Save the final output to disk
    final_gdf.to_file(os.path.join(output_path, "final_output.shp"))
    
    # Remove temp files
    for file in batch_files:
        os.remove(file)

# Test the function
input_layer = r"/gpfs/glad1/Exch/USA_2022/Data/shp_planet/planet_tiles_global.shp"
output_path = r"/gpfs/glad1/Theo/Data/Planet_retile_output"
retile_planet(input_layer, output_path, 4, 5000)
