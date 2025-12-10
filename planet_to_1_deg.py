# Theo Kerr on 9/3/2024

# Imports
import geopandas as gpd
import os

def planet_to_deg(planet_tiles_path: str, deg_tiles_path: str, output_path: str, columns_to_retain: list) -> None:
    """Takes input planet tiles and 1 degree tiles shapefiles and creates a new shapefile at planet tile resolution with both locations."""
    
    # Load shapefiles
    planet_tiles = gpd.read_file(planet_tiles_path)
    deg_tiles = gpd.read_file(deg_tiles_path)
    
    # Ensure CRSs match
    if deg_tiles.crs != planet_tiles.crs:
        deg_tiles = deg_tiles.to_crs(planet_tiles.crs)
    
    # Perform a spatial join to find intersecting tiles
    merged_gdf = gpd.sjoin(planet_tiles, deg_tiles, how='inner', predicate='intersects')
    
    # Simplify GeoDataFrame and keep necessary columns
    merged_gdf = merged_gdf[columns_to_retain]
    
    # Save the result
    output_file = os.path.join(output_path, "Planet_tiles_and_degree.shp")
    merged_gdf.to_file(output_file)
    print(f"Saved output to {output_file}")

# Test function
planet_tiles = r"/gpfs/glad1/Exch/USA_2022/Data/shp_planet/planet_tiles_global.shp"
deg_tiles = r"/gpfs/glad1/Exch/Andres_2023/shp_osm/shp_all_tiles/all_processed_tiles.shp"
output_path = r"/gpfs/glad1/Theo/Data/test_Planet_and_1_degree"
columns = ['location', 'TILE', 'geometry']
planet_to_deg(planet_tiles, deg_tiles, output_path, columns)