import geopandas as gpd
from shapely.geometry import Polygon
import os

# Creat tile name functions
def format_coord(degree, is_lat):
    direction = ""
    abs_degree = abs(int(degree))
    if is_lat:
        direction = "N" if degree >= 0 else "S"
        return f"{abs_degree}{direction}"
    else:
        direction = "E" if degree >= 0 else "W"
        return f"{abs_degree:03d}{direction}"
    
def generate_tile_name(lat, lon):
    return f"{format_coord(lat, is_lat=True)}_{format_coord(lon, is_lat=False)}"

# Create 10 degree grid layer
grid_polygons = []
tile_names = []
grid_cell_size = 10
for lon in range(-180, 181, grid_cell_size):  
    for lat in range(-90, 91, grid_cell_size): 
        polygon = Polygon([(lon, lat), 
                           (lon + grid_cell_size, lat), 
                           (lon + grid_cell_size, lat + grid_cell_size), 
                           (lon, lat + grid_cell_size)])
        grid_polygons.append(polygon)
        top_left_lat = lat + grid_cell_size
        tile_names.append(generate_tile_name(top_left_lat, lon))
        
gdf = gpd.GeoDataFrame({'tile_name': tile_names, 'geometry': grid_polygons}, crs="EPSG:4326")
gdf.to_file("/gpfs/glad1/Theo/Data/10_deg_tiles/10_deg_tiles.shp")

# Merge 10 degree layer with planet tiles
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

# Planet to 10 degree
planet_tiles = "/gpfs/glad1/Exch/USA_2022/Data/shp_planet/planet_tiles_global.shp"
deg_tiles = "/gpfs/glad1/Theo/Data/10_deg_tiles/10_deg_tiles.shp"
output_path = "/gpfs/glad1/Theo/Data/Planet_and_10_degree"
columns = ['location', 'tile_name', 'geometry']
planet_to_deg(planet_tiles, deg_tiles, output_path, columns)

# Export to csv
gdf = gpd.read_file("/gpfs/glad1/Theo/Data/Planet_and_10_degree/Planet_tiles_and_degree.shp")
gdf = gdf[['location', 'tile_name']]

gdf.to_csv("/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Planet_tile_list/Planet_tile_list.csv", index=False)
