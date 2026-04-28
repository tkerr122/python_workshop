import networkx as nx
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
import os

# Load and concatenate all tile files
print("loading tiles...")
input_dir = "/gpfs/glad1/Theo/Data/Pastures_test/test_output/block_polygons"
output_file = "/gpfs/glad1/Theo/Data/Pastures_test/test_output/merged_polygons.fgb"
SNAP_TOLERANCE = 1e-8

tiles = [
    os.path.join(input_dir, file)
    for file in os.listdir(input_dir)
    if file.endswith(".fgb")
]
gdfs = [gpd.read_file(t) for t in tiles]
gdf = pd.concat(gdfs, ignore_index=True)
gdf = gpd.GeoDataFrame(gdf, crs=gdfs[0].crs)

# Fix floating point gaps at tile seams
print("removing gaps...")
gdf["geometry"] = gdf.geometry.buffer(SNAP_TOLERANCE).buffer(-SNAP_TOLERANCE)
gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

# Build adjacency graph
G = nx.Graph()
G.add_nodes_from(range(len(gdf)))

print("getting touching polygons")
tree = gdf.sindex
for i, geom in enumerate(gdf.geometry):
    candidates = list(tree.query(geom, predicate="touches"))
    for j in candidates:
        if i != j:
            G.add_edge(i, j)

# Dissolve each connected component into one polygon
print("merging polygons...")
merged = []
for component in nx.connected_components(G):
    idx = list(component)
    subset = gdf.iloc[idx]
    merged_geom = unary_union(subset.geometry)
    merged.append(
        {
            "geometry": merged_geom,
            "block_id": ",".join(subset["block_id"].astype(str).unique()),
        }
    )

result = gpd.GeoDataFrame(merged, columns=["geometry", "block_id"], crs=gdf.crs)
result = result.explode(index_parts=False).reset_index(drop=True)
result.to_file(output_file, driver="FlatGeobuf")
print("done")
