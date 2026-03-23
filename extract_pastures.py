# Imports
from osgeo import gdal, osr
from scipy import ndimage
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely, skimage, os, argparse
gdal.UseExceptions()
console = Console()

"""
I have written this script to be a command-line utility for extracting areas
enclosed by linear features.
===============================================================================
-p option: path to linear features input raster
-od option: output directory
-sk option: whether or not to write out skeleton raster
-t option: probability threshold of line features
-cr option: pixels to consider for initial gap bridging
-ma option: minimum component size (in pixels) to retain
-gt option: gap tolerance for closing gaps
-at option: angle tolerance for closing gaps
-max option: maximum area for extracted plots
"""

# Stage 1: Load the linear features
def load_linear_features(linear_features_path: str, threshold: int = 50) -> tuple:
    """
    Loads in a raster of linear features with pixel values from 0-100
    probability. Keeps pixels above the threshold and returns a numpy 
    array, and the geostransform & projection. 

    Args:
        linear_features_path (str): path to linear features raster
        threshold (int, optional): threshold value for masking. 
        Defaults to 50.

    Returns:
        tuple: returns the thresholded features array, along with the 
        geotransform and projection from the original raster for later use.
    """
    with console.status("Loading and thresholding linear features...", spinner="dots"):
        # Load the linear features
        linear_features = gdal.Open(linear_features_path)
        xsize = linear_features.RasterXSize
        ysize = linear_features.RasterYSize
        transform = linear_features.GetGeoTransform()
        projection = linear_features.GetSpatialRef()
        
        linear_features_array = linear_features.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize).astype(np.uint8)
        
        # Select features above the threshold
        condition_mask = (linear_features_array > threshold)
        thresholded_linear_features = np.where(condition_mask, 1, 0)
    
    console.print("\u2713 Loaded and thresholded linear features", style="dim green")
    
    return thresholded_linear_features, transform, projection

# Stage 2: Morphologically clean
def morphological_clean(linear_features_array: np.ndarray, closing_radius: int = 2, min_area: int = 50) -> np.ndarray:
    """
    Uses scipy's ndimage to bridge pixel gaps up to the closing radius value, 
    and remove small areas.

    Args:
        linear_features_array (np.ndarray): numpy array of linear features 
        created in the previous step.
        closing_radius (int, optional): pixels to consider for initial
        gap bridging. Defaults to 2.
        min_area (int, optional): component size (in pixels) to retain. 
        Defaults to 50.

    Returns:
        np.ndarray: cleaned linear features array.
    """
    with console.status("Removing single pixel gaps and small areas...", spinner="dots"):
        # Remove single pixel gaps
        structure = ndimage.generate_binary_structure(2, 1)
        structure = ndimage.iterate_structure(structure, closing_radius)
        closed_gaps = ndimage.binary_closing(linear_features_array, structure=structure)
        
        # Remove small areas
        labeled, _ = ndimage.label(closed_gaps)
        component_sizes = np.bincount(labeled.ravel())
        to_keep = component_sizes >= min_area
        to_keep[0] = False
        cleaned = to_keep[labeled]
                
    console.print("\u2713 Removed single pixel gaps and small areas", style="dim green")
    
    return cleaned.astype(np.uint8)

# Stage 3: Skeletonize
def skeletonize(cleaned_array: np.ndarray, transform: tuple, projection: osr.SpatialReference, output_dir: str = None, rasterize: bool = False) -> np.ndarray:
    """
    Uses sklearn's skeletonize function on the linear features array and 
    returns the single-pixel-wide centerlines. Also writes them to a 
    raster in the output dir if specified.

    Args:
        cleaned_array (np.ndarray): numpy array of linear features.
        transform (tuple): the geotransform tuple of the original raster.
        projection(osr.SpatialReference): the projection of the original raster.
        output_dir (str, optional): path to output directory, if desired.
        Defaults to None.
        rasterize (bool, optional): whether or not to write out a skeleton raster.
        Defaults to False.

    Returns:
        np.ndarray: centerline array.
    """
    with console.status("Skeletonizing linear features...", spinner="dots"):
        # Skeletonize
        skeleton = skimage.morphology.skeletonize(cleaned_array)
        
        # Write output raster if specified
        if rasterize:
            if output_dir is None:
                raise ValueError("Output directory must be provided if rasterize=True")

            output_tiff = os.path.join(output_dir, "skeleton.tif")
            ysize, xsize = cleaned_array.shape
            output = gdal.GetDriverByName("GTiff").Create(
                output_tiff, xsize, ysize, 1, gdal.GDT_Byte, 
                options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]
                )
            output_band = output.GetRasterBand(1)
            output_band.SetNoDataValue(255)
            output.SetGeoTransform(transform)
            output.SetProjection(projection.ExportToWkt())
            output_band.WriteArray(skeleton)
            
            output_band = None
            output = None
    
    console.print("\u2713 Skeletonized linear features", style="dim green")
    
    return skeleton.astype(np.uint8)

# Stage 4: Vectorize
def vectorize(skeleton_array: np.ndarray, transform: tuple, projection: osr.SpatialReference) -> gpd.GeoDataFrame:
    """Visits all nodes created from the skeleton array and vectorizes them.

    Args:
        skeleton_array (np.ndarray): numpy array of skeletonized features
        transform (tuple): the geotransform tuple of the original raster
        projection (osr.SpatialReference): the projection of the original raster
        output_dir (str): path to the output directory, if desired.
        Defaults to None.

    Returns:
        gpd.GeoDataFrame: vectorized lines gdf
    """
    # Get all skeleton pixel coordinates
    rows, cols = np.where(skeleton_array)
    skeleton_set = set(zip(rows.tolist(), cols.tolist()))

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True) as progress:
        task = progress.add_task("Vectorizing skeleton...", total=len(skeleton_set))
        lines = []

        for r, c in skeleton_set:
            for dr, dc in [
                (0, 1),   # E
                (1, 0),   # S
                (1, 1),   # SE
                (1, -1),  # SW
            ]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in skeleton_set:
                    x1 = transform[0] + c * transform[1]
                    y1 = transform[3] + r * transform[5]
                    x2 = transform[0] + nc * transform[1]
                    y2 = transform[3] + nr * transform[5]
                    lines.append(shapely.LineString([(x1, y1), (x2, y2)]))
            progress.update(task, advance=1)

        crs = projection.GetAuthorityCode(None)
        gdf = gpd.GeoDataFrame(geometry=lines, crs=f"EPSG:{crs}")

    console.print("\u2713 Vectorized skeleton", style="dim green")

    return gdf

# Stage 5: Close gaps 
def close_gaps(gdf: gpd.GeoDataFrame, gap_tolerance: float = 10.0, angle_tolerance: float = 30.0) -> gpd.GeoDataFrame:
    """Find all dangling endpoints of the lines and search within the
    distance of "gap_tolerance" to close them. Also uses angle tolerance
    to only close gaps between lines that are plausibly connected (no
    sharp angle connections.)

    Args:
        gdf (gpd.GeoDataFrame): lines gdf returned by vectorize function.
        gap_tolerance (float, optional): distance between lines segments 
        to close. Defaults to 10.0.
        angle_tolerance (float, optional): angle threshold beyond which 
        gap will not be closed. Defaults to 30.0.

    Returns:
        gpd.GeoDataFrame: lines gdf with gaps closed.
    """
    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True) as progress:
        task = progress.add_task("Preparing to close gaps...", total=None) 
        
        # Extract all endpoints and identify dangling ones
        endpoint_counts = {}
        endpoint_to_line = {}

        for idx, line in enumerate(gdf.geometry):
            for point in [line.coords[0], line.coords[-1]]:
                if point not in endpoint_counts:
                    endpoint_counts[point] = 0
                    endpoint_to_line[point] = idx
                endpoint_counts[point] += 1

        # Dangling endpoints appear only once
        dangling = [pt for pt, count in endpoint_counts.items() if count == 1]
        dangling_geoms = [shapely.Point(pt) for pt in dangling]

        # Build a GeoDataFrame of dangling endpoints for spatial indexing 
        dangling_gdf = gpd.GeoDataFrame(
            {"coords": dangling, "line_idx": [endpoint_to_line[pt] for pt in dangling]},
            geometry=dangling_geoms,
            crs=gdf.crs
        )
        
        # Calculate bearing at each dangling endpoint 
        progress.update(task, description="Finding bearings...", total=len(dangling))

        def endpoint_bearing(line, at_start: bool, lookback: int = 5) -> float:
            coords = list(line.coords)
            # Clamp lookback to available coordinates
            lookback = min(lookback, len(coords) - 1)
            if at_start:
                p1, p2 = coords[lookback], coords[0]
            else:
                p1, p2 = coords[-(lookback + 1)], coords[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return np.degrees(np.arctan2(dx, dy)) % 360
        
        bearings = []
        for pt in dangling:
            line_idx = endpoint_to_line[pt]
            line = gdf.geometry.iloc[line_idx]
            at_start = (line.coords[0] == pt)
            bearings.append(endpoint_bearing(line, at_start))
            
            progress.update(task, advance=1)
        
        # Add bearings to the gdf
        dangling_gdf["bearing"] = bearings

        # Find candidate pairs and bridge valid gaps 
        sindex = dangling_gdf.sindex
        visited_pairs = set()
        bridging_lines = []
        
        # Close gaps
        progress.update(task, description="Closing gaps...", total=len(dangling_gdf), completed=0)

        for i, row in dangling_gdf.iterrows():
            # Query spatial index for candidates within gap tolerance
            buffer = row.geometry.buffer(gap_tolerance)
            candidates = dangling_gdf.iloc[sindex.query(buffer)]

            for j, candidate in candidates.iterrows():
                # Skip identical points
                if i == j:
                    continue
                
                # Avoid processing the same pair twice
                pair = tuple(sorted((i, j)))
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                # Skip endpoints belonging to the same line
                if row["line_idx"] == candidate["line_idx"]:
                    continue

                # Check bearing similarity
                angle_diff = abs(row["bearing"] - candidate["bearing"]) % 360
                angle_diff = min(angle_diff, 360 - angle_diff)
                is_similar = angle_diff <= angle_tolerance
                is_opposite = abs(angle_diff - 180) <= angle_tolerance
                if not (is_similar or is_opposite):
                    continue
                
                # Synthesize bridging line
                bridging_lines.append(shapely.LineString([row.geometry, candidate.geometry]))

            progress.update(task, advance=1)
            
        # Update progress bar
        progress.update(task, description="Writing GeoDataFrame...", total=None)
            
        # Append bridging lines to the GeoDataFrame 
        if bridging_lines:
            bridging_gdf = gpd.GeoDataFrame(geometry=bridging_lines, crs=gdf.crs)
            gdf = pd.concat([gdf, bridging_gdf], ignore_index=True)

    console.print(f"\u2713 Closed {len(bridging_lines)} gaps", style="dim green")

    return gdf

# Stage 6: Merge and export
def merge_and_export(gdf: gpd.GeoDataFrame, output_dir: str, transform: tuple) -> gpd.GeoDataFrame:
    """Takes the input gdf and merges the geometries into 1 geometry, also 
    simplifying to remove "staircase" effect from vectorizing.
    Finally, exports the gdf to a GeoPackage in the output directory.

    Args:
        gdf (gpd.GeoDataFrame): gdf of lines with gaps closed.
        output_dir (str): path to the output directory.
        transform (tuple): the geotransform of the original raster.

    Returns:
        gpd.GeoDataFrame: simplified lines gdf.
    """
    with console.status("Merging and exporting...", spinner="dots"):
        # Merge connected LineStrings
        merged = shapely.line_merge(shapely.union_all(gdf.geometry.values))
        merged = merged.simplify(tolerance=transform[1])
        if merged.geom_type == "MultiLineString":
            geometries = list(merged.geoms)
        else:
            geometries = [merged]
        merged_gdf = gpd.GeoDataFrame(geometry=geometries, crs=gdf.crs)

        # Export to GeoPackage
        gpkg_path = os.path.join(output_dir, "linear_features.gpkg")
        merged_gdf.to_file(gpkg_path, driver="GPKG")

    console.print("\u2713 Merged and exported", style="dim green")
    
    return merged_gdf

# Step 7: Polygonize
def polygonize(gdf: gpd.GeoDataFrame, output_dir: str, max_area: float = None) -> None:
    """Extracts polygons enclosed by the lines gdf up to the "max_area" (if 
    specified) and writes them to a GeoPackage in the output directory.

    Args:
        gdf (gpd.GeoDataFrame): merged lines gdf.
        output_dir (str): path to the output directory
        max_area (float, optional): maximum area for the polygons.
        Defaults to None.
    """
    with console.status("Polygonizing linear features...", spinner="dots"):
        # Node the lines at all intersections before polygonizing
        noded = shapely.union_all(gdf.geometry.values)
        
        # Extract enclosed polygons
        polygons = list(shapely.ops.polygonize(noded))
        if not polygons:
            console.print("No enclosed polygons found", style="dim yellow")
            return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

        polygon_gdf = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)

        # Calculate area in square meters
        polygon_gdf["area_m2"] = polygon_gdf.geometry.area

        # Filter out artifact polygons above max area if specified
        if max_area is not None:
            polygon_gdf = polygon_gdf[polygon_gdf["area_m2"] <= max_area].reset_index(drop=True)

        # Export to GeoPackage
        gpkg_path = os.path.join(output_dir, "plots.gpkg")
        polygon_gdf.to_file(gpkg_path, driver="GPKG")

    console.print(f"\u2713 Found {len(polygon_gdf)} enclosed polygons", style="dim green")

# Main function
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script to find areas enclosed by linear features")
    parser.add_argument("-p", "--linear-features-path", type=str, help="Path to linear features input raster", required=True)
    parser.add_argument("-od", "--output-dir", type=str, help="Output directory", required=True)
    parser.add_argument("-sk", "--sk-raster", action="store_true", help="Whether or not to write out skeleton raster")
    parser.add_argument("-t", "--probability-threshold", type=int, default=50, help="Probability threshold of line features")
    parser.add_argument("-cr", "--closing-radius", type=int, default=2, help="Pixels to consider for initial gap bridging")
    parser.add_argument("-ma", "--min-area", type=int, default=50, help="Component size (in pixels) to retain")
    parser.add_argument("-gt", "--gap-tolerance", type=float, default=10.0, help="Gap tolerance for closing gaps")
    parser.add_argument("-at", "--angle-tolerance", type=float, default=30.0, help="Angle tolerance for closing gaps")
    parser.add_argument("-max", "--max-area", type=float, default=None, help="Maximum area for extracted plots")
    
    # Parse arguments, set up variables
    args = parser.parse_args()
    linear_features_path = args.linear_features_path
    output_dir = args.output_dir
    sk_raster = args.sk_raster
    probability_threshold = args.probability_threshold
    closing_radius = args.closing_radius
    min_area = args.min_area
    gap_tolerance = args.gap_tolerance
    angle_tolerance = args.angle_tolerance
    max_area = args.max_area
    
    os.makedirs(output_dir, exist_ok=True)

    # Start message
    console.print(f"\nEXTRACTING PASTURES FROM {os.path.basename(linear_features_path)}", style="bold cyan")
    
    # Stage 1: Load linear features
    thresholded_linear_features, transform, projection = load_linear_features(linear_features_path, probability_threshold)
    
    # Stage 2: Morphological cleaning
    cleaned_array = morphological_clean(thresholded_linear_features, closing_radius, min_area)
    
    # Stage 3: Skeletonize
    skeleton_array = skeletonize(cleaned_array, transform, projection, output_dir, rasterize=sk_raster)
    
    # Stage 4: Vectorize
    vectorized_gdf = vectorize(skeleton_array, transform, projection)
    
    # Stage 5: Close gaps
    closed_gaps = close_gaps(vectorized_gdf, gap_tolerance, angle_tolerance)
    
    # Stage 6: Merge and export
    merged_gdf = merge_and_export(closed_gaps, output_dir, transform)
    
    # Stage 7: Polygonize
    polygonize(merged_gdf, output_dir, max_area)
    
    # End message
    console.print(f"\nExtracted pastures written to {output_dir}\n", style="bold green")

if __name__ == "__main__":
    main()