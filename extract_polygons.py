# Imports
from osgeo import gdal, osr
from scipy import ndimage
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console
import geopandas as gpd
import numpy as np
import shapely, skimage, fiona, os, argparse, tempfile, gc, shutil, sys

# Env settings
gdal.UseExceptions()
TILE_SIZE = 4096
BATCH_SIZE = 10000

"""
I have written this script to be a command-line utility for extracting areas
enclosed by linear features.
===============================================================================
-p option: path to linear features input raster
-od option: output directory
-sk option: whether or not to write out skeleton raster
-t option: probability threshold of line features. Defaults to 50.
-cr option: pixels to consider for initial gap bridging
-ma option: minimum component size (in pixels) to retain
-gt option: gap tolerance for closing gaps (in meters)
-at option: angle tolerance for closing gaps
-max option: maximum area for extracted plots
"""
# Helper function 1: Calculate bearing
def _endpoint_bearing(line, at_start: bool, lookback: int = 5) -> float:
    coords = list(line.coords)
    lookback = min(lookback, len(coords) - 1)
    if at_start:
        p1, p2 = coords[lookback], coords[0]
    else:
        p1, p2 = coords[-(lookback + 1)], coords[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy)) % 360

# Helper function 2: read gpkg in chunks
def _read_gpkg_chunks(path, chunksize, crs):
    with fiona.open(path) as src:
        chunk = []
        for feature in src:
            chunk.append(feature)
            if len(chunk) >= chunksize:
                yield gpd.GeoDataFrame.from_features(chunk, crs=crs)
                chunk = []
        if chunk:
            yield gpd.GeoDataFrame.from_features(chunk, crs=crs)

# Stage 1: Load the linear features
def load_linear_features(console: Console, tmp_dir: str, linear_features_path: str, threshold: int = 50) -> tuple:
    """
    Loads in a raster of linear features with pixel values from 0-100
    probability. Keeps pixels above the threshold and writes a thresholded
    binary raster to a temporary file.

    Args:
        console (Console): rich Console object.
        linear_features_path (str): path to linear features raster.
        threshold (int, optional): threshold value for masking.
        Defaults to 50.

    Returns:
        tuple: path to thresholded raster, geotransform, projection,
        xsize, ysize.
    """
    linear_features = gdal.Open(linear_features_path)
    xsize = linear_features.RasterXSize
    ysize = linear_features.RasterYSize
    transform = linear_features.GetGeoTransform()
    projection = linear_features.GetSpatialRef()
    band = linear_features.GetRasterBand(1)

    # Create a temporary output raster
    thresholded_path = os.path.join(tmp_dir, "thresholded.tif")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        thresholded_path, xsize, ysize, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]
    )
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection.ExportToWkt())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(255)

    n_tiles_x = (xsize + TILE_SIZE - 1) // TILE_SIZE
    n_tiles_y = (ysize + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = n_tiles_x * n_tiles_y

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:
        task = progress.add_task("Loading and thresholding...", total=total_tiles)

        for y in range(0, ysize, TILE_SIZE):
            for x in range(0, xsize, TILE_SIZE):
                win_xsize = min(TILE_SIZE, xsize - x)
                win_ysize = min(TILE_SIZE, ysize - y)

                tile = band.ReadAsArray(x, y, win_xsize, win_ysize).astype(np.uint8)
                thresholded = np.where(tile > threshold, 1, 0).astype(np.uint8)
                out_band.WriteArray(thresholded, x, y)

                progress.update(task, advance=1)

    out_band.FlushCache()
    out_ds = None
    linear_features = None

    console.print("\u2713 Loaded and thresholded linear features", style="dim green")
    
    return thresholded_path, transform, projection, xsize, ysize


# Stage 2: Morphological cleaning
def morphological_clean(console: Console, tmp_dir: str, thresholded_path: str, transform: tuple, projection: osr.SpatialReference, xsize: int, ysize: int, closing_radius: int = 2, min_area: int = 50) -> str:
    """
    Uses scipy's ndimage to bridge pixel gaps up to the closing radius
    value, then uses gdal.SieveFilter to remove small areas. Reads and
    writes in tiles to minimise memory usage.

    Args:
        console (Console): rich Console object.
        thresholded_path (str): path to thresholded raster from Stage 1.
        transform (tuple): geotransform of the original raster.
        projection (osr.SpatialReference): projection of the original raster.
        xsize (int): raster width in pixels.
        ysize (int): raster height in pixels.
        closing_radius (int, optional): pixels to consider for gap bridging.
        Defaults to 2.
        min_area (int, optional): component size in pixels to retain.
        Defaults to 50.

    Returns:
        str: path to cleaned raster.
    """
    overlap = closing_radius + 1

    in_ds = gdal.Open(thresholded_path)
    in_band = in_ds.GetRasterBand(1)

    cleaned_path = os.path.join(tmp_dir, "cleaned.tif")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        cleaned_path, xsize, ysize, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]
    )
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection.ExportToWkt())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(255)

    structure = ndimage.generate_binary_structure(2, 1)
    structure = ndimage.iterate_structure(structure, closing_radius)

    n_tiles_x = (xsize + TILE_SIZE - 1) // TILE_SIZE
    n_tiles_y = (ysize + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = n_tiles_x * n_tiles_y

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:
        task = progress.add_task("Closing gaps...", total=total_tiles)

        for y in range(0, ysize, TILE_SIZE):
            for x in range(0, xsize, TILE_SIZE):
                win_xsize = min(TILE_SIZE, xsize - x)
                win_ysize = min(TILE_SIZE, ysize - y)

                # Read tile with overlap to avoid edge effects
                x_start = max(0, x - overlap)
                y_start = max(0, y - overlap)
                x_end = min(xsize, x + win_xsize + overlap)
                y_end = min(ysize, y + win_ysize + overlap)

                tile = in_band.ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
                closed = ndimage.binary_closing(tile, structure=structure).astype(np.uint8)

                # Write only the non-overlapping center region
                cx = x - x_start
                cy = y - y_start
                out_band.WriteArray(closed[cy:cy + win_ysize, cx:cx + win_xsize], x, y)

                progress.update(task, advance=1)

        out_band.FlushCache()
        out_ds = None
        in_ds = None

        # Run sieve filter in place on the closed raster
        progress.update(task, description="Removing small areas...", total=None, completed=0)
        sieve_ds = gdal.Open(cleaned_path, gdal.GA_Update)
        gdal.SieveFilter(
            sieve_ds.GetRasterBand(1),
            None,
            sieve_ds.GetRasterBand(1),
            threshold=min_area,
            connectedness=8
        )
        sieve_ds = None

    console.print("\u2713 Removed single pixel gaps and small areas", style="dim green")
    
    return cleaned_path

# Stage 3: Skeletonize
def skeletonize(console, cleaned_path: str, transform: tuple, projection: osr.SpatialReference, xsize: int, ysize: int, output_dir: str = None, rasterize: bool = False) -> str:
    """
    Uses skimage's skeletonize function on the cleaned raster and writes
    the single-pixel-wide centerlines to a temporary raster file.

    Args:
        console (Console): rich Console object.
        cleaned_path (str): path to cleaned raster from Stage 2.
        transform (tuple): geotransform of the original raster.
        projection (osr.SpatialReference): projection of the original raster.
        xsize (int): raster width in pixels.
        ysize (int): raster height in pixels.
        output_dir (str, optional): path to output directory, if desired.
        Defaults to None.
        rasterize (bool, optional): whether or not to write out a skeleton raster.
        Defaults to False.

    Returns:
        str: path to skeleton raster.
    """
    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:
        task = progress.add_task("Reading cleaned raster...", total=None)

        # Read cleaned raster from disk
        in_ds = gdal.Open(cleaned_path)
        cleaned_array = in_ds.GetRasterBand(1).ReadAsArray(0, 0, xsize, ysize)
        in_ds = None

        progress.update(task, description="Skeletonizing linear features...")

        skeleton = skimage.morphology.skeletonize(cleaned_array).astype(np.uint8)
        del cleaned_array
        gc.collect()

        # Write skeleton to temp file if specified
        # Optionally copy to output dir
        if rasterize:
            if output_dir is None:
                raise ValueError("Output directory must be provided if rasterize=True")
            
            progress.update(task, description="Writing skeleton raster...")

            output_tiff = os.path.join(output_dir, "skeleton.tif")
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(
                output_tiff, xsize, ysize, 1, gdal.GDT_Byte,
                options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]
            )
            out_ds.SetGeoTransform(transform)
            out_ds.SetProjection(projection.ExportToWkt())
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(255)
            out_band.WriteArray(skeleton)
            out_band.FlushCache()
            out_ds = None

    console.print("\u2713 Skeletonized linear features", style="dim green")
    
    return skeleton

# Stage 4: Vectorize
def vectorize(console: Console, tmp_dir: str, skeleton_array: np.ndarray, transform: tuple, projection: osr.SpatialReference) -> str:
    """
    Visits all adjacent skeleton pixel pairs and vectorizes them as
    LineStrings, writing to a GeoPackage in batches to minimise memory
    usage.

    Args:
        console (Console): rich Console object.
        skeleton_path (str): path to skeleton raster temp file from Stage 3.
        skeleton_array (np.ndarray): skeleton array from Stage 3.
        transform (tuple): geotransform of the original raster.
        projection (osr.SpatialReference): projection of the original raster.
        output_dir (str): path to the output directory.

    Returns:
        str: path to the vectorized GeoPackage.
    """
    rows, cols = np.where(skeleton_array)
    skeleton_set = set(zip(rows.tolist(), cols.tolist()))
    crs = f"EPSG:{projection.GetAuthorityCode(None)}"

    # Clean up skeleton array
    del skeleton_array
    gc.collect()

    gpkg_path = os.path.join(tmp_dir, "skeleton_lines.gpkg")

    batch = []
    first_write = True

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:
        task = progress.add_task("Vectorizing skeleton...", total=len(skeleton_set))

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
                    batch.append(shapely.LineString([(x1, y1), (x2, y2)]))

            if len(batch) >= BATCH_SIZE:
                gdf = gpd.GeoDataFrame(geometry=batch, crs=crs)
                if first_write:
                    gdf.to_file(gpkg_path, driver="GPKG", mode="w", engine="pyogrio")
                    first_write = False
                else:
                    gdf.to_file(gpkg_path, driver="GPKG", mode="a", engine="pyogrio")
                batch = []

            progress.update(task, advance=1)

        # Write any remaining lines
        if batch:
            gdf = gpd.GeoDataFrame(geometry=batch, crs=crs)
            if first_write:
                gdf.to_file(gpkg_path, driver="GPKG", mode="w", engine="pyogrio")
            else:
                gdf.to_file(gpkg_path, driver="GPKG", mode="a", engine="pyogrio")

    console.print("\u2713 Vectorized skeleton", style="dim green")
    
    return gpkg_path

# Stage 5: Close gaps
def close_gaps(console: Console, tmp_dir: str, gpkg_path: str, projection: osr.SpatialReference, gap_tolerance: float = 10.0, angle_tolerance: float = 30.0) -> str:
    """
    Identifies dangling endpoints in the vectorized line network and
    bridges gaps between them within the gap and angle tolerance.

    Args:
        console (Console): rich Console object.
        gpkg_path (str): path to vectorized GeoPackage from Stage 4.
        output_dir (str): path to the output directory.
        transform (tuple): geotransform of the original raster.
        projection (osr.SpatialReference): projection of the original raster.
        gap_tolerance (float, optional): maximum gap distance in meters.
        Defaults to 10.0.
        angle_tolerance (float, optional): maximum bearing difference in
        degrees. Defaults to 30.0.

    Returns:
        str: path to GeoPackage with bridging lines appended.
    """
    crs = f"EPSG:{projection.GetAuthorityCode(None)}"

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:

        # Pass 1: Count all endpoints to find dangling ones
        # Read total feature count for progress tracking
        with fiona.open(gpkg_path) as src:
            total_lines = len(src)

        task = progress.add_task("Finding dangling endpoints...", total=total_lines)

        endpoint_counts = {}
        endpoint_to_line = {}

        for chunk in _read_gpkg_chunks(gpkg_path, chunksize=BATCH_SIZE, crs=crs):
            for idx, row in chunk.iterrows():
                line = row.geometry
                if line is None:
                    continue
                for point in [line.coords[0], line.coords[-1]]:
                    if point not in endpoint_counts:
                        endpoint_counts[point] = 0
                        endpoint_to_line[point] = idx
                    endpoint_counts[point] += 1
                progress.update(task, advance=1)

        dangling = [pt for pt, count in endpoint_counts.items() if count == 1]
        del endpoint_counts
        gc.collect()

        # Pass 2: Calculate bearings for dangling endpoints
        progress.update(task, description="Calculating bearings...", total=len(dangling), completed=0)

        dangling_set = set(dangling)
        bearings = {}
        line_idx_map = {}

        for chunk in _read_gpkg_chunks(gpkg_path, chunksize=BATCH_SIZE, crs=crs):
            for idx, row in chunk.iterrows():
                line = row.geometry
                if line is None:
                    continue
                start, end = line.coords[0], line.coords[-1]
                if start in dangling_set:
                    bearings[start] = _endpoint_bearing(line, at_start=True)
                    line_idx_map[start] = idx
                    progress.update(task, advance=1)
                if end in dangling_set:
                    bearings[end] = _endpoint_bearing(line, at_start=False)
                    line_idx_map[end] = idx
                    progress.update(task, advance=1)

        del dangling_set
        gc.collect()

        # Build dangling GeoDataFrame for spatial indexing
        progress.update(task, description="Building spatial index...", total=None, completed=0)

        dangling_gdf = gpd.GeoDataFrame(
            {
                "coords": dangling,
                "bearing": [bearings[pt] for pt in dangling],
                "line_idx": [line_idx_map[pt] for pt in dangling]
            },
            geometry=[shapely.Point(pt) for pt in dangling],
            crs=crs
        )
        del bearings, line_idx_map
        gc.collect()

        # Find and synthesize bridging lines
        sindex = dangling_gdf.sindex
        visited_pairs = set()
        bridging_lines = []

        progress.update(task, description="Closing gaps...", total=len(dangling_gdf), completed=0)
        
        bridging_path = os.path.join(tmp_dir, "bridging.gpkg")
        first_write = True

        for i, row in dangling_gdf.iterrows():
            buffer = row.geometry.buffer(gap_tolerance)
            candidates = dangling_gdf.iloc[sindex.query(buffer)]
            candidates = candidates[candidates.geometry.distance(row.geometry) <= gap_tolerance]

            for j, candidate in candidates.iterrows():
                if i == j:
                    continue
                pair = tuple(sorted((i, j)))
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                if row["line_idx"] == candidate["line_idx"]:
                    continue

                angle_diff = abs(row["bearing"] - candidate["bearing"]) % 360
                angle_diff = min(angle_diff, 360 - angle_diff)
                is_similar = angle_diff <= angle_tolerance
                is_opposite = abs(angle_diff - 180) <= angle_tolerance
                if not (is_similar or is_opposite):
                    continue

                bridging_lines.append(shapely.LineString([row.geometry, candidate.geometry]))

            # Write bridging lines in batches
            if len(bridging_lines) >= BATCH_SIZE:
                bridging_gdf = gpd.GeoDataFrame(geometry=bridging_lines, crs=crs)
                if first_write:
                    bridging_gdf.to_file(bridging_path, driver="GPKG", mode="w", engine="pyogrio")
                    first_write = False
                else:
                    bridging_gdf.to_file(bridging_path, driver="GPKG", mode="a", engine="pyogrio")
                bridging_lines = []

            progress.update(task, advance=1)

        # Write remaining bridging lines
        if bridging_lines:
            bridging_gdf = gpd.GeoDataFrame(geometry=bridging_lines, crs=crs)
            if first_write:
                bridging_gdf.to_file(bridging_path, driver="GPKG", mode="w", engine="pyogrio")
            else:
                bridging_gdf.to_file(bridging_path, driver="GPKG", mode="a", engine="pyogrio")
            
    console.print(f"\u2713 Closed gaps", style="dim green")
    
    return bridging_path

# Stage 6: Merge and export
def merge_and_export(console: Console, gpkg_path: str, bridging_path: str, input_file: str, output_dir: str, transform: tuple, projection: osr.SpatialReference) -> str:
    """
    Reads the vectorized lines GeoPackage in chunks, incrementally unions
    and merges them, simplifies to remove staircase artifacts, and writes
    the result to a GeoPackage.

    Args:
        console (Console): rich Console object.
        gpkg_path (str): path to vectorized GeoPackage from Stage 5.
        input_file (str): input filename stem for output naming.
        output_dir (str): path to the output directory.
        transform (tuple): geotransform of the original raster.
        projection (osr.SpatialReference): projection of the original raster.

    Returns:
        str: path to merged GeoPackage.
    """
    crs = f"EPSG:{projection.GetAuthorityCode(None)}"
    with fiona.open(gpkg_path) as src:
        total_lines = len(src)
    if os.path.exists(bridging_path):
        with fiona.open(bridging_path) as src2:
            total_lines += len(src2)
        
    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:

        task = progress.add_task("Merging lines...", total=total_lines)

        # Incrementally union chunks together
        accumulated = None

        for path in [gpkg_path, bridging_path]:
            if not os.path.exists(path):
                continue
            for chunk in _read_gpkg_chunks(gpkg_path, chunksize=BATCH_SIZE, crs=crs):
                chunk_union = shapely.union_all(chunk.geometry.values)
                if accumulated is None:
                    accumulated = chunk_union
                else:
                    accumulated = accumulated.union(chunk_union)
                del chunk, chunk_union
                gc.collect()
                progress.update(task, advance=BATCH_SIZE)

        # Merge, simplify
        progress.update(task, description="Simplifying...", total=None, completed=0)
        merged = shapely.line_merge(accumulated)
        del accumulated
        gc.collect()

        merged = merged.simplify(tolerance=transform[1])

        # Explode MultiLineString to individual geometries
        if merged.geom_type == "MultiLineString":
            geometries = list(merged.geoms)
        else:
            geometries = [merged]
        del merged
        gc.collect()

        # Write to GeoPackage in batches
        progress.update(task, description="Writing GeoPackage...", total=len(geometries), completed=0)
        merged_path = os.path.join(output_dir, f"{input_file}_linear_features.gpkg")
        first_write = True

        for i in range(0, len(geometries), BATCH_SIZE):
            batch = geometries[i:i + BATCH_SIZE]
            gdf = gpd.GeoDataFrame(geometry=batch, crs=crs)
            if first_write:
                gdf.to_file(merged_path, driver="GPKG", mode="w", engine="pyogrio")
                first_write = False
            else:
                gdf.to_file(merged_path, driver="GPKG", mode="a", engine="pyogrio")
            progress.update(task, advance=len(batch))
    
    console.print("\u2713 Merged and exported", style="dim green")
    
    return merged_path

# Stage 7: Polygonize
def polygonize(console: Console, merged_path: str, input_file: str, output_dir: str, projection: osr.SpatialReference, max_area: float = None) -> None:
    """
    Extracts polygons enclosed by the merged lines and writes them to a
    GeoPackage in the output directory.

    Args:
        console (Console): rich Console object.
        merged_path (str): path to merged lines GeoPackage from Stage 6.
        input_file (str): input filename stem for output naming.
        output_dir (str): path to the output directory.
        projection (osr.SpatialReference): projection of the original raster.
        max_area (float, optional): maximum polygon area in square meters
        to retain. Defaults to None.
    """
    crs = f"EPSG:{projection.GetAuthorityCode(None)}"
    with fiona.open(merged_path) as src:
        total_lines = len(src)

    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  MofNCompleteColumn(),
                  TimeElapsedColumn(),
                  transient=True,
                  console=console) as progress:

        # Incrementally union chunks to build noded geometry
        task = progress.add_task("Noding lines...", total=total_lines)
        accumulated = None

        for chunk in _read_gpkg_chunks(merged_path, chunksize=BATCH_SIZE, crs=crs):
            chunk_union = shapely.union_all(chunk.geometry.values)
            if accumulated is None:
                accumulated = chunk_union
            else:
                accumulated = accumulated.union(chunk_union)
            del chunk, chunk_union
            gc.collect()
            progress.update(task, advance=BATCH_SIZE)

        # Polygonize
        progress.update(task, description="Polygonizing...", total=None, completed=0)
        polygons = list(shapely.ops.polygonize(accumulated))
        del accumulated
        gc.collect()

        if not polygons:
            console.print("No enclosed polygons found", style="dim yellow")
            return

        # Filter and write in batches
        progress.update(task, description="Writing polygons...", total=len(polygons), completed=0)
        output_path = os.path.join(output_dir, f"{input_file}_plots.gpkg")
        first_write = True

        for i in range(0, len(polygons), BATCH_SIZE):
            batch = polygons[i:i + BATCH_SIZE]
            batch_gdf = gpd.GeoDataFrame(geometry=batch, crs=crs)
            batch_gdf["area_m2"] = batch_gdf.geometry.area

            if max_area is not None:
                batch_gdf = batch_gdf[batch_gdf["area_m2"] <= max_area]

            if batch_gdf.empty:
                continue

            if first_write:
                batch_gdf.to_file(output_path, driver="GPKG", mode="w", engine="pyogrio")
                first_write = False
            else:
                batch_gdf.to_file(output_path, driver="GPKG", mode="a", engine="pyogrio")

            progress.update(task, advance=len(batch))

    console.print(f"\u2713 Polygonized linear features", style="dim green")
    
# Extract polygons
def extract_polygons(linear_features_path, input_file, output_dir, verbose=False, probability_threshold=50, closing_radius=2, min_area=50, sk_raster=False, gap_tolerance=10.0, angle_tolerance=30.0, max_area=None) -> None:
    # Setup
    console = Console(quiet=not verbose)
    console.print(f"\nEXTRACTING POLYGONS FROM {os.path.basename(linear_features_path)}", style="bold cyan")
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Stage 1: Load linear features
        thresholded_path, transform, projection, xsize, ysize = load_linear_features(console, tmp_dir, linear_features_path, probability_threshold)
        
        # Stage 2: Morphological cleaning
        cleaned_path = morphological_clean(console, tmp_dir, thresholded_path, transform, projection, xsize, ysize, closing_radius, min_area)
        
        # Stage 3: Skeletonize
        skeleton_array = skeletonize(console, cleaned_path, transform, projection, xsize, ysize, output_dir, sk_raster)
        
        # Stage 4: Vectorize
        vectorized_gpkg = vectorize(console, tmp_dir, skeleton_array, transform, projection)
        
        # Stage 5: Close gaps
        closed_gaps_gpkg = close_gaps(console, tmp_dir, vectorized_gpkg, projection, gap_tolerance, angle_tolerance)
        
        # Stage 6: Merge and export
        merged_path = merge_and_export(console, vectorized_gpkg, closed_gaps_gpkg, input_file, output_dir, transform, projection)
        
        # Stage 7: Polygonize
        polygonize(console, merged_path, input_file, output_dir, projection, max_area)
    
        # End message
        console.print(f"\nExtracted polygons written to {output_dir}\n", style="bold green")
    
    except Exception as e:
        console.print(f"\n\u2717Error during extraction: {e}", style="bold red")
        console.print_exception()
        raise
    
    finally:
        shutil.rmtree(tmp_dir)
    
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
    
    verbose = True
    input_file = os.path.splitext(os.path.basename(linear_features_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Extract polygons
    try:
        extract_polygons(linear_features_path, input_file, output_dir, verbose, probability_threshold, closing_radius, min_area, sk_raster, gap_tolerance, angle_tolerance, max_area)
    except Exception:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()