# Imports
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from dataclasses import dataclass, field
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
import os, logging

# Env settings
gdal.UseExceptions()
gdal.ConfigurePythonLogging(logger_name="log")
console = Console()


# =============================================================================
# GLOBALS
# =============================================================================
INPUT_RASTER_DIR = "/gpfs/glad1/Exch/Andres_2023/by_Theo/REPROJECTED_3857_v2"
OUTPUT_DIR = f"/gpfs/glad1/Theo/Data/Pastures_test/south_america_polygons"
N_WORKERS = 200  # Number of CPUs to use
GAP_THRESHOLD = 40  # Maximum size of gaps to close (in pixels)
PROBABILITY_THRESHOLD = 15  # Minimum probability for linear features
MIN_AREA = 80  # Minimum size of extracted polygons (in pixels)
EPSG_CODE = 4326  # EPSG code for the final merged vector file

# -----------------------------------------------------------------------------
# Logging setup using Rich
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "log_polygon_extractor.log"
            ),
            mode="w",
        )
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Custom class
# =============================================================================
@dataclass
class RasterInfo:
    """Class to store raster info for later access. Derives certain fields when
    initialized"""

    id: str
    xsize: int
    ysize: int
    transform: tuple
    projection: osr.SpatialReference

    # Derived fields - excluded from __init__
    xmin: float = field(init=False)
    xmax: float = field(init=False)
    ymin: float = field(init=False)
    ymax: float = field(init=False)
    pixel_width: float = field(init=False)
    pixel_height: float = field(init=False)

    def __post_init__(self):
        self.xmin = self.transform[0]
        self.ymax = self.transform[3]
        self.pixel_width = self.transform[1]
        self.pixel_height = self.transform[5]
        self.xmax = self.xmin + self.pixel_width * self.xsize
        self.ymin = self.ymax + self.pixel_height * self.ysize


# =============================================================================
# Utility functions
# =============================================================================
def get_raster_info(raster_path: str) -> RasterInfo:
    """Opens a raster using GDAL and gets columns (xsize), ysize (ysize),
    geotransform, and projection (as a spatial reference object)

    Args:
        raster_path (str): Path to raster dataset

    Returns:
        RasterInfo: RasterInfo object with above information,
        as well as derived objects
    """
    raster_id = os.path.splitext(os.path.basename(raster_path))[0]
    ds = gdal.Open(raster_path)
    info = RasterInfo(
        id=raster_id,
        xsize=ds.RasterXSize,
        ysize=ds.RasterYSize,
        transform=ds.GetGeoTransform(),
        projection=ds.GetSpatialRef(),
    )
    ds = None

    return info


def load_raster(
    input_raster_path: str,
    prob_threshold: int,
) -> tuple:
    """Loads a raster and associated information, and thresholds it to the given
    probability threshold.

    Args:
        input_raster_path (str): Path to raster
        prob_threshold (int): Probability threshold for the array

    Returns:
        tuple: Raster as numpy array, RasterInfo
    """
    # Get raster info
    raster_info = get_raster_info(input_raster_path)

    # Read in as array
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    ds = None

    # Threshold
    raster_array = (raster_array > prob_threshold).astype(np.uint8)

    if not raster_array.any():
        log.debug("Raster load failed, raster_array is empty")

    return raster_array, raster_info


def get_interior(closed_lines: np.ndarray, min_area: int) -> np.ndarray:
    """Finds areas enclosed by lines, which would be areas of zeros fully
    enclosed by ones.

    Args:
        closed_lines (np.array): Lines array with gaps closed
        min_area (int): Minimum area for the interior areas, in pixels

    Returns:
        np.array: Array as np.uint16 data type
    """
    # Find the pixels that are fully enclosed by lines and not lines
    background = closed_lines == 0

    if not background.any():
        log.debug("Getting background failed, empty array")

    labeled_bg, _ = ndimage.label(background)
    border_labels = set()

    for edge in (
        labeled_bg[0, :],
        labeled_bg[-1, :],  # Top and bottom rows
        labeled_bg[:, 0],
        labeled_bg[:, -1],
    ):  # Left and right columns
        border_labels.update(edge.flat)

    border_labels.discard(0)  # Discard lines (ndimage.label makes these 0)
    interior = np.isin(labeled_bg, list(border_labels), invert=True) & background

    # Remove small enclosed regions
    if min_area > 0:
        labeled_interior, _ = ndimage.label(interior)
        sizes = ndimage.sum(
            interior, labeled_interior, range(1, labeled_interior.max() + 1)
        )
        small_labels = np.where(np.array(sizes) < min_area)[0] + 1
        interior[np.isin(labeled_interior, small_labels)] = False

    return interior.astype(np.uint16)


def inspect_file(file: str) -> dict:
    """Worker function for getting file size in parallel.

    Args:
        file (str): Path to file

    Returns:
        dict: File name and size in bytes
    """
    size_bytes = os.path.getsize(file)

    return {"file": file, "size_bytes": size_bytes}


def check_memory_and_merge(files: list, progress: Progress, num_workers: int) -> dict:
    """Uses ThreadPoolExecutor to find file size for a given list of files, and aborts
    if the size is above 1 TB. Estimates the memory needed to merge the files by taking
    input file size and multiplying by 18, which roughly accounts for the buffer and
    graph steps in the merge_vectors function.

    Args:
        files (list): List of absolute paths to files
        progress (Progress): Rich Progress bar
        num_workers (int): Number of CPUs to use

    Returns:
        dict: Status and total size in GBs
    """
    task = progress.add_task("Checking memory", total=len(files))
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(inspect_file, file): file for file in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress.update(task, advance=1)

    # Aggregate after all workers complete
    total_size_bytes = sum(r["size_bytes"] for r in results)
    estimated_ram_gb = (total_size_bytes * 18) / 1e9

    if estimated_ram_gb > 1000:
        return {
            "status": "aborted",
            "total_size_gb": estimated_ram_gb,
        }

    progress.console.print(f"Total size to merge is approx. {estimated_ram_gb:.2f} GB")

    return {
        "status": "success",
        "total_size_gb": estimated_ram_gb,
    }


def merge_vectors(
    input_dir: str,
    output_path: str,
    num_workers: int,
    progress: Progress,
    crs: int = 3857,
    snap_tolerance: float = 1e-8,
) -> dict:
    """Takes an input folder of vectors to merge and checks size, aborts if necessary,
    loads all the gdfs into 1 gdf, buffers to account for polygons sharing a tile
    boundary, and uses an adjacency graph to dissolve.

    Args:
        input_dir (str): Path to vector folder to merge
        output_path (str): Path to output directory
        num_workers (int): Number of CPUs to use when checking memory
        progress (Progress): Rich Progress bar
        crs (int, optional): EPSG code for reprojection. Defaults to 3857.
        snap_tolerance (float, optional): Overlap distance for buffering.
        Defaults to 1e-8.

    Returns:
        dict: Status as either:
        status == empty: reason
        status == aborted: total size gb (aborted if over 1000)
        status == success
    """

    tiles = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".fgb")
    ]

    # Guard against empty tile list
    if not tiles:
        return {"status": "empty", "reason": "no .fgb files found in input_dir"}

    # Check memory constraints
    mem_result = check_memory_and_merge(tiles, progress, num_workers)
    if mem_result["status"] == "aborted":
        return mem_result

    # Load gdfs
    load_task = progress.add_task("Loading tiles...", total=len(tiles))
    gdfs = []
    for t in tiles:
        gdfs.append(gpd.read_file(t))
        progress.update(load_task, advance=1)

    gdf = pd.concat(gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, crs=gdfs[0].crs)

    # Fix floating point gaps at tile seams
    snap_task = progress.add_task("Snapping tile seams...", total=None)
    gdf["geometry"] = gdf.geometry.buffer(snap_tolerance).buffer(-snap_tolerance)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    progress.update(snap_task, completed=1, total=1)

    # Build adjacency graph
    graph_task = progress.add_task("Building adjacency graph...", total=len(gdf))
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    tree = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        candidates = list(tree.query(geom, predicate="touches"))
        for j in candidates:
            if i != j:
                G.add_edge(i, j)
        progress.update(graph_task, advance=1)

    # Dissolve each connected component
    components = list(nx.connected_components(G))
    dissolve_task = progress.add_task("Dissolving components...", total=len(components))
    merged = []
    for component in components:
        idx = list(component)
        subset = gdf.iloc[idx]
        merged_geom = unary_union(subset.geometry)
        merged.append({"geometry": merged_geom})
        progress.update(dissolve_task, advance=1)

    # Save
    save_task = progress.add_task("Saving output...", total=None)
    result = gpd.GeoDataFrame(merged, columns=["geometry"], crs=gdf.crs)
    result = result.explode(index_parts=False).reset_index(drop=True)
    result = result.to_crs(epsg=crs)
    result.to_file(output_path, driver="FlatGeobuf")
    progress.update(save_task, completed=1, total=1)

    return {"status": "success"}


# =============================================================================
# Driver functions
# =============================================================================
def close_gaps(raster_array: np.ndarray, gap_threshold: int) -> np.ndarray:
    """Uses the gap threshold and ndimage convolution and labeling to find and
    close gaps between skeleton endpoints.

    Args:
        raster_array (np.ndarray): Array for the raster
        gap_threshold (int): Maximum gap size to bridge, in pixels

    Returns:
        np.ndarray: Raster array with the gaps closed
    """
    # Skeletonize
    bool_arr = raster_array.astype(bool)
    skeleton = skeletonize(bool_arr).astype(np.uint8)
    if not skeleton.any():
        log.debug("skeletonize failed, thresholded array is empty")

    # Find endpoints
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(skeleton, kernel, mode="constant", cval=0)

    # Remove pixels with no neighbors
    skeleton[neighbor_count == 1] = 0

    # Store endpoints
    endpoints = (skeleton == 1) & (neighbor_count == 2)  # Count: itself & neighbor
    ep_coords = list(zip(*np.where(endpoints)))
    result = skeleton.copy()
    if len(ep_coords) < 2:
        log.debug("No skeleton endpoints found, no gaps closed")
        return result

    # Get all other pixels, create KDTree for faster lookup
    skel_coords = np.array(list(zip(*np.where(skeleton == 1))))
    tree = cKDTree(skel_coords)

    # Label connected components
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, _ = ndimage.label(skeleton, structure=structure)

    # Close gaps with shortest distance
    for i, (r1, c1) in enumerate(ep_coords):
        best_dist = np.inf
        best_coord = None

        # Initialize the KDTree at gap_threshold distance
        indices = tree.query_ball_point([r1, c1], gap_threshold)

        # Create segment labels
        l1 = labeled[r1, c1]

        for idx in indices:
            r2, c2 = skel_coords[idx]

            # Segment labels
            l2 = labeled[r2, c2]

            # Skip if connected
            if l1 == l2:
                continue

            dist = np.hypot(r2 - r1, c2 - c1)
            if dist < best_dist:
                best_dist = dist
                best_coord = (r2, c2)

        # Draw the line between endpoint to close the gap
        if best_coord is not None:
            r2, c2 = best_coord
            rr, cc = draw_line(int(r1), int(c1), int(r2), int(c2))

            # Clip to array bounds just in case
            valid = (
                (rr >= 0)
                & (rr < skeleton.shape[0])
                & (cc >= 0)
                & (cc < skeleton.shape[1])
            )
            result[rr[valid], cc[valid]] = 1

    return result


def find_enclosed_polygons(
    closed_lines: np.ndarray,
    output_dir: str,
    raster_info: RasterInfo,
    min_area: int,
) -> int:
    """Uses the get_interior function to find areas enclosed by lines and
    writes them to a FlatGeobuf file.

    Args:
        closed_lines (np.ndarray): Array of lines with gaps closed
        output_dir (str): Path to output directory
        raster_info (RasterInfo): RasterInfo object
        min_area (int): Minimum area for enclosed areas

    Returns:
        int: Number of enclosed polygons found
    """
    # Get interior pixels
    interior = get_interior(closed_lines, min_area)

    if interior.max() == 0:
        return 0

    mem_ds = gdal.GetDriverByName("MEM").Create(
        "", raster_info.xsize, raster_info.ysize, 1, gdal.GDT_Byte
    )
    mem_band = mem_ds.GetRasterBand(1)
    mem_band.SetNoDataValue(0)
    mem_ds.SetGeoTransform(raster_info.transform)
    mem_ds.SetProjection(raster_info.projection.ExportToWkt())
    mem_band.WriteArray(interior)

    # Create blank vector dataset in memory for polygonization
    polygons_path = os.path.join(output_dir, f"{raster_info.id}_polygons.fgb")
    polygons = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(polygons_path)
    polygons_layer = polygons.CreateLayer(
        "enclosed_polygons", srs=raster_info.projection, geom_type=ogr.wkbPolygon
    )

    # Polygonize
    gdal.Polygonize(mem_band, mem_band, polygons_layer, -1, [], callback=None)
    nb_polygons = polygons_layer.GetFeatureCount()
    mem_ds = None
    polygons = None

    return nb_polygons


# =============================================================================
# Extract polygons
# =============================================================================
def extract_polygons(
    output_dir: str,
    input_raster_path: str,
    gap_threshold: int,
    prob_threshold: int,
    min_area: int,
) -> dict:
    """Main function for the workers to run in parallel. Loads the raster, closes gaps
    between lines, then finds any enclosed polygons. Returns a dict with descriptive
    status messaging.

    Args:
        output_dir (str): Path to output directory
        input_raster_path (str): Path to input raster
        gap_threshold (int): Maximum size of gaps to close (in pixels)
        prob_threshold (int): Minimum probability for linear features thresholding
        min_area (int): Minimum size of extracted polygons (in pixels)

    Returns:
        dict: Raster id, status, and number of extracted polygons
    """
    # Step 1: load raster
    raster_array, raster_info = load_raster(input_raster_path, prob_threshold)

    # Step 2: close gaps
    closed_lines = close_gaps(raster_array, gap_threshold)

    # Step 3: find enclosed polygons
    polygons = find_enclosed_polygons(closed_lines, output_dir, raster_info, min_area)

    if polygons == 0:
        return {
            "raster_id": raster_info.id,
            "status": "null",
            "polygon_count": 0,
        }

    return {
        "raster_id": raster_info.id,
        "status": "success",
        "polygon_count": polygons,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Make output folders
    output_tile_dir = os.path.join(OUTPUT_DIR, "raster_polygons")
    os.makedirs(output_tile_dir, exist_ok=True)

    # Load input rasters
    input_rasters = [
        os.path.join(INPUT_RASTER_DIR, raster)
        for raster in os.listdir(INPUT_RASTER_DIR)
        if raster.endswith(".tif")
    ]

    # Set up arguments for parallel processing
    worker_args = [
        (
            output_tile_dir,
            raster,
            GAP_THRESHOLD,
            PROBABILITY_THRESHOLD,
            MIN_AREA,
        )
        for raster in input_rasters
    ]

    # Parallel process
    completed_tiles = []
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            f"Getting polygons for {INPUT_RASTER_DIR}...", total=len(worker_args)
        )

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(extract_polygons, *args): args[1] for args in worker_args
            }
            for future in as_completed(futures):
                raster_path = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        completed_tiles.append(result["raster_id"])
                        log.info(f"{result['raster_id']}: {result['status']}")
                    elif result["status"] == "null":
                        log.warning(
                            f"{result['raster_id']}: no enclosed polygons found, skipped"
                        )
                    else:
                        log.error(
                            f"{result['raster_id']}: {result.get('reason', 'unknown error')}"
                        )
                except Exception as exc:
                    log.error(f"{raster_path} failed: {exc}")
                finally:
                    progress.update(task, advance=1)

    console.print(f"{len(completed_tiles)} rasters written successfully.")

    # Merge polygons
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        total_to_merge = os.listdir(output_tile_dir)

        if total_to_merge:
            output_vector_path = os.path.join(OUTPUT_DIR, "polygons.fgb")

            try:
                result = merge_vectors(
                    output_tile_dir, output_vector_path, N_WORKERS, progress, EPSG_CODE
                )
                if result["status"] == "success":
                    log.info("Merging complete")
                elif result["status"] == "aborted":
                    log.warning(
                        f"Estimated GB to load was {result['total_size_gb']}."
                        f"Merging was therefore aborted"
                    )
                elif result["status"] == "empty":
                    log.error(
                        f"Input directory to merge was empty"
                        f"Directory: {output_tile_dir}"
                    )
            except Exception as exc:
                log.error(f"Merging failed: {exc}")

    console.print("All polygons merged")


if __name__ == "__main__":
    main()
