# Imports
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from dataclasses import dataclass, field
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
import os, logging, math, fiona

# Env settings
gdal.UseExceptions()
gdal.ConfigurePythonLogging(logger_name="log")
console = Console()


# =============================================================================
# GLOBALS
# =============================================================================
INPUT_RASTER = "/gpfs/glad1/Theo/Data/Pastures_test/v1_test_lines.tif"
OUTPUT_DIR = f"/gpfs/glad1/Theo/Data/Pastures_test/test_output2"
N_WORKERS = 100  # Number of CPUs to use
BLOCKSIZE = 2048  # Size of the block
BUFFER_DIST = 2048  # Size of the surrounding buffer
GAP_THRESHOLD = 40  # Maximum size of gaps to close (in pixels)
PROBABILITY_THRESHOLD = 15  # For linear features
MIN_AREA = 80  # Minimum size of extracted polygons (in pixels)

# -----------------------------------------------------------------------------
# Logging setup using Rich
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "log_extract_polygons_by_block.log",
            ),
            mode="w",
        )
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Custom classes
# =============================================================================
@dataclass
class BlockInfo:
    """Class to store block info"""

    # Block's id
    block_id: int
    # True block (what the buffer is built around)
    block_col: int  # x offset of the block
    block_row: int  # y offset of the block
    block_width: int  # pixel columns in the block
    block_height: int  # pixel rows in the block

    # Buffered region: what we actually read from disk
    buf_col: int  # x offset of the buffered read
    buf_row: int  # y offset of the buffered read
    buf_width: int  # pixel columns actually read
    buf_height: int  # pixel rows actually read

    # How far the true block is from the buffered origin
    inner_col_offset: int  # = block_col - buf_col
    inner_row_offset: int  # = block_row - buf_row


@dataclass
class RasterInfo:
    """Class to store raster info for later access. Derives certain fields when
    initialized"""

    cols: int
    rows: int
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
        self.xmax = self.xmin + self.pixel_width * self.cols
        self.ymin = self.ymax + self.pixel_height * self.rows


# =============================================================================
# Utility functions
# =============================================================================
def get_raster_info(raster_path: str) -> RasterInfo:
    """Opens a raster using GDAL and gets columns (xsize), rows (ysize),
    geotransform, and projection (as a spatial reference object)

    Args:
        raster_path (str): Path to raster dataset

    Returns:
        RasterInfo: RasterInfo object with above information,
        as well as derived objects
    """
    ds = gdal.Open(raster_path)
    info = RasterInfo(
        cols=ds.RasterXSize,
        rows=ds.RasterYSize,
        transform=ds.GetGeoTransform(),
        projection=ds.GetSpatialRef(),
    )
    ds = None

    return info


def get_block_info(
    block_id: int, raster_info: RasterInfo, blocksize: int, buffer: int
) -> BlockInfo:
    """Locates a block based on the block_id in the raster and computes a
    BlockInfo object with associated buffer zone.

    Args:
        block_id (int): Block's unique id
        raster_info (RasterInfo): RasterInfo object from the input raster
        blocksize (int): Size of the block, in pixels
        buffer (int): Size of the buffer, in pixels

    Raises:
        ValueError: If block id exceeds the bounds of the raster

    Returns:
        BlockInfo: BlockInfo object
    """
    n_blocks_x = math.ceil(raster_info.cols / blocksize)
    n_blocks_y = math.ceil(raster_info.rows / blocksize)
    block_row_idx = block_id // n_blocks_x
    block_col_idx = block_id % n_blocks_x

    if block_row_idx >= n_blocks_y:
        raise ValueError(f"block_id {block_id} exceeds raster extent.")

    # Get true block extents (some blocks on the edge may be smaller than blocksize)
    block_col = block_col_idx * blocksize
    block_row = block_row_idx * blocksize
    block_width = min(blocksize, raster_info.cols - block_col)
    block_height = min(blocksize, raster_info.rows - block_row)

    # Get buffer extents, clamped to raster dimensions
    buf_col = max(0, block_col - buffer)
    buf_row = max(0, block_row - buffer)
    buf_right = min(raster_info.cols, block_col + block_width + buffer)
    buf_bottom = min(raster_info.rows, block_row + block_height + buffer)
    buf_width = buf_right - buf_col
    buf_height = buf_bottom - buf_row

    # Get offset of the true block inside the buffered array
    inner_col_offset = block_col - buf_col
    inner_row_offset = block_row - buf_row

    return BlockInfo(
        block_id=block_id,
        block_col=block_col,
        block_row=block_row,
        block_width=block_width,
        block_height=block_height,
        buf_col=buf_col,
        buf_row=buf_row,
        buf_width=buf_width,
        buf_height=buf_height,
        inner_col_offset=inner_col_offset,
        inner_row_offset=inner_row_offset,
    )


def load_block(
    block_id: int,
    input_raster_path: str,
    blocksize: int,
    buffer_dist: int,
    prob_threshold: int,
) -> tuple:
    """Loads a block and associated information, and thresholds it to the given
    probability threshold.

    Args:
        block_id (int): Block's unique id
        input_raster_path (str): Path to raster
        blocksize (int): Size of the block, in pixels
        buffer_dist (int): Size of the buffer, in pixels
        prob_threshold (int): Probability threshold for the block

    Returns:
        tuple: Block as numpy array, BlockInfo, RasterInfo
    """
    # Get raster info
    raster_info = get_raster_info(input_raster_path)

    # Get block info
    block_info = get_block_info(block_id, raster_info, blocksize, buffer_dist)

    # Read in block
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    block = band.ReadAsArray(
        block_info.buf_col,  # xoff
        block_info.buf_row,  # yoff
        block_info.buf_width,  # xsize
        block_info.buf_height,  # ysize
    )

    ds = None

    # Threshold
    block = (block > prob_threshold).astype(np.uint8)

    if not block.any():
        log.debug("Block load failed, block array is empty")

    return block, block_info, raster_info


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
    size_bytes = os.path.getsize(file)
    with fiona.open(file) as src:
        feature_count = len(src)
    return {"file": file, "size_bytes": size_bytes, "features": feature_count}


def check_memory_and_merge(files: list, progress: Progress, num_workers: int) -> dict:
    task = progress.add_task("Checking memory", total=len(files))
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(inspect_file, file): file for file in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress.update(task, advance=1)

    # Aggregate after all workers complete
    total_features = sum(r["features"] for r in results)
    total_size_bytes = sum(r["size_bytes"] for r in results)
    estimated_ram_gb = (total_size_bytes * 4) / 1e9

    if estimated_ram_gb > 900:
        return {
            "status": "aborted",
            "total_features": total_features,
            "total_size_gb": estimated_ram_gb,
        }

    return {
        "status": "success",
        "total_features": total_features,
        "total_size_gb": estimated_ram_gb,
    }


def merge_vectors(
    input_dir: str,
    output_path: str,
    num_workers: int,
    progress: Progress,
    snap_tolerance: float = 1e-8,
) -> dict:
    tiles = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".fgb")
    ]

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
        merged.append(
            {
                "geometry": merged_geom,
                "block_id": ",".join(subset["block_id"].astype(str).unique()),
            }
        )
        progress.update(dissolve_task, advance=1)

    # Save
    save_task = progress.add_task("Saving output...", total=None)
    result = gpd.GeoDataFrame(merged, columns=["geometry", "block_id"], crs=gdf.crs)
    result = result.explode(index_parts=False).reset_index(drop=True)
    result.to_file(output_path, driver="FlatGeobuf")
    progress.update(save_task, completed=1, total=1)

    return {"status": "success"}


# =============================================================================
# Driver functions
# =============================================================================
def close_gaps(
    block: np.ndarray, gap_threshold: int, block_info: BlockInfo
) -> np.ndarray:
    """Uses the gap threshold and ndimage convolution and labeling to find and
    close gaps between skeleton endpoints.

    Args:
        block (np.ndarray): Array for the block
        gap_threshold (int): Maximum gap size to bridge, in pixels
        block_info (BlockInfo): BlockInfo object

    Returns:
        np.ndarray: Block array with the gaps closed
    """
    # Skeletonize
    bool_arr = block.astype(bool)
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

    # Create labeling scheme for segments only connected in the block
    inner_mask = np.zeros_like(skeleton, dtype=bool)
    inner_mask[
        block_info.inner_row_offset : block_info.inner_row_offset
        + block_info.block_height,
        block_info.inner_col_offset : block_info.inner_col_offset
        + block_info.block_width,
    ] = True

    # Label over full buffer extent
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_buffer, _ = ndimage.label(skeleton, structure=structure)

    # Label over inner block only
    skeleton_inner = skeleton * inner_mask.astype(np.uint8)
    labeled_inner, _ = ndimage.label(skeleton_inner, structure=structure)

    # Close gaps with shortest distance
    for i, (r1, c1) in enumerate(ep_coords):
        best_dist = np.inf
        best_coord = None

        # Initialize the KDTree at gap_threshold distance
        indices = tree.query_ball_point([r1, c1], gap_threshold)

        # Create segment labels
        l1_buf = labeled_buffer[r1, c1]
        l1_inn = labeled_inner[r1, c1]

        for idx in indices:
            r2, c2 = skel_coords[idx]

            # Segment labels
            l2_buf = labeled_buffer[r2, c2]
            l2_inn = labeled_inner[r2, c2]

            # Skip only if connected in BOTH block and buffer
            if l1_inn == l2_inn and l1_buf == l2_buf:
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
    block_info: BlockInfo,
    raster_info: RasterInfo,
    min_area: int,
) -> int:
    """Uses the get_interior function to find areas enclosed by lines and
    writes them to a block-wide FlatGeobuf file.

    Args:
        closed_lines (np.ndarray): Array of lines with gaps closed
        output_dir (str): Path to output directory
        block_info (BlockInfo): BlockInfo object
        raster_info (RasterInfo): RasterInfo object
        min_area (int): Minimum area for enclosed areas

    Returns:
        int: Number of enclosed polygons found
    """
    # Get interior pixels within block & buffer
    interior_buffer = get_interior(closed_lines, min_area)

    # Crop this to the block extent (removing buffer)
    r0 = block_info.inner_row_offset
    c0 = block_info.inner_col_offset
    interior_block = interior_buffer[
        r0 : r0 + block_info.block_height,  # Top and bottom rows
        c0 : c0 + block_info.block_width,
    ]  # Left and right columns

    if interior_block.max() == 0:
        return 0

    # Store block id + 1 because blocks are 0-indexed
    interior_block[interior_block == 1] = block_info.block_id + 1

    # Create raster dataset in memory for polygonization,
    gt = list(raster_info.transform)
    gt[0] = (
        raster_info.xmin + block_info.block_col * raster_info.pixel_width
    )  # x origin
    gt[3] = (
        raster_info.ymax + block_info.block_row * raster_info.pixel_height
    )  # y origin

    mem_ds = gdal.GetDriverByName("MEM").Create(
        "", block_info.block_width, block_info.block_height, 1, gdal.GDT_Byte
    )
    mem_ds.SetGeoTransform(gt)
    mem_ds.SetProjection(raster_info.projection.ExportToWkt())
    mem_band = mem_ds.GetRasterBand(1)
    mem_band.SetNoDataValue(0)
    mem_band.WriteArray(interior_block)

    # Create blank vector dataset in memory for polygonization
    block_polygons_path = os.path.join(
        output_dir, f"{block_info.block_id}_polygons.fgb"
    )
    block_polygons = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(
        block_polygons_path
    )
    polygons_layer = block_polygons.CreateLayer(
        "enclosed_polygons", srs=raster_info.projection, geom_type=ogr.wkbPolygon
    )
    fieldname = ogr.FieldDefn("block_id", ogr.OFTInteger)
    polygons_layer.CreateField(fieldname)

    # Polygonize
    gdal.Polygonize(mem_band, mem_band, polygons_layer, 0, [], callback=None)
    nb_polygons = len(polygons_layer)

    mem_ds = None
    block_polygons = None

    return nb_polygons


# =============================================================================
# Extract polygons
# =============================================================================
def extract_polygons(
    block_id: int,
    output_dir: str,
    input_raster_path: str,
    blocksize: int,
    buffer_dist: int,
    gap_threshold: int,
    prob_threshold: int,
    min_area: int,
) -> dict:
    # Check for buffer distance/gap threshold tolerance
    if buffer_dist < gap_threshold:
        log.warning(
            f"Buffer ({buffer_dist}) < gap threshold ({gap_threshold})."
            "Edge artifacts are likely."
            "Consider setting buffer distance >= gap threshold"
        )

    # Step 1: load block
    block, block_info, raster_info = load_block(
        block_id, input_raster_path, blocksize, buffer_dist, prob_threshold
    )

    # Step 2: close gaps
    closed_lines = close_gaps(block, gap_threshold, block_info)

    # Step 3: find enclosed polygons
    polygons = find_enclosed_polygons(
        closed_lines, output_dir, block_info, raster_info, min_area
    )

    if polygons == 0:
        return {
            "block_id": block_id,
            "status": "null",
            "polygon_count": 0,
            "polygons": polygons,
        }

    return {
        "block_id": block_id,
        "status": "success",
        "polygon_count": polygons,
        "polygons": polygons,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Make output folders
    output_block_dir = os.path.join(OUTPUT_DIR, "block_polygons")
    os.makedirs(output_block_dir, exist_ok=True)

    # Open raster dataset and get number of blocks
    raster_info = get_raster_info(INPUT_RASTER)

    nx = math.ceil(raster_info.cols / BLOCKSIZE)
    ny = math.ceil(raster_info.rows / BLOCKSIZE)
    total_blocks = nx * ny

    # Set up arguments for parallel processing
    worker_args = [
        (
            block_id,
            output_block_dir,
            INPUT_RASTER,
            BLOCKSIZE,
            BUFFER_DIST,
            GAP_THRESHOLD,
            PROBABILITY_THRESHOLD,
            MIN_AREA,
        )
        for block_id in range(total_blocks)
    ]

    # Parallel process
    completed_blocks = []
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Getting polygons by block...", total=len(worker_args))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(extract_polygons, *args): args[0] for args in worker_args
            }
            for future in as_completed(futures):
                block_id = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        completed_blocks.append(result["block_id"])
                        log.info(f"{result['block_id']}: {result['status']}")
                    elif result["status"] == "null":
                        log.warning(
                            f"{result['block_id']}: no enclosed polygons found, skipped"
                        )
                    else:
                        log.error(
                            f"{result['block_id']}: {result.get('reason', 'unknown error')}"
                        )
                except Exception as exc:
                    log.error(f"{block_id} failed: {exc}")
                finally:
                    progress.update(task, advance=1)

    console.print(f"{len(completed_blocks)} blocks written successfully.")

    # Merge block polygons
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        total_to_merge = os.listdir(output_block_dir)

        if total_to_merge:
            output_vector_path = os.path.join(OUTPUT_DIR, "polygons.fgb")

            try:
                result = merge_vectors(
                    output_block_dir, output_vector_path, N_WORKERS, progress
                )
                if result["status"] == "success":
                    log.info("Merging complete")
                elif result["status"] == "aborted":
                    log.warning(
                        f"Number of features was {result['total_features']}."
                        f"Estimated GB to load was {result['total_size_gb']}."
                        "Merging was therefore aborted"
                    )
            except Exception as exc:
                log.error(f"Merging failed: {exc}")

    console.print("All polygons merged")


if __name__ == "__main__":
    main()
