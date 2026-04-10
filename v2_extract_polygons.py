# Imports
from osgeo import gdal, ogr, osr
from scipy import ndimage
from skimage.morphology import skeletonize, disk
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from dataclasses import dataclass, field
import numpy as np
import os, logging, math

# Env settings
gdal.UseExceptions()
console = Console()


# =============================================================================
# GLOBALS
# =============================================================================
BLOCK_ID = 0
INPUT_RASTER = "/gpfs/glad1/Theo/Data/Pastures_test/test_rasters.tif"
OUTPUT_DIR = "/gpfs/glad1/Theo/Data/Pastures_test/test_output/v2"
N_WORKERS = 100
BLOCKSIZE = 512
BUFFER_DIST = 256
GAP_THRESHOLD = 100
PROBABILITY_THRESHOLD = 10


# -----------------------------------------------------------------------------
# Logging setup using Rich
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            log_time_format="[%H:%M:%S]"
        ),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_polygon_extractor.log"),
            mode="w"
        )
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# Custom classes
# =============================================================================
@dataclass
class BlockInfo():
    # Block's id
    block_id: int 
    # True block (what the buffer is built around)
    block_col: int          # x offset of the block
    block_row: int          # y offset of the block
    block_width: int        # pixel columns in the block
    block_height: int       # pixel rows in the block
    
    # Buffered region: what we actually read from disk
    buf_col: int            # x offset of the buffered read
    buf_row: int            # y offset of the buffered read
    buf_width: int          # pixel columns actually read
    buf_height: int         # pixel rows actually read

    # How far the true block is from the buffered origin
    inner_col_offset: int   # = block_col - buf_col
    inner_row_offset: int   # = block_row - buf_row
    
@dataclass
class RasterInfo():
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
        self.xmin        = self.transform[0]
        self.ymax        = self.transform[3]
        self.pixel_width = self.transform[1]
        self.pixel_height= self.transform[5]
        self.xmax        = self.xmin + self.pixel_width * self.cols
        self.ymin        = self.ymax + self.pixel_height * self.rows

# =============================================================================
# Utility functions
# =============================================================================
def get_raster_info(raster_path):
    ds = gdal.Open(raster_path)
    info = RasterInfo(
        cols=ds.RasterXSize,
        rows=ds.RasterYSize,
        transform=ds.GetGeoTransform(),
        projection=ds.GetSpatialRef()
    )
    ds = None
    
    return info
    
def get_block_info(block_id, raster_info, blocksize, buffer):
    n_blocks_x = math.ceil(raster_info.cols  / blocksize)
    n_blocks_y = math.ceil(raster_info.rows / blocksize)
    block_row_idx = block_id // n_blocks_x
    block_col_idx = block_id  % n_blocks_x

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
    buf_width = buf_right  - buf_col
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

def load_block(block_id, input_raster_path, blocksize, buffer_dist, prob_threshold):
    # Get raster info
    raster_info = get_raster_info(input_raster_path)
    
    # Get block info
    block_info = get_block_info(block_id, raster_info, blocksize, buffer_dist)
    
    # Read in block
    ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    block = band.ReadAsArray(
        block_info.buf_col,    # xoff
        block_info.buf_row,    # yoff
        block_info.buf_width,  # xsize
        block_info.buf_height  # ysize
    )
    
    ds = None
    
    # Convert to mask
    block = (block > prob_threshold).astype(np.uint8)
    
    if not block.any():
        log.debug("Block load failed, block array is empty")
    
    return block, block_info, raster_info

def get_interior(closed_lines):
    # Find the pixels that are fully enclosed by lines and not lines
    background = (closed_lines == 0)
    
    if not background.any():
        log.debug("Getting background failed, empty array")
        
    labeled_bg, _ = ndimage.label(background)
    border_labels = set()
    
    for edge in (labeled_bg[0, :], labeled_bg[-1, :],  # Top and bottom rows
                 labeled_bg[:, 0], labeled_bg[:, -1]): # Left and right columns
        border_labels.update(edge.flat)
    
    border_labels.discard(0)  # Discard lines (ndimage.label makes these 0)
    interior = np.isin(labeled_bg, list(border_labels), invert=True) & background
    
    return interior.astype(np.uint8)


# =============================================================================
# Driver functions
# =============================================================================
def close_gaps(block, gap_threshold):
    bool_arr = block.astype(bool)
    
    # Use morphological closing to remove gaps
    closed = ndimage.binary_closing(bool_arr, structure=disk(gap_threshold))
    
    # Skeletonize
    closed = skeletonize(closed)
    
    if not closed.any():
        log.debug("skeletonize failed, closed array is empty")
    
    return closed.astype(np.uint8)

def find_enclosed_polygons(closed_lines: np.ndarray, block_info: BlockInfo, raster_info: RasterInfo):
    # Get interior pixels within block & buffer
    interior_buffer = get_interior(closed_lines)
    
    # Crop this to the block extent (removing buffer)
    r0 = block_info.inner_row_offset
    c0 = block_info.inner_col_offset
    interior_block = interior_buffer[r0:r0 + block_info.block_height,  # Top and bottom rows
                              c0:c0 + block_info.block_width]   # Left and right columns
    
    if interior_block.max() == 0:
        return 0
    
    # Create a blank raster dataset in memory for polygonization
    gt = list(raster_info.transform)
    gt[0] = raster_info.xmin + block_info.block_col * raster_info.pixel_width  # x origin
    gt[3] = raster_info.ymax + block_info.block_row * raster_info.pixel_height # y origin
        
    mem_ds = gdal.GetDriverByName("MEM").Create(
        "", block_info.block_width, block_info.block_height, 1, gdal.GDT_Byte
    )
    mem_ds.SetGeoTransform(gt)
    mem_ds.SetProjection(raster_info.projection.ExportToWkt())
    
    mem_band = mem_ds.GetRasterBand(1)
    mem_band.SetNoDataValue(0)
    
    # Create blank vector dataset in memory for polygonization
    temp_vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    temp_layer = temp_vector.CreateLayer("enclosed_polygons", srs=raster_info.projection, geom_type=ogr.wkbPolygon)
    fieldname = ogr.FieldDefn("block_id", ogr.OFTInteger)
    temp_layer.CreateField(fieldname)
    
    # Polygonize
    mem_band.WriteArray(interior_block)
    gdal.Polygonize(mem_band, mem_band, temp_layer, 0, [], callback=None)
    polygons = []
    for feat in temp_layer:
        geom = feat.GetGeometryRef()
        if geom:
            polygons.append((block_info.block_id, geom.ExportToWkt()))
            
    mem_ds = None
    temp_vector = None
    
    return polygons
    
# =============================================================================
# Extract polygons
# =============================================================================
def extract_polygons(block_id, input_raster_path, blocksize, buffer_dist, gap_threshold, prob_threshold):
    # Check for buffer distance/gap threshold tolerance
    if buffer_dist < gap_threshold:
        log.warning(
            f"Buffer ({buffer_dist}) < gap threshold ({gap_threshold})."
            "Edge artifacts are likely. Consider setting buffer distance >= gap threshold")
    
    # Step 1: load block
    block, block_info, raster_info = load_block(block_id, input_raster_path, blocksize, buffer_dist, prob_threshold)
    
    # Step 2: close gaps
    closed_lines = close_gaps(block, gap_threshold)
    
    # Step 3: find enclosed polygons
    polygons = find_enclosed_polygons(closed_lines, block_info, raster_info)
    if polygons == 0:
        log.debug("no enclosed polygons found")
        return {
            "block_id": block_id,
            "status": "null",
            "polygon_count": 0,
            "polygons": []
        }
    
    return {
        "block_id": block_id,
        "status": "success",
        "polygon_count": len(polygons),
        "polygons": polygons
    }
    

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Make output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open raster dataset and get number of blocks
    log.debug("Getting raster info")
    info = get_raster_info(INPUT_RASTER)
    
    nx = math.ceil(info.cols / BLOCKSIZE)
    ny = math.ceil(info.rows / BLOCKSIZE)
    total_blocks = nx * ny
    
    # Create output geopackage for polygons
    output_vector_path = os.path.join(OUTPUT_DIR, "polygons.gpkg")
    output_vector = ogr.GetDriverByName("GPKG").CreateDataSource(output_vector_path)
    output_layer = output_vector.CreateLayer("enclosed_polygons", srs=info.projection, geom_type=ogr.wkbPolygon)
    fieldname = ogr.FieldDefn("block_id", ogr.OFTInteger)
    output_layer.CreateField(fieldname)
    
    # Set up arguments for parallel processing
    worker_args = [
        (block_id, INPUT_RASTER, BLOCKSIZE, BUFFER_DIST, GAP_THRESHOLD, PROBABILITY_THRESHOLD)
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
        transient=True,
    ) as progress:

        task = progress.add_task("Getting polygons by block...", total=len(worker_args))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(extract_polygons, *args): args[0] for args in worker_args}
            for future in as_completed(futures):
                block_id = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        completed_blocks.append(result["block_id"])
                        log.info(f"{result['block_id']}: {result['status']}")
                    elif result["status"] == "null":
                        log.warning(f"{result['block_id']}: no enclosed polygons found, skipped")
                    else:
                        log.error(f"{result['block_id']}: {result.get('reason', 'unknown error')}")
                except Exception as exc:
                    log.error(f"{block_id} failed: {exc}")
                finally:
                    progress.update(task, advance=1)

    log.info(f"{len(completed_blocks)} blocks written successfully.")
    
    output_vector = None
    
if __name__ == "__main__":
    main()
    
    # log.debug("Getting raster info")
    
    # info = get_raster_info(INPUT_RASTER)
    
    # log.debug(f"Extracting polygons for block {BLOCK_ID}")
    
    # result = extract_polygons(BLOCK_ID, INPUT_RASTER, BLOCKSIZE, BUFFER_DIST, GAP_THRESHOLD, PROBABILITY_THRESHOLD)
    
    # output_vector_path = os.path.join(OUTPUT_DIR, "polygons.gpkg")
    # output_vector = ogr.GetDriverByName("GPKG").CreateDataSource(output_vector_path)
    # output_layer = output_vector.CreateLayer("enclosed_polygons", srs=info.projection, geom_type=ogr.wkbPolygon)
    # fieldname = ogr.FieldDefn("block_id", ogr.OFTInteger)
    # output_layer.CreateField(fieldname)
    
    # with Progress(
    #     SpinnerColumn(),
    #     "[progress.description]{task.description}",
    #     MofNCompleteColumn(),
    #     TimeElapsedColumn(),
    #     console=console,
    #     transient=True,
    # ) as progress:

    #     task = progress.add_task("Writing polygons...", total=len(result["polygons"]))
        
    #     for b_id, wkt_geom in result["polygons"]:
    #         feat = ogr.Feature(output_layer.GetLayerDefn())
    #         feat.SetField("block_id", b_id)
    #         feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt_geom, info.projection))
    #         output_layer.CreateFeature(feat)
    #         feat = None
            
    # output_vector = None