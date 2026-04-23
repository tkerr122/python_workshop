# Imports
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize
from rich.console import Console
from dataclasses import dataclass, field
import numpy as np
import os, logging, math

# Env settings
gdal.UseExceptions()
gdal.ConfigurePythonLogging(logger_name="log")
console = Console()


# =============================================================================
# GLOBALS
# =============================================================================
BLOCK_ID = 32
INPUT_RASTER = "/gpfs/glad1/Theo/Data/Pastures_test/v1_test_lines.tif"
OUTPUT_DIR = f"/gpfs/glad1/Theo/Data/Pastures_test/block{BLOCK_ID}"
BLOCKSIZE = 2048     # Size of the block
BUFFER_DIST = 2048   # Size of the surrounding buffer
GAP_THRESHOLD = 30   # Maximum size of gaps to close (in pixels)
PROBABILITY_THRESHOLD = 15  # For linear features
MIN_AREA = 80        # Minimum size of extracted polygons (in pixels)

# -----------------------------------------------------------------------------
# Logging setup using Rich
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_log_polygon_extractor.log"),
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

#* ---DEBUG---
def write_block_debug(block_info, raster_info, output_path):
    gt = raster_info.transform
    
    def pixel_to_geo(col, row):
        x = gt[0] + col * gt[1]
        y = gt[3] + row * gt[5]

        return x, y

    def make_box(col, row, width, height):
        x_min, y_max = pixel_to_geo(col, row)
        x_max, y_min = pixel_to_geo(col + width, row + height)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_min, y_max)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        return poly

    driver = ogr.GetDriverByName("FlatGeobuf")
    ds = driver.CreateDataSource(output_path)
    layer = ds.CreateLayer("blocks", srs=raster_info.projection, geom_type=ogr.wkbUnknown)
    layer.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    for label, col, row, width, height in [
        ("block",  block_info.block_col, block_info.block_row, block_info.block_width,  block_info.block_height),
        ("buffer", block_info.buf_col,   block_info.buf_row,   block_info.buf_width,    block_info.buf_height),
    ]:
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("label", label)
        feat.SetGeometry(make_box(col, row, width, height))
        layer.CreateFeature(feat)

    ds = None


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
    
    # Threshold
    block = (block > prob_threshold).astype(np.uint8)
    
    if not block.any():
        log.debug("Block load failed, block array is empty")
    
    return block, block_info, raster_info

def get_interior(closed_lines, min_area):
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
    
    # Remove small enclosed regions
    if min_area > 0:
        labeled_interior, _ = ndimage.label(interior)
        sizes = ndimage.sum(interior, labeled_interior, range(1, labeled_interior.max() + 1))
        small_labels = np.where(np.array(sizes) < min_area)[0] + 1
        interior[np.isin(labeled_interior, small_labels)] = False
    
    return interior.astype(np.uint16)

def merge_vectors(input_dir, output_path, info, progress, task):
    out_ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("merged_polygons", srs=info.projection, geom_type=ogr.wkbPolygon)
    fieldname = ogr.FieldDefn("block_id", ogr.OFTInteger)
    out_layer.CreateField(fieldname)

    for file in os.listdir(input_dir):
        if file.endswith(".fgb"):
            filepath = os.path.join(input_dir, file)
            ds = ogr.Open(filepath)
            lyr = ds.GetLayer()
            for feat in lyr:
                out_feat = ogr.Feature(out_layer.GetLayerDefn())
                out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                fid = feat.GetField("block_id") - 1
                fid = (fid - 1) if fid is not None else None
                out_feat.SetField("block_id", fid)
                out_layer.CreateFeature(out_feat)
                out_feat = None
            
            ds = None

        progress.update(task, advance=1)
    
    out_ds = None

# def get_block_geographic_bounds(block_info: BlockInfo, raster_info: RasterInfo) -> tuple:
#     """Returns (xmin, ymin, xmax, ymax) of the TRUE block (not buffered) in map coordinates."""
#     xmin = raster_info.xmin + block_info.block_col * raster_info.pixel_width
#     xmax = xmin + block_info.block_width * raster_info.pixel_width
#     # pixel_height is negative (top-down), so ymax is the top edge
#     ymax = raster_info.ymax + block_info.block_row * raster_info.pixel_height
#     ymin = ymax + block_info.block_height * raster_info.pixel_height
#     return (xmin, ymin, xmax, ymax)


# def touches_block_edge(geom_env: tuple, block_env: tuple, tol: float) -> bool:
#     """
#     geom_env:  (minx, miny, maxx, maxy) from OGR geometry envelope
#     block_env: (xmin, ymin, xmax, ymax) of the true block
#     tol:       tolerance in map units (e.g. 1 pixel width)
#     """
#     gminx, gminy, gmaxx, gmaxy = geom_env
#     bminx, bminy, bmaxx, bmaxy = block_env
#     return (
#         abs(gminx - bminx) < tol or
#         abs(gminy - bminy) < tol or
#         abs(gmaxx - bmaxx) < tol or
#         abs(gmaxy - bmaxy) < tol
#     )


# def merge_vectors(
#     input_dir, output_path, info: RasterInfo, blocksize, buffer,
#     progress, task
# ):
#     tol = abs(info.pixel_width)  # 1-pixel tolerance for edge detection

#     # --- Output FGB (interior polygons only in pass 1) ---
#     out_driver = ogr.GetDriverByName("FlatGeobuf")
#     out_ds = out_driver.CreateDataSource(output_path)
#     out_layer = out_ds.CreateLayer("merged_polygons", srs=info.projection, geom_type=ogr.wkbPolygon)
#     out_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTInteger))

#     # --- Candidates FGB (boundary-touching polygons) ---
#     candidates_path = os.path.join(os.path.dirname(output_path), "_candidates.fgb")
#     cand_ds = out_driver.CreateDataSource(candidates_path)
#     cand_layer = cand_ds.CreateLayer("candidates", srs=info.projection, geom_type=ogr.wkbPolygon)
#     cand_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTInteger))

#     for file in sorted(os.listdir(input_dir)):
#         if not file.endswith(".fgb"):
#             continue

#         filepath = os.path.join(input_dir, file)

#         # Derive block_id from filename - adjust to your naming convention
#         try:
#             block_id = int(os.path.splitext(file)[0].split("_")[-1])
#         except ValueError:
#             progress.update(task, advance=1)
#             continue

#         block_info = get_block_info(block_id, info, blocksize, buffer)
#         block_env = get_block_geographic_bounds(block_info, info)

#         ds = ogr.Open(filepath)
#         lyr = ds.GetLayer()

#         for feat in lyr:
#             geom = feat.GetGeometryRef()
#             if geom is None:
#                 continue

#             # OGR envelope: (minX, maxX, minY, maxY) — note the axis order
#             ogr_env = geom.GetEnvelope()
#             geom_env = (ogr_env[0], ogr_env[2], ogr_env[1], ogr_env[3])  # -> (minx, miny, maxx, maxy)

#             fid = feat.GetField("block_id")
#             fid = (fid - 1) if fid is not None else None

#             if touches_block_edge(geom_env, block_env, tol):
#                 target_layer = cand_layer
#             else:
#                 target_layer = out_layer

#             out_feat = ogr.Feature(target_layer.GetLayerDefn())
#             out_feat.SetGeometry(geom.Clone())
#             out_feat.SetField("block_id", fid)
#             target_layer.CreateFeature(out_feat)
#             out_feat = None

#         ds = None
#         progress.update(task, advance=1)

#     # Flush and close both layers before pass 2
#     out_ds = None
#     cand_ds = None

#     return candidates_path

# =============================================================================
# Driver functions
# =============================================================================
def close_gaps(block, gap_threshold, block_info):
    # Skeletonize
    bool_arr = block.astype(bool)
    skeleton = skeletonize(bool_arr).astype(np.uint8)
    if not skeleton.any():
        log.debug("skeletonize failed, thresholded array is empty")

    # Find endpoints
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(skeleton, kernel, mode='constant', cval=0)
    
    # Remove pixels with no neighbors
    skeleton[neighbor_count == 1] = 0
    
    # Store endpoints
    endpoints = (skeleton == 1) & (neighbor_count == 2) # Count: itself & neighbor
    ep_coords = list(zip(*np.where(endpoints)))
    result = skeleton.copy()
    if len(ep_coords) < 2:
        log.debug("No skeleton endpoints found, no gaps closed")
        return result
    
    # Get all other pixels
    skel_coords = np.array(list(zip(*np.where(skeleton == 1))))
    tree = cKDTree(skel_coords)

    # Create labeling scheme for connected segments
    inner_mask = np.zeros_like(skeleton, dtype=bool)
    inner_mask[
        block_info.inner_row_offset : block_info.inner_row_offset + block_info.block_height,
        block_info.inner_col_offset : block_info.inner_col_offset + block_info.block_width
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

        indices = tree.query_ball_point([r1, c1], gap_threshold)
        
        # Create segment labels
        l1_buf = labeled_buffer[r1, c1]
        l1_inn = labeled_inner[r1, c1]
        
        for idx in indices:
            r2, c2 = skel_coords[idx]
            
            # Segment labels
            l2_buf = labeled_buffer[r2, c2]
            l2_inn = labeled_inner[r2, c2]
            
            # Skip only if connected in BOTH labelings
            if l1_inn == l2_inn and l1_buf == l2_buf:
                continue

            dist = np.hypot(r2 - r1, c2 - c1)
            if dist < best_dist:
                best_dist = dist
                best_coord = (r2, c2)

        if best_coord is not None:
            r2, c2 = best_coord
            rr, cc = draw_line(int(r1), int(c1), int(r2), int(c2))
            # Clip to array bounds just in case
            valid = (rr >= 0) & (rr < skeleton.shape[0]) & \
                    (cc >= 0) & (cc < skeleton.shape[1])
            result[rr[valid], cc[valid]] = 1
    
    #* ---DEBUG---            
    # return result
    return result, skeleton

def find_enclosed_polygons(closed_lines: np.ndarray, output_dir: str, block_info: BlockInfo, raster_info: RasterInfo, min_area):
    # Get interior pixels within block & buffer
    interior_buffer = get_interior(closed_lines, min_area)
    
    # Crop this to the block extent (removing buffer)
    r0 = block_info.inner_row_offset
    c0 = block_info.inner_col_offset
    interior_block = interior_buffer[r0:r0 + block_info.block_height,  # Top and bottom rows
                              c0:c0 + block_info.block_width]   # Left and right columns
    
    if interior_block.max() == 0:
        return 0
    
    # Store block id + 1 because blocks are 0-indexed
    interior_block[interior_block == 1] = block_info.block_id + 1
    
    # Create raster dataset in memory for polygonization, 
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
    mem_band.WriteArray(interior_block)
        
    # Create blank vector dataset in memory for polygonization
    block_polygons_path = os.path.join(output_dir, f"{block_info.block_id}_polygons.fgb")
    block_polygons = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(block_polygons_path)
    polygons_layer = block_polygons.CreateLayer("enclosed_polygons", srs=raster_info.projection, geom_type=ogr.wkbPolygon)
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
def extract_polygons(block_id, output_dir, input_raster_path, blocksize, buffer_dist, gap_threshold, prob_threshold, min_area):
    # Check for buffer distance/gap threshold tolerance
    if buffer_dist < gap_threshold:
        log.warning(
            f"Buffer ({buffer_dist}) < gap threshold ({gap_threshold})."
            "Edge artifacts are likely. Consider setting buffer distance >= gap threshold")
    
    # Step 1: load block
    block, block_info, raster_info = load_block(block_id, input_raster_path, blocksize, buffer_dist, prob_threshold)
    
    #* ---DEBUG--- Step 2: close gaps
    # closed_lines = close_gaps(block, gap_threshold)
    closed_lines, skeleton = close_gaps(block, gap_threshold, block_info)
    
    # Step 3: find enclosed polygons
    polygons = find_enclosed_polygons(closed_lines, output_dir, block_info, raster_info, min_area)
    
    #* ---DEBUG---
    block_info_path = os.path.join(OUTPUT_DIR, "block_info.fgb")
    skeleton_path = os.path.join(OUTPUT_DIR, "skeleton.tif")
    closed_lines_path = os.path.join(OUTPUT_DIR, "closed_lines.tif")
    write_block_debug(block_info, raster_info, output_path=block_info_path)
    
    write_col = block_info.buf_col
    write_row = block_info.buf_row
    write_width = block_info.buf_width
    write_height = block_info.buf_height
    gt = list(raster_info.transform)
    gt[0] = raster_info.xmin + write_col * raster_info.pixel_width   # x origin
    gt[3] = raster_info.ymax + write_row * raster_info.pixel_height  # y origin

    skeleton_ds = gdal.GetDriverByName("GTiff").Create(
        skeleton_path, write_width, write_height, 1, gdal.GDT_Byte
    )
    skeleton_ds.SetGeoTransform(gt)
    skeleton_ds.SetProjection(raster_info.projection.ExportToWkt())
    skeleton_band = skeleton_ds.GetRasterBand(1)
    skeleton_band.SetNoDataValue(0)
    skeleton_band.WriteArray(skeleton)
    
    closed_ds = gdal.GetDriverByName("GTiff").Create(
        closed_lines_path, write_width, write_height, 1, gdal.GDT_Byte
    )
    closed_ds.SetGeoTransform(gt)
    closed_ds.SetProjection(raster_info.projection.ExportToWkt())
    closed_band = closed_ds.GetRasterBand(1)
    closed_band.SetNoDataValue(0)
    closed_band.WriteArray(closed_lines)
    
    skeleton_ds = None
    closed_ds = None
    
    if polygons == 0:
        return {
            "block_id": block_id,
            "status": "null",
            "polygon_count": 0,
            "polygons": polygons
        }
    
    return {
        "block_id": block_id,
        "status": "success",
        "polygon_count": polygons,
        "polygons": polygons
    }
    

# =============================================================================
# MAIN
# =============================================================================
def main():
    console.print(f"Processing block {BLOCK_ID}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = extract_polygons(BLOCK_ID, OUTPUT_DIR, INPUT_RASTER, BLOCKSIZE, BUFFER_DIST, GAP_THRESHOLD, PROBABILITY_THRESHOLD, MIN_AREA)
    console.print(result)
    
if __name__ == "__main__":
    main()
    