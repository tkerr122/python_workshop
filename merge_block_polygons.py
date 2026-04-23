# Imports
from osgeo import gdal, ogr, osr
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
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
INPUT_RASTER = "/gpfs/glad1/Theo/Data/Pastures_test/v1_test_lines.tif"
OUTPUT_DIR = f"/gpfs/glad1/Theo/Data/Pastures_test/test_output"
N_WORKERS = 40      # Number of CPUs to use
BLOCKSIZE = 2048     # Size of the block
BUFFER_DIST = 2048   # Size of the surrounding buffer
GAP_THRESHOLD = 40   # Maximum size of gaps to close (in pixels)
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
        
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self):
        """Returns dict mapping root -> list of members."""
        groups = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups        

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

def get_block_envelope(block_info: BlockInfo, raster_info: RasterInfo) -> tuple:
    # Get the block bounds in relation to the raster, not including the buffer
    xmin = raster_info.xmin + block_info.block_col * raster_info.pixel_width
    xmax = xmin + block_info.block_width * raster_info.pixel_width
    ymax = raster_info.ymax + block_info.block_row * raster_info.pixel_height # top edge since pixel_height is negative
    ymin = ymax + block_info.block_height * raster_info.pixel_height
    
    return (xmin, ymin, xmax, ymax)

def touches_block_edge(geom_env: tuple, block_env: tuple, tol: float) -> bool:
    """
    geom_env:  (minx, miny, maxx, maxy) from OGR geometry envelope
    block_env: (xmin, ymin, xmax, ymax) of the true block
    tol:       tolerance in map units (e.g. 1 pixel width)
    """
    gminx, gminy, gmaxx, gmaxy = geom_env
    bminx, bminy, bmaxx, bmaxy = block_env
    return (
        abs(gminx - bminx) < tol or
        abs(gminy - bminy) < tol or
        abs(gmaxx - bmaxx) < tol or
        abs(gmaxy - bmaxy) < tol
    )

def envelopes_touch(env_a: tuple, env_b: tuple, tol: float) -> bool:
    """
    Fast pre-filter: checks if two bounding boxes are within tol of each other.
    env: (minx, miny, maxx, maxy)
    """
    aminx, aminy, amaxx, amaxy = env_a
    bminx, bminy, bmaxx, bmaxy = env_b
    return (
        aminx <= bmaxx + tol and
        amaxx >= bminx - tol and
        aminy <= bmaxy + tol and
        amaxy >= bminy - tol
    )

def get_window_candidates(cand_layer, window_env: tuple) -> list:
    """
    Spatially filter candidates layer to the window extent,
    returning list of (fid, geometry, block_id, envelope) tuples.
    """
    minx, miny, maxx, maxy = window_env
    cand_layer.SetSpatialFilterRect(minx, miny, maxx, maxy)
    features = []
    
    for feat in cand_layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        ogr_env = geom.GetEnvelope()
        env = (ogr_env[0], ogr_env[2], ogr_env[1], ogr_env[3])  # -> (minx, miny, maxx, maxy)
        features.append((
            feat.GetFID(),
            geom.Clone(),
            feat.GetField("block_id"),
            env
        ))
    cand_layer.SetSpatialFilter(None)
    
    return features

def ensure_polygon(geom):
    """
    Forces a geometry to wkbPolygon or wkbMultiPolygon.
    If Union produces a GeometryCollection, extracts polygon parts.
    """
    geom_type = geom.GetGeometryType()
    if geom_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return geom
    if geom_type == ogr.wkbGeometryCollection:
        multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for i in range(geom.GetGeometryCount()):
            part = geom.GetGeometryRef(i)
            if part.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                multi.AddGeometry(part)
        return multi if multi.GetGeometryCount() > 0 else geom
    return geom

def wrap_as_multi(geom):
    """Wraps a Polygon in a MultiPolygon for schema consistency."""
    if geom.GetGeometryType() == ogr.wkbPolygon:
        multi = ogr.Geometry(ogr.wkbMultiPolygon)
        multi.AddGeometry(geom)
        return multi
    return geom

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
    
    return result

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
    
    # Step 2: close gaps
    closed_lines = close_gaps(block, gap_threshold, block_info)
    
    # Step 3: find enclosed polygons
    polygons = find_enclosed_polygons(closed_lines, output_dir, block_info, raster_info, min_area)
    
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
# Merge vectors 
# =============================================================================
def merge_vectors(input_dir, output_path, raster_info: RasterInfo, blocksize, buffer_dist, progress):
    # ---------------------------------
    # PASS 1: get touching polygons
    # ---------------------------------
    task1 = progress.add_task("Finding touching polygons...", total=len(os.listdir(input_dir)))
    tol = abs(raster_info.pixel_width)  # 1-pixel tolerance for edge detection

    # Create output layers
    out_driver = ogr.GetDriverByName("FlatGeobuf")
    out_ds = out_driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("merged_polygons", srs=raster_info.projection, geom_type=ogr.wkbPolygon)
    out_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTInteger))
    
    candidates_path = os.path.join(os.path.dirname(output_path), "_candidates.fgb")
    cand_ds = out_driver.CreateDataSource(candidates_path)
    cand_layer = cand_ds.CreateLayer("candidates", srs=raster_info.projection, geom_type=ogr.wkbPolygon)
    cand_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTInteger))

    # Merge vectors
    for file in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        
        # Get block_id from filename
        block_id = int(os.path.splitext(file)[0].rsplit("_")[0])        

        # Get block raster_info & envelope
        block_info = get_block_info(block_id, raster_info, blocksize, buffer_dist)
        block_env = get_block_envelope(block_info, raster_info)

        # Load block and its polygons
        ds = ogr.Open(filepath)
        lyr = ds.GetLayer()

        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue

            # OGR envelope: (minX, maxX, minY, maxY) — note the axis order
            ogr_env = geom.GetEnvelope()
            geom_env = (ogr_env[0], ogr_env[2], ogr_env[1], ogr_env[3])  # -> (minx, miny, maxx, maxy)

            fid = feat.GetField("block_id")
            fid = (fid - 1) if fid is not None else None

            if touches_block_edge(geom_env, block_env, tol):
                target_layer = cand_layer
            else:
                target_layer = out_layer

            out_feat = ogr.Feature(target_layer.GetLayerDefn())
            out_feat.SetGeometry(geom.Clone())
            out_feat.SetField("block_id", fid)
            target_layer.CreateFeature(out_feat)
            out_feat = None

        ds = None
        progress.update(task1, advance=1)

    # Flush and close both layers before pass 2
    out_ds = None
    cand_ds = None
    
    # ---------------------------------
    # PASS 2: Merge touching polygons
    # ---------------------------------
    n_blocks_x = math.ceil(raster_info.cols / blocksize)
    n_blocks_y = math.ceil(raster_info.rows / blocksize)
    window_size = 5
    stride = window_size - 1  # 1-block overlap
    
    # Get total number of windows for progress meter
    n_windows_x = math.ceil(n_blocks_x / stride)
    n_windows_y = math.ceil(n_blocks_y / stride)
    total_windows = n_windows_x * n_windows_y
    task2 = progress.add_task("Merging touching polygons by window...", total=total_windows)

    # Load candidates layer
    cand_ds = ogr.Open(candidates_path)
    cand_layer = cand_ds.GetLayer()

    # Persistent union-find and geometry/block_id stores
    uf = UnionFind()
    geom_store = {}     # fid -> geometry
    blkid_store = {}    # fid -> block_id (integer)

    # Sweep windows 
    for row_start in range(0, n_blocks_y, stride):
        for col_start in range(0, n_blocks_x, stride):
            row_end = min(row_start + window_size, n_blocks_y)
            col_end = min(col_start + window_size, n_blocks_x)

            # Geographic extent of this window (in true block coordinates)
            win_xmin = raster_info.xmin + col_start * blocksize * raster_info.pixel_width
            win_xmax = raster_info.xmin + col_end   * blocksize * raster_info.pixel_width
            win_ymax = raster_info.ymax + row_start * blocksize * raster_info.pixel_height  # pixel_height negative
            win_ymin = raster_info.ymax + row_end   * blocksize * raster_info.pixel_height
            window_env = (win_xmin, win_ymin, win_xmax, win_ymax)

            # Fetch candidates in this window
            window_feats = get_window_candidates(cand_layer, window_env)
            if not window_feats:
                progress.update(task2, advance=1)
                continue

            # Register all features with union-find and stores
            for fid, geom, block_id, env in window_feats:
                uf.add(fid)
                if fid not in geom_store:
                    geom_store[fid] = geom
                    blkid_store[fid] = block_id

            # Build adjacency: pre-filter by envelope, confirm with .Touches()
            for i in range(len(window_feats)):
                fid_a, geom_a, _, env_a = window_feats[i]
                for j in range(i + 1, len(window_feats)):
                    fid_b, geom_b, _, env_b = window_feats[j]
                    if uf.find(fid_a) == uf.find(fid_b):
                        continue  # already in same component
                    if not envelopes_touch(env_a, env_b, tol):
                        continue  # fast reject
                    if geom_a.Touches(geom_b):
                        uf.union(fid_a, fid_b)

            progress.update(task2, advance=1)

    # ---------------------------------
    # PASS 3: write features to output layer
    # ---------------------------------
    final_output = output_path.replace(".fgb", "_pass2.fgb")
    final_ds = out_driver.CreateDataSource(final_output)
    final_layer = final_ds.CreateLayer("merged_polygons", srs=raster_info.projection, geom_type=ogr.wkbMultiPolygon)
    final_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTString))

    # Copy interior polygons from pass 1 output, converting block_id to string
    existing_ds = ogr.Open(output_path)
    existing_layer = existing_ds.GetLayer()
    feat_count = existing_layer.GetFeatureCount()
    task3 = progress.add_task("Writing pass 1 polygons to disk...", total=feat_count)
    for feat in existing_layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        out_feat = ogr.Feature(final_layer.GetLayerDefn())
        out_feat.SetGeometry(wrap_as_multi(geom.Clone()))
        out_feat.SetField("block_id", str(feat.GetField("block_id")))
        final_layer.CreateFeature(out_feat)
        out_feat = None
        
        progress.update(task3, advance=1)
        
    existing_ds = None

    # Write dissolved candidate components
    components = uf.components()
    task4 = progress.add_task("Writing merged polygons to disk...", total=len(components))
    for root, members in components.items():
        out_feat = ogr.Feature(final_layer.GetLayerDefn())
        if len(members) == 1:
            # No merge needed - write as-is
            fid = members[0]
            geom = wrap_as_multi(geom_store[fid])
            out_feat.SetGeometry(geom)
            out_feat.SetField("block_id", str(blkid_store[fid]))
        else:
            # Dissolve component geometries
            dissolved = geom_store[members[0]]
            for fid in members[1:]:
                dissolved = dissolved.Union(geom_store[fid])
            dissolved = ensure_polygon(dissolved)
            dissolved = wrap_as_multi(dissolved)

            # Build sorted block_id string
            block_ids = sorted(set(
                blkid_store[fid] for fid in members
                if blkid_store[fid] is not None
            ))
            block_id_str = ",".join(str(bid) for bid in block_ids)

            out_feat.SetGeometry(dissolved)
            out_feat.SetField("block_id", block_id_str)
        
        final_layer.CreateFeature(out_feat)
        out_feat = None
        progress.update(task4, advance=1)

    final_ds = None
    cand_ds = None

    # Replace original output with pass 2 result
    os.replace(final_output, output_path)    

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
        (block_id, output_block_dir, INPUT_RASTER, BLOCKSIZE, BUFFER_DIST, GAP_THRESHOLD, PROBABILITY_THRESHOLD, MIN_AREA)
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
            merge_vectors(output_block_dir, output_vector_path, raster_info, BLOCKSIZE, BUFFER_DIST, progress)
        
    console.print("All polygons merged")
    
    
if __name__ == "__main__":
    main()
    