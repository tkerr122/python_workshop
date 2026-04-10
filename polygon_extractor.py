"""
Block-based raster processing: find polygons enclosed by linear features.

Pipeline per block:
  1. load_block()     - read block + neighbor buffer from input raster
  2. close_gaps()     - skeletonize + binary closing to bridge line gaps
  3. find_polygons()  - label enclosed regions within the closed lines
  4. write_block()    - clip result back to the true block extent and write
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockInfo:
    """All geometry needed to read/write one block with an optional buffer."""

    # True block (no buffer) — pixel offsets in the full raster
    block_col: int          # x offset of the block
    block_row: int          # y offset of the block
    block_width: int        # pixel columns in the block
    block_height: int       # pixel rows in the block

    # Buffered region — what we actually read from disk
    buf_col: int            # x offset of the buffered read
    buf_row: int            # y offset of the buffered read
    buf_width: int          # pixel columns actually read
    buf_height: int         # pixel rows actually read

    # How far the true block is from the buffered origin
    inner_col_offset: int   # = block_col - buf_col
    inner_row_offset: int   # = block_row - buf_row


def compute_block_info(
    block_id: int,
    raster_width: int,
    raster_height: int,
    blocksize: int = 512,
    buffer: int = 0,
) -> BlockInfo:
    """
    Compute pixel offsets for a block (identified by its flat index) and an
    optional surrounding buffer that is clamped to the raster extent.

    Blocks tile the raster in row-major order:
        block_id = row_index * n_cols_of_blocks + col_index

    Parameters
    ----------
    block_id      : flat (row-major) block index
    raster_width  : full raster width  in pixels
    raster_height : full raster height in pixels
    blocksize     : nominal block side length in pixels
    buffer        : extra pixels to read around the true block on each side
                    (clamped so we never read outside the raster)
    """
    n_blocks_x = math.ceil(raster_width  / blocksize)
    n_blocks_y = math.ceil(raster_height / blocksize)

    block_row_idx = block_id // n_blocks_x
    block_col_idx = block_id  % n_blocks_x

    if block_row_idx >= n_blocks_y:
        raise ValueError(f"block_id {block_id} exceeds raster extent.")

    # True block extents (edge blocks may be smaller than blocksize)
    block_col    = block_col_idx * blocksize
    block_row    = block_row_idx * blocksize
    block_width  = min(blocksize, raster_width  - block_col)
    block_height = min(blocksize, raster_height - block_row)

    # Buffered extents — clamp to [0, raster dimension)
    buf_col    = max(0, block_col - buffer)
    buf_row    = max(0, block_row - buffer)
    buf_right  = min(raster_width,  block_col + block_width  + buffer)
    buf_bottom = min(raster_height, block_row + block_height + buffer)
    buf_width  = buf_right  - buf_col
    buf_height = buf_bottom - buf_row

    # Offset of the true block inside the buffered array
    inner_col_offset = block_col - buf_col
    inner_row_offset = block_row - buf_row

    return BlockInfo(
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


# ---------------------------------------------------------------------------
# Step 1 – load block (with buffer)
# ---------------------------------------------------------------------------

def load_block(
    input_raster: str,
    block_id: int,
    blocksize: int = 512,
    buffer: int = 64,
    band_index: int = 1,
) -> tuple[np.ndarray, BlockInfo]:
    """
    Read a buffered block from *input_raster*.

    Returns
    -------
    array     : 2-D uint8 array of shape (buf_height, buf_width).
                Values are binarised: non-zero → 1 (line present).
    block_info: BlockInfo describing all offsets for later steps.
    """
    ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {input_raster}")

    band = ds.GetRasterBand(band_index)
    raster_width  = ds.RasterXSize
    raster_height = ds.RasterYSize

    bi = compute_block_info(block_id, raster_width, raster_height, blocksize, buffer)

    # Single ReadAsArray call covering the buffered extent
    array = band.ReadAsArray(
        bi.buf_col,    # xoff
        bi.buf_row,    # yoff
        bi.buf_width,  # xsize
        bi.buf_height, # ysize
    )

    # Binarise: treat any non-zero pixel as a line pixel
    array = (array != 0).astype(np.uint8)

    ds = None  # close dataset
    return array, bi


# ---------------------------------------------------------------------------
# Step 2 – close gaps in linear features
# ---------------------------------------------------------------------------

def close_gaps(
    input_array: np.ndarray,
    gap_threshold: int = 5,
    skeletonize_first: bool = True,
) -> np.ndarray:
    """
    Close small gaps in binary linear features.

    Strategy
    --------
    1. (Optional) Skeletonize to thin multi-pixel-wide lines to 1-pixel width.
       This prevents dilation from merging features that are close but not
       connected.
    2. Binary dilation + erosion (= morphological closing) with a structuring
       element whose reach equals *gap_threshold* pixels.  This bridges gaps
       up to ~2 × gap_threshold pixels wide.
    3. Re-skeletonize to restore thin lines.

    Parameters
    ----------
    input_array    : binary array (1 = line, 0 = background)
    gap_threshold  : maximum gap in pixels to close (half the closing radius)
    skeletonize_first : thin lines before closing (recommended for linear
                        features to prevent merging nearby parallel lines)

    Returns
    -------
    closed : binary uint8 array with gaps filled
    """
    arr = input_array.astype(bool)

    # --- 1. Optional pre-skeletonize ---
    if skeletonize_first:
        arr = skeletonize(arr)

    # --- 2. Morphological closing ---
    # Build a disk-like structuring element of radius gap_threshold
    structure = ndimage.generate_binary_structure(2, 1)          # 4-connected cross
    structure = ndimage.iterate_structure(structure, gap_threshold)  # expand to radius

    closed = ndimage.binary_closing(arr, structure=structure)

    # --- 3. Post-skeletonize to restore 1-pixel-wide lines ---
    closed = skeletonize(closed)

    return closed.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3 – find enclosed polygons and write to vector layer
# ---------------------------------------------------------------------------

def _interior_mask(closed_lines: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of pixels that are background AND fully enclosed
    by lines (i.e. not reachable from any raster border).

    Strategy: label all background connected components, then discard any
    component whose pixels touch the array border.  What remains is interior.
    """
    background = (closed_lines == 0)
    labeled_bg, _ = ndimage.label(background)

    border_labels: set[int] = set()
    for edge in (labeled_bg[0, :], labeled_bg[-1, :],
                 labeled_bg[:, 0], labeled_bg[:, -1]):
        border_labels.update(edge.flat)
    border_labels.discard(0)  # 0 == line pixels, not a background region

    interior = np.isin(labeled_bg, list(border_labels), invert=True) & background
    return interior.astype(np.uint8)


def find_enclosed_polygons(
    closed_lines: np.ndarray,
    bi: BlockInfo,
    geotransform: tuple,
    projection: str,
    out_layer: ogr.Layer,
) -> int:
    """
    Find pixels enclosed by lines, strip the buffer, convert to vector
    polygons via ``gdal.Polygonize``, and append them to *out_layer*.

    The buffer strip happens in raster space before polygonization so that
    polygons straddling the true block boundary are not emitted — they will
    be produced correctly by the adjacent block which shares that edge.

    Parameters
    ----------
    closed_lines : binary uint8 array covering the *buffered* extent
    bi           : BlockInfo for this block (carries buffer offsets)
    geotransform : 6-element GDAL geotransform of the *full* input raster
    projection   : WKT projection string of the input raster
    out_layer    : open, writable OGR layer to append features into

    Returns
    -------
    n_polygons : number of polygons written for this block
    """
    # --- 1. Compute interior mask over the buffered array ---
    interior_buf = _interior_mask(closed_lines)

    # --- 2. Crop to the true block extent (strip buffer) ---
    r0 = bi.inner_row_offset
    c0 = bi.inner_col_offset
    interior_block = interior_buf[r0:r0 + bi.block_height,
                                  c0:c0 + bi.block_width]

    if interior_block.max() == 0:
        return 0  # no enclosed pixels in this block

    # --- 3. Build an in-memory raster for gdal.Polygonize ---
    # Adjust geotransform origin so coordinates map to the true block position
    gt = list(geotransform)
    gt[0] = geotransform[0] + bi.block_col * geotransform[1]   # x origin
    gt[3] = geotransform[3] + bi.block_row * geotransform[5]   # y origin

    mem_driver = gdal.GetDriverByName("MEM")
    mem_ds = mem_driver.Create(
        "", bi.block_width, bi.block_height, 1, gdal.GDT_Byte
    )
    mem_ds.SetGeoTransform(gt)
    mem_ds.SetProjection(projection)

    mem_band = mem_ds.GetRasterBand(1)
    mem_band.WriteArray(interior_block)
    mem_band.SetNoDataValue(0)

    # --- 4. Polygonize: raster blobs → vector polygons ---
    # Pass mem_band as the mask so only pixels == 1 produce polygons
    n_before = out_layer.GetFeatureCount()
    gdal.Polygonize(mem_band, mem_band, out_layer, 0, [], callback=None)
    n_polygons = out_layer.GetFeatureCount() - n_before

    mem_ds = None
    return n_polygons


# ---------------------------------------------------------------------------
# Step 4 – vector output helpers (replaces raster write_block)
# ---------------------------------------------------------------------------

def create_output_vector(
    output_vector: str,
    projection_wkt: str,
    driver_name: str = "GPKG",
    layer_name: str = "enclosed_polygons",
) -> tuple[ogr.DataSource, ogr.Layer]:
    """
    Create (or overwrite) a vector dataset and return the open datasource
    and its single polygon layer.

    Parameters
    ----------
    output_vector  : file path (e.g. "output.gpkg" or "output.shp")
    projection_wkt : WKT CRS string copied from the input raster
    driver_name    : OGR driver; "GPKG" (default) or "ESRI Shapefile", etc.
    layer_name     : name for the layer inside the dataset

    Returns
    -------
    (datasource, layer) — both remain open; caller must set to None when done.
    """
    drv = ogr.GetDriverByName(driver_name)
    if drv is None:
        raise RuntimeError(f"OGR driver '{driver_name}' not available.")

    import os
    if os.path.exists(output_vector):
        drv.DeleteDataSource(output_vector)

    ds = drv.CreateDataSource(output_vector)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_wkt)

    layer = ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    # Add a simple integer id field
    field_defn = ogr.FieldDefn("block_id", ogr.OFTInteger)
    layer.CreateField(field_defn)

    return ds, layer


def n_blocks(input_raster: str, blocksize: int = 512) -> int:
    """Total number of blocks in the raster."""
    ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    nx = math.ceil(ds.RasterXSize / blocksize)
    ny = math.ceil(ds.RasterYSize / blocksize)
    ds = None
    return nx * ny


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_raster(
    input_raster: str,
    output_vector: str,
    blocksize: int = 512,
    buffer: int = 64,
    gap_threshold: int = 5,
    band_index: int = 1,
    ogr_driver: str = "GPKG",
) -> None:
    """
    End-to-end pipeline: read → close gaps → polygonize enclosed regions →
    append to vector dataset.

    Each block is processed with a surrounding pixel buffer so that
    gap-closing works correctly near block boundaries.  Only polygons whose
    centroids fall inside the true block extent are written, preventing
    double-counting at block edges.

    Parameters
    ----------
    input_raster  : path to the input raster (linear features)
    output_vector : path for the output vector file (e.g. "out.gpkg")
    blocksize     : tile size in pixels (square)
    buffer        : extra pixels around the block for gap-closing context;
                    should be ≥ gap_threshold to avoid edge artefacts
    gap_threshold : maximum gap in pixels to close
    band_index    : 1-based band index to read from the input
    ogr_driver    : OGR driver name; "GPKG" or "ESRI Shapefile" etc.
    """
    import warnings

    if buffer < gap_threshold:
        warnings.warn(
            f"buffer ({buffer}) < gap_threshold ({gap_threshold}). "
            "Edge artefacts are likely. Consider setting buffer ≥ gap_threshold.",
            stacklevel=2,
        )

    # Read raster metadata once
    src_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    geotransform = src_ds.GetGeoTransform()
    projection   = src_ds.GetProjection()
    src_ds = None

    out_ds, out_layer = create_output_vector(output_vector, projection, ogr_driver)
    total = n_blocks(input_raster, blocksize)
    total_polys = 0

    for block_id in range(total):
        # 1. Load buffered block
        buffered_array, bi = load_block(
            input_raster, block_id, blocksize=blocksize,
            buffer=buffer, band_index=band_index,
        )

        # 2. Close gaps in the buffered array
        closed = close_gaps(buffered_array, gap_threshold=gap_threshold)

        # 3. Polygonize enclosed regions; appends directly to out_layer
        n = find_enclosed_polygons(
            closed, bi, geotransform, projection, out_layer
        )
        total_polys += n
        print(f"  block {block_id + 1}/{total}  polygons this block: {n}", end="\r", flush=True)

    out_ds = None  # flush and close
    print(f"\nDone. {total_polys} polygons written to: {output_vector}")


# ---------------------------------------------------------------------------
# Parallelisation drop-in
# ---------------------------------------------------------------------------

def _process_one_block(args: tuple) -> list:
    """
    Worker: process one block and return a list of WKT polygon strings
    (with their block_id) so the main process can write them serially.

    Vector datasets are NOT thread/process safe for concurrent writes, so
    workers return geometries as WKT and the collector writes them.
    """
    input_raster, block_id, blocksize, buffer, gap_threshold, band_index = args

    src_ds      = gdal.Open(input_raster, gdal.GA_ReadOnly)
    geotransform = src_ds.GetGeoTransform()
    projection   = src_ds.GetProjection()
    src_ds = None

    buffered_array, bi = load_block(input_raster, block_id, blocksize, buffer, band_index)
    closed = close_gaps(buffered_array, gap_threshold)

    # Polygonize into a throw-away in-memory OGR layer, then harvest WKT
    mem_drv   = ogr.GetDriverByName("Memory")
    mem_ogr   = mem_drv.CreateDataSource("")
    srs       = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    tmp_layer = mem_ogr.CreateLayer("tmp", srs=srs, geom_type=ogr.wkbPolygon)
    tmp_layer.CreateField(ogr.FieldDefn("block_id", ogr.OFTInteger))

    find_enclosed_polygons(closed, bi, geotransform, projection, tmp_layer)

    results = []
    for feat in tmp_layer:
        geom = feat.GetGeometryRef()
        if geom:
            results.append((block_id, geom.ExportToWkt()))

    mem_ogr = None
    return results


def process_raster_parallel(
    input_raster: str,
    output_vector: str,
    blocksize: int = 512,
    buffer: int = 64,
    gap_threshold: int = 5,
    band_index: int = 1,
    n_workers: int = 4,
    ogr_driver: str = "GPKG",
) -> None:
    """
    Parallel version of process_raster().

    Workers are embarrassingly parallel (read + compute only).
    The main process collects WKT results and writes them serially to avoid
    concurrent OGR write conflicts.
    """
    from multiprocessing import Pool

    src_ds     = gdal.Open(input_raster, gdal.GA_ReadOnly)
    projection = src_ds.GetProjection()
    src_ds = None

    out_ds, out_layer = create_output_vector(output_vector, projection, ogr_driver)
    total = n_blocks(input_raster, blocksize)

    args = [
        (input_raster, bid, blocksize, buffer, gap_threshold, band_index)
        for bid in range(total)
    ]

    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    total_polys = 0
    with Pool(n_workers) as pool:
        for block_results in pool.imap_unordered(_process_one_block, args):
            for block_id, wkt in block_results:
                feat = ogr.Feature(out_layer.GetLayerDefn())
                feat.SetField("block_id", block_id)
                feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt, srs))
                out_layer.CreateFeature(feat)
                total_polys += 1

    out_ds = None
    print(f"Parallel processing done. {total_polys} polygons written to: {output_vector}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find polygons enclosed by linear raster features."
    )
    parser.add_argument("input",  help="Input raster (linear features)")
    parser.add_argument("output", help="Output vector file (e.g. out.gpkg)")
    parser.add_argument("--blocksize",     type=int, default=512)
    parser.add_argument("--buffer",        type=int, default=64,
                        help="Buffer pixels around each block (≥ gap_threshold)")
    parser.add_argument("--gap_threshold", type=int, default=5,
                        help="Max gap in pixels to close")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of parallel workers (1 = serial)")
    args = parser.parse_args()

    if args.workers > 1:
        process_raster_parallel(
            args.input, args.output,
            blocksize=args.blocksize,
            buffer=args.buffer,
            gap_threshold=args.gap_threshold,
            n_workers=args.workers,
        )
    else:
        process_raster(
            args.input, args.output,
            blocksize=args.blocksize,
            buffer=args.buffer,
            gap_threshold=args.gap_threshold,
        )
