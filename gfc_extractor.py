# Theo Kerr

# Imports/env settings
from shapely.geometry import box
from scipy import stats
from osgeo import gdal, ogr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import numpy as np
import geopandas as gpd
import os, logging, math
console = Console()
gdal.UseExceptions()

# =============================================================================
# GLOBALS
# =============================================================================
GFC_TILES = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_tiles"
TRAINING_SHP = "/gpfs/glad1/Theo/Shapefiles/GFC/gfc_training.shp"
OUTPUT_DIR = "/gpfs/glad1/Theo/Data/Global_Forest_Change/output_training_4_6_2026"
N_WORKERS = 150

# -----------------------------------------------------------------------------
# Logging setup using Rich
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            log_time_format="[%H:%M:%S]"
        ),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_gfc_extractor.log"),
            mode="w"
        )
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# Utility functions
# =============================================================================
def get_raster_info(raster_path):
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    transform = ds.GetGeoTransform()
    projection = ds.GetSpatialRef()
    
    info = {"cols": cols,
            "rows": rows,
            "transform": transform,
            "projection": projection,        
            "xmin": transform[0], 
            "ymax": transform[3],
            "xmax": transform[0] + transform[1] * cols,
            "ymin": transform[3] + transform[5] * rows
    }
    
    return ds, info

def compute_mode_nonzero(pixels):
    change_pixels = pixels[pixels != 0]
    if change_pixels.size == 0:
        return None
    
    mode_result = stats.mode(change_pixels, keepdims=False)
    
    log.debug(f"Compute_mode: mode = {int(mode_result.mode)}")
    return int(mode_result.mode)

def classify_change(year_start, year_end):
    # Setup: treat pandas NA, None, and float NaN uniformly
    start_null = (year_start is None) or (
        isinstance(year_start, float) and np.isnan(year_start)
    )
    end_null = (year_end is None) or (
        isinstance(year_end, float) and np.isnan(year_end)
    )
    
    # Case 1: change is natural, need to extract mode year of change
    if start_null and end_null:
        return "null_both"

    # Case 2: change is manmade, need to extract all change
    if year_start == 100 and year_end == 100:
        return "extract_all"
    
    # Case 3: change is natural, need to extract specific years
    if year_start < 100 and year_end < 100:
        return "natural"
    
    # Case 4: change is manmade, need to extract specific years
    if year_start > 100 and year_end > 100:
        return "manmade"
    
    return "skip"
    
def sort_tiles(tiles_dir, lat_dir, lon_dir):
    # Get NW tiles
    tiles = []
    for lat in range(0, 90, 10):
        for lon in range(0, 190, 10):
            # Format lat and lon
            lat_str=f"{lat:02d}{lat_dir}"
            lon_str=f"{lon:03d}{lon_dir}"
            
            # Construct filepath
            path = os.path.join(tiles_dir, f"Hansen_GFC-2024-v1.12_lossyear_{lat_str}_{lon_str}_change.tif")
            if os.path.isfile(path) == False:
                continue
            
            # Add to list
            tiles.append(path)
            
    return sorted(tiles)
        
def merge_output(input_tiles, output_tiff):
    # Build VRT
    vrt_path = output_tiff.replace(".tif", ".vrt")
    gdal.BuildVRT(vrt_path, input_tiles)
    
    # Merge rasters
    gdal.Translate(output_tiff, vrt_path, creationOptions=[
        "COMPRESS=LZW",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "BIGTIFF=YES",
        "NUM_THREADS=25"
    ])
    

# =============================================================================
# Get pixels that the polygon overlaps
# =============================================================================
def get_pixels(raster_path, polygon_wkt):
    # Load raster, get polygon bounds
    raster, raster_info = get_raster_info(raster_path)
    if raster is None:
        raise IOError(f"GDAL could not open raster: {raster_path}")
    
    band = raster.GetRasterBand(1)
    geom = ogr.CreateGeometryFromWkt(polygon_wkt)
    env = geom.GetEnvelope()
    poly_x_min, poly_x_max, poly_y_min, poly_y_max = env
    
    # Check overlap, return None if no overlap
    if (poly_x_max < raster_info["xmin"] or poly_x_min > raster_info["xmax"] or
            poly_y_max < raster_info["ymin"] or poly_y_min > raster_info["ymax"]):
        raster = None
        return None
    
    # Step 1: Compute window offsets, clamped to raster extent
    win_x_min = max(poly_x_min, raster_info["xmin"])
    win_x_max = min(poly_x_max, raster_info["xmax"])
    win_y_min = max(poly_y_min, raster_info["ymin"])
    win_y_max = min(poly_y_max, raster_info["ymax"])

    col_off  = math.floor((win_x_min - raster_info["transform"][0]) / raster_info["transform"][1])
    row_off  = math.floor((win_y_max - raster_info["transform"][3]) / raster_info["transform"][5])
    win_cols = max(1, int(np.ceil((win_x_max - win_x_min) / raster_info["transform"][1])))
    win_rows = max(1, int(np.ceil((win_y_max - win_y_min) / abs(raster_info["transform"][5]))))

    col_off  = min(col_off,  raster_info["cols"] - 1)
    row_off  = min(row_off,  raster_info["rows"] - 1)
    win_cols = min(win_cols, raster_info["cols"] - col_off)
    win_rows = min(win_rows, raster_info["rows"] - row_off)

    pixel_array = band.ReadAsArray(col_off, row_off, win_cols, win_rows)
    raster = None
    
    if pixel_array is None:
        return None
    
    # Step 2: create temp raster as template to burn the polygon into
    win_x_origin = raster_info["transform"][0] + col_off * raster_info["transform"][1]
    win_y_origin = raster_info["transform"][3] + row_off * raster_info["transform"][5]
    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", win_cols, win_rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform((win_x_origin, raster_info["transform"][1], 0, win_y_origin, 0, raster_info["transform"][5]))
    mask_ds.SetProjection(raster_info["projection"].ExportToWkt())

    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(0)

    # Step 3: create an OGR layer from polygon
    ogr_mem_driver = ogr.GetDriverByName("Memory")
    ogr_ds = ogr_mem_driver.CreateDataSource("memdata")
    ogr_layer = ogr_ds.CreateLayer("poly", srs=raster_info["projection"])
    feature = ogr.Feature(ogr_layer.GetLayerDefn())
    feature.SetGeometry(geom)
    ogr_layer.CreateFeature(feature)

    # Step 4: burn the polygon into the temp raster
    gdal.RasterizeLayer(mask_ds, [1], ogr_layer, burn_values=[1])
    polygon_mask = mask_band.ReadAsArray()
    
    mask_ds.FlushCache()
    mask_ds = None
    ogr_ds = None
    
    # Step 5: find where this polygon is in the full raster
    win_rows_idx, win_cols_idx = np.where(polygon_mask == 1)
    abs_col_coords = win_cols_idx + col_off
    abs_row_coords = win_rows_idx + row_off

    pixels = pixel_array[polygon_mask == 1].astype(np.uint8)
    
    return pixels, abs_col_coords, abs_row_coords


# =============================================================================
# Write output for a block
# =============================================================================
def write_block(out_band, all_cols, all_rows, all_values, x_off, y_off, cols, rows):
    # Create a mask for the current section of the raster
    block_mask = (
        (all_cols >= x_off) & (all_cols < x_off + cols) &
        (all_rows >= y_off) & (all_rows < y_off + rows)
    )

    if not block_mask.any():
        return
    
    # Initialize the array with zeros
    block_array = np.zeros((rows, cols), dtype=np.uint8)

    # Localise coordinates to block-space (0-indexed within this block)
    local_cols = all_cols[block_mask] - x_off
    local_rows = all_rows[block_mask] - y_off
    block_array[local_rows, local_cols] = all_values[block_mask].astype(np.uint8)

    # Write positionally: xoff/yoff tell GDAL where in the full raster this belongs
    out_band.WriteArray(block_array, x_off, y_off)


# =============================================================================
# Extract GFC change per tile
# =============================================================================
def extract_gfc_change(gfc_tile_path, training_shp, output_dir):
    # Step 1: get tile info
    tile_name = Path(gfc_tile_path).stem
    _, gfc_info = get_raster_info(gfc_tile_path)

    # Step 2: load polygons that intersect the tile
    tile_bbox = box(gfc_info["xmin"], gfc_info["ymin"], gfc_info["xmax"], gfc_info["ymax"])
    gdf = gpd.read_file(training_shp, bbox=tile_bbox)

    # Reproject if necessary
    tile_epsg = gfc_info["projection"].GetAuthorityCode(None)
    shp_epsg = gdf.crs.to_epsg()

    if str(shp_epsg) != tile_epsg:
        gdf = gdf.to_crs(epsg=int(tile_epsg))
    
    if gdf.empty:
        return {"tile": tile_name, "status": "skipped", "polygon_count": 0}
    
    
    # Step 3: extract pixels under each polygon
    all_cols, all_rows, all_pixels = [], [], []
    for _, row in gdf.iterrows():
        # Get polygon info, export geom to wkt
        year_start = row.get("year_start")
        year_end = row.get("year_end")
        mode = classify_change(year_start, year_end)
        geom_wkt = row.geometry.wkt
        
        # Extract pixels under polygon
        result = get_pixels(gfc_tile_path, geom_wkt)
        if result is None:
            continue
        
        pixels, col_coords, row_coords = result
        if pixels.size == 0:
            continue
        
        # Process the pixels according to the mode
        if mode == "null_both":
            mode_value = compute_mode_nonzero(pixels)
            if mode_value is None:
                continue
            mask = (pixels == mode_value)

            if not mask.any():
                continue
            
            all_cols.append(col_coords[mask])
            all_rows.append(row_coords[mask])
            all_pixels.append(pixels[mask])
            
        elif mode == "extract_all":
            mask = (pixels >= 1) & (pixels <= 24)

            if not mask.any():
                continue
            
            all_cols.append(col_coords[mask])
            all_rows.append(row_coords[mask])
            all_pixels.append(pixels[mask] + 100)
        
        elif mode == "natural":
            mask = (pixels >= year_start) & (pixels <= year_end)

            if not mask.any():
                continue
            
            all_cols.append(col_coords[mask])
            all_rows.append(row_coords[mask])
            all_pixels.append(pixels[mask])
        
        elif mode == "manmade":
            ys = year_start - 100
            ye = year_end - 100
            mask = (pixels >= ys) & (pixels <= ye)

            if not mask.any():
                continue
            
            all_cols.append(col_coords[mask])
            all_rows.append(row_coords[mask])
            all_pixels.append(pixels[mask] + 100)
            
        else: 
            continue
        
    if not all_cols:
        return {"tile": tile_name, "status": "no_change", "polygon_count": len(gdf)}
    
    all_cols   = np.concatenate(all_cols)
    all_rows   = np.concatenate(all_rows)
    all_values = np.concatenate(all_pixels).astype(np.uint8)
    
    # Step 4: set up output tiff
    os.makedirs(output_dir, exist_ok=True)
    output_tiff = os.path.join(output_dir, f"{tile_name}_change.tif")
    output = gdal.GetDriverByName("GTiff").Create(
        output_tiff, gfc_info["cols"], gfc_info["rows"], 1, gdal.GDT_Byte, options=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES"
        ]
    )
    output.SetGeoTransform(gfc_info["transform"])
    output.SetProjection(gfc_info["projection"].ExportToWkt())
    output_band = output.GetRasterBand(1)
    output_band.Fill(255)
    output_band.SetNoDataValue(255)
    
    # Step 5: write out pixels to new raster matching gfc tile extents
    block_size = 512
    for y in range(0, gfc_info["rows"], block_size):
        rows = min(block_size, gfc_info["rows"] - y)  # Handles edge case for remaining rows
        for x in range(0, gfc_info["cols"], block_size):
            cols = min(block_size, gfc_info["cols"] - x)  # Handles edge case for remaining cols
            write_block(
                output_band, all_cols, all_rows, all_values,
                x, y, cols, rows
            )
        
    output.FlushCache()    
    output = None
    
    return {
        "tile":          tile_name,
        "status":        "success",
        "polygon_count": len(gdf),
        "pixel_count":   len(all_pixels),
        "output":        output_tiff,
    }
    
# =============================================================================
# MAIN
# =============================================================================
def main():
    # Make output folders (which are nested)
    output_tiles_dir = os.path.join(OUTPUT_DIR, "tiles")
    os.makedirs(output_tiles_dir, exist_ok=True)
    
    # Get all tile paths
    tile_paths = sorted([os.path.join(GFC_TILES, tile) for tile in os.listdir(GFC_TILES) if tile.endswith(".tif")])
    
    # Build list of arguments
    worker_args = [
        (tile_path, TRAINING_SHP, output_tiles_dir)
        for tile_path in tile_paths
    ]
    
    # Parallel process
    completed_tiles = []
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        task = progress.add_task("Processing tiles...", total=len(worker_args))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(extract_gfc_change, *args): args[0] for args in worker_args}

            for future in as_completed(futures):
                tile_path = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        completed_tiles.append(result["output"])
                        log.info(f"{result['tile']}: {result['status']}")
                    elif result["status"] == "skipped":
                        log.warning(f"{result['tile']}: no intersecting polygons, skipped")
                    else:
                        log.error(f"{result['tile']}: {result.get('reason', 'unknown error')}")
                except Exception as exc:
                    log.error(f"{tile_path} failed: {exc}")
                finally:
                    progress.update(task, advance=1)

    log.info(f"{len(completed_tiles)} tiles written successfully.")
    
    # Merge the output
    if completed_tiles:
        # Get tile lists per quadrant
        nw_tiles = sort_tiles(output_tiles_dir, "N", "W")
        ne_tiles = sort_tiles(output_tiles_dir, "N", "E")
        sw_tiles = sort_tiles(output_tiles_dir, "S", "W")
        se_tiles = sort_tiles(output_tiles_dir, "S", "E")
        
        # Set up output tiffs
        nw_tiff = os.path.join(OUTPUT_DIR, "NW_forest_change.tif")
        ne_tiff = os.path.join(OUTPUT_DIR, "NE_forest_change.tif")
        sw_tiff = os.path.join(OUTPUT_DIR, "SW_forest_change.tif")
        se_tiff = os.path.join(OUTPUT_DIR, "SE_forest_change.tif")
        
        # Call in parallel
        worker_args = [
            (nw_tiles, nw_tiff),
            (ne_tiles, ne_tiff),
            (sw_tiles, sw_tiff),
            (se_tiles, se_tiff)
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(merge_output, *args): args[1] for args in worker_args}
            for future in as_completed(futures):
                output_tiff = futures[future]
                try:
                    future.result()
                    log.info(f"Completed: {output_tiff}")
                except Exception as e:
                    log.error(f"Failed {output_tiff}: {e}")
                    
        
if __name__ == "__main__":
    main()