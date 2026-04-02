# Theo Kerr

# Imports/env settings
from shapely.geometry import box
from scipy import stats
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import os, time
console = Console()
gdal.UseExceptions()


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
    
    return int(mode_result.mode)

def classify_change(year_start, year_end):
    # Case 1: change is natural, need to extract mode year of change
    if np.isnan(year_start) and np.isnan(year_end):
        return "null_both"
    
    # Case 2: change is manmade, need to extract all change
    elif year_start == 100 and year_end == 100:
        return "extract_all"
    
    # Case 3: change is natural, need to extract specific years
    elif year_start < 100 and year_end < 100:
        return "natural"
    
    # Case 4: change is manmade, need to extract specific years
    elif year_start > 100 and year_end > 100:
        return "manmade"
    
    else: 
        raise ValueError("Changetype must be either 'natural' or 'manmade'")


# =============================================================================
# Get pixels that the polygon overlaps
# =============================================================================

def get_pixels(raster_path, polygon_wkt):
    # Load raster, get polygon bounds
    raster, raster_info = get_raster_info(raster_path)
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

    col_off  = int((win_x_min - raster_info["transform"][0]) / raster_info["transform"][1])
    row_off  = int((win_y_max - raster_info["transform"][3]) / raster_info["transform"][5])
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

    pixels = pixel_array[polygon_mask == 1].astype(np.int16)
    
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

    # Initialize the array with zeros
    block_array = np.zeros((rows, cols), dtype=np.uint16)

    if block_mask.any():
        # Localise coordinates to block-space (0-indexed within this block)
        local_cols = all_cols[block_mask] - x_off
        local_rows = all_rows[block_mask] - y_off
        block_array[local_rows, local_cols] = all_values[block_mask].astype(np.uint16)

    # Write positionally — xoff/yoff tell GDAL where in the full raster this belongs
    out_band.WriteArray(block_array, x_off, y_off)

# =============================================================================
# Extract GFC change per tile
# =============================================================================

def extract_gfc_change(gfc_tile_path, training_shp, output_dir):
    # Step 1: get tile info
    with console.status(f"Reading {gfc_tile_path}...", spinner="dots"):
        tile_name = Path(gfc_tile_path).stem
        _, gfc_info = get_raster_info(gfc_tile_path)
    console.log(f"✓ Read in {gfc_tile_path}")

    # Step 2: load polygons that intersect the tile
    with console.status("Finding intersecting polygons...", spinner="dots"):
        
        tile_bbox = box(gfc_info["xmin"], gfc_info["ymin"], gfc_info["xmax"], gfc_info["ymax"])
        gdf = gpd.read_file(training_shp, bbox=tile_bbox)

        # Reproject if necessary
        tile_epsg = gfc_info["projection"].GetAuthorityCode(None)
        shp_epsg = gdf.crs.to_epsg()

        if str(shp_epsg) != tile_epsg:
            gdf = gdf.to_crs(epsg=int(tile_epsg))
        
        if gdf.empty:
            return {"tile": tile_name, "status": "skipped", "polygon_count": 0}
        
    console.log(f"✓ Got {len(gdf)} polygons")
    
    # Step 3: extract pixels under each polygon
    all_cols, all_rows, all_pixels = [], [], []
    with console.status("Extracting pixels", spinner="dots"):
        for _, row in gdf.iterrows():
            # Get polygon info, export geom to wkt
            year_start = row.get("year_start")
            year_end = row.get("year_end")
            mode = classify_change(year_start, year_end)
            geom_wkt = row.geometry.wkt
            
            # Extract pixels under polygon
            pixels, col_coords, row_coords = get_pixels(gfc_tile_path, geom_wkt)
            if pixels is None:
                continue
            if pixels.size == 0:
                continue
            
            # Process the pixels according to the mode
            if mode == "null_both":
                mode_value = compute_mode_nonzero(pixels)
                if mode_value is None:
                    continue
                mask = (pixels == mode_value)
                
                all_cols.append(col_coords[mask])
                all_rows.append(row_coords[mask])
                all_pixels.append(pixels[mask])
                
            elif mode == "extract_all":
                mask = (pixels >= 1) & (pixels <= 24)
                
                all_cols.append(col_coords[mask])
                all_rows.append(row_coords[mask])
                all_pixels.append(pixels[mask] + 100)
            
            elif mode == "natural":
                mask = (pixels >= year_start) & (pixels <= year_end)
                
                all_cols.append(col_coords[mask])
                all_rows.append(row_coords[mask])
                all_pixels.append(pixels[mask])
            
            elif mode == "manmade":
                ys = year_start - 100
                ye = year_end - 100
                mask = (pixels >= ys) & (pixels <= ye)
                
                all_cols.append(col_coords[mask])
                all_rows.append(row_coords[mask])
                all_pixels.append(pixels[mask] + 100)
                
            else: 
                continue
            
            if not mask.any():
                continue
            
            # Add pixels to list
            all_cols.append(col_coords[mask])
            all_rows.append(row_coords[mask])
            all_pixels.append(pixels[mask])
    
    console.log(f"✓ Extracted {len(all_pixels)} polygons")
    
    if not all_cols:
        return {"tile": tile_name, "status": "no_change", "polygon_count": len(gdf)}
    
    all_cols   = np.concatenate(all_cols)
    all_rows   = np.concatenate(all_rows)
    all_values = np.concatenate(all_pixels).astype(np.uint16)
    
    # Step 4: set up output tiff
    os.makedirs(output_dir, exist_ok=True)
    output_tiff = os.path.join(output_dir, f"{tile_name}_change.tif")
    output = gdal.GetDriverByName("GTiff").Create(
        output_tiff, gfc_info["cols"], gfc_info["rows"], 1, gdal.GDT_UInt16, options=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES"
        ]
    )
    output.SetGeoTransform(gfc_info["transform"])
    output.SetProjection(gfc_info["projection"].ExportToWkt())
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(255)
    output_band.Fill(0)
    
    # Step 5: write out pixels to new raster matching gfc tile extents
    block_size = 512
    total_blocks = (gfc_info["cols"] // block_size + 1) * (gfc_info["rows"] // block_size + 1)
    with Progress(SpinnerColumn(),
              "[progress.description]{task.description}",
              MofNCompleteColumn(),
              TimeElapsedColumn(),
              transient=True) as progress:
        task = progress.add_task("Writing out raster in blocks...", total=total_blocks)

        for y in range(0, gfc_info["rows"] + 1, block_size):
            rows = min(block_size, gfc_info["rows"] - y)  # Handles edge case for remaining rows
            for x in range(0, gfc_info["cols"] + 1, block_size):
                cols = min(block_size, gfc_info["cols"] - x)  # Handles edge case for remaining cols
                write_block(
                    output_band, all_cols, all_rows, all_values,
                    x, y, cols, rows
                )
                progress.update(task, advance=1)
        
    console.log("✓ Wrote out raster")

    output.FlushCache()    
    output = None
    
    return {
        "tile":          tile_name,
        "status":        "success",
        "polygon_count": len(gdf),
        "pixel_count":   len(all_pixels),
        "output":        output_tiff,
    }
    
gfc_tile = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_tiles/Hansen_GFC-2024-v1.12_lossyear_40N_100W.tif"
training_shp = "/gpfs/glad1/Theo/Shapefiles/GFC/gfc_training.shp"
output_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/test"

completed_tiles = extract_gfc_change(gfc_tile, training_shp, output_dir)