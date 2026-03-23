# Theo Kerr

# Global imports/env settings
from osgeo import gdal
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import numpy as np
import os, argparse
gdal.UseExceptions()
console = Console()

# Local imports
from get_planet_tiles import get_planet_tiles

"""
This script is a command-line utility to split a given raster or folder of
rasters into tiles matching the planet tile scheme.
===============================================================================
-p option: path to the raster or folder of rasters to be processed
-od option: path to the output directory for the tiles
-crs: crs for the output tiled rasters. Defaults to EPSG:3857
"""

# Tile training
def tile_training(raster_path, output_dir, crs="EPSG:3857"):
    # Set up variables
    tiles_to_keep = []
    
    # Get planet tiles
    with Progress(SpinnerColumn(),
            "[progress.description]{task.description}",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True) as progress:
        task = progress.add_task("Getting planet tiles", total=None)
        planet_tile_list = get_planet_tiles(raster_path)
        planet_tiles = "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
    
        # Loop through planet tiles
        progress.update(task, description="Splitting progress", total=len(planet_tile_list))
        for tile in planet_tile_list:
            # Build warp options
            current_tile = f'"location" = \'{tile}\''
            warp_options = gdal.WarpOptions(format="GTiff", 
                                            dstSRS=crs,
                                            cutlineDSName=planet_tiles,
                                            cutlineWhere=current_tile,
                                            cropToCutline=True,
                                            multithread=True,
                                            warpMemoryLimit=2000,
                                            creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=100"],
                                            warpOptions=["NUM_THREADS=100"])
            
            # Warp the raster for current tile
            dst_ds = os.path.join(output_dir, f"{tile}.tif")
            
            gdal.Warp(dst_ds, raster_path, options=warp_options)
            
            progress.update(task, advance=1)
            
            # Remove blank raster
            ds = gdal.Open(dst_ds)
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            ds = None
            band = None
            
            blank_condition =  np.isin(array, [0, 255])
            if np.all(blank_condition):
                os.remove(dst_ds)
                continue
            
            # Add tile to list
            tiles_to_keep.append(tile)
    
        # Sort list and remove duplicates before returning
        tiles_to_keep = sorted(set(tiles_to_keep))
    
    console.print(f"\u2713 Split raster into {len(tiles_to_keep)}", style="dim green")
    
    return tiles_to_keep

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for tiling a raster or folder of rasters")
    parser.add_argument("-p", "--training-path", type=str, help="Path to raster(s)", required=True)
    parser.add_argument("-od", "--output-dir", type=str, help="Path to output folder", required=True)
    parser.add_argument("-crs", type=int, help="EPSG code for desired crs", default=3857)
    
    # Parse args
    args = parser.parse_args()
    
    # Set up variables    
    training_path = args.training_path
    output_dir = args.output_dir
    crs = f"EPSG:{args.crs}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start message
    console.print(f"TILING TRAINING FOR {os.path.basename(training_path)}", style="bold cyan")
    
    # Tile training in folder/file
    if os.path.isdir(training_path):
        # Get training
        rasters = os.listdir(training_path)
        
        # Tile each raster
        planet_tile_list = []
        for raster in rasters:
            # Set up path
            path = os.path.join(training_path, raster)
            
            # Tile raster
            tile_list = tile_training(path, output_dir, crs)
            planet_tile_list.extend(tile_list)

        # Write out planet tiles list
        tiles_txt = os.path.join(output_dir, "planet_tiles.txt")
        planet_tile_list = sorted(set(planet_tile_list))
        
        with open(tiles_txt, "w") as f:
            f.write("location\n")
            f.writelines(f"{tile}\n"for tile in planet_tile_list)
    
    else:
        tile_list = tile_training(training_path, output_dir, crs)
        tiles_txt = os.path.join(output_dir, "planet_tiles.txt")
        
        with open(tiles_txt, "w") as f:
            f.write("location\n")
            f.writelines(f"{tile}\n"for tile in tile_list)
    
    # End message
    console.print(f"\n\u2713 All tiles written to {output_dir}\n", style="bold green")
            
if __name__ == "__main__":
    main()