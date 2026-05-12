# Theo Kerr

# Global imports/env settings
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import numpy as np
import os, argparse

# Local imports
from get_planet_tiles import get_planet_tiles

gdal.UseExceptions()
console = Console()

"""
This script is a command-line utility to split a given raster or folder of
rasters into tiles matching the planet tile scheme.
===============================================================================
-p option: path to the raster or folder of rasters to be processed
-od option: path to the output directory for the tiles
-crs: crs for the output tiled rasters. Defaults to EPSG:3857
"""


# Process tile
def process_tile(
    tile,
    raster_path,
    output_dir,
    crs,
    planet_tiles,
):
    # Build warp options
    current_tile = f"\"location\" = '{tile}'"
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=crs,
        cutlineDSName=planet_tiles,
        cutlineWhere=current_tile,
        cropToCutline=True,
        warpMemoryLimit=2000,
        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )

    # Warp the raster for current tile
    dst_ds = os.path.join(output_dir, f"{tile}.tif")

    gdal.Warp(dst_ds, raster_path, options=warp_options)

    # Remove blank raster
    ds = gdal.Open(dst_ds)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    ds = None
    band = None

    blank_condition = np.isin(array, [0, 255])
    if np.all(blank_condition):
        os.remove(dst_ds)
        return None

    return tile


# Tile training
def tile_training(raster_path, output_dir, crs="EPSG:3857"):
    # Set up variables
    tiles_to_keep = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Get planet tiles
        planet_tiles_task = progress.add_task("Getting planet tiles", total=None)
        planet_tile_list = get_planet_tiles(raster_path)
        progress.update(planet_tiles_task, completed=1, total=1)

        # Build worker arguments
        workers = len(planet_tile_list) if len(planet_tile_list) < 100 else 100
        planet_tiles = (
            "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.shp"
        )
        worker_args = [
            (tile, raster_path, output_dir, crs, planet_tiles)
            for tile in planet_tile_list
        ]

        tile_task = progress.add_task(
            f"Splitting {raster_path} into tiles", total=len(planet_tile_list)
        )

        # Tile raster in parallel
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_tile, *args): args[0] for args in worker_args
            }
            for future in as_completed(futures):
                tile = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        tiles_to_keep.append(result)
                except Exception as exc:
                    console.print_exception(
                        f"Tile {tile} failed: {exc}", word_wrap=True
                    )
                finally:
                    progress.update(tile_task, advance=1)

        # Sort list and remove duplicates before returning
        tiles_to_keep = sorted(set(tiles_to_keep))

    console.print(
        f"\u2713 Split raster into {len(tiles_to_keep)} tiles", style="dim green"
    )

    return tiles_to_keep


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Script for tiling a raster or folder of rasters"
    )
    parser.add_argument(
        "-p", "--training-path", type=str, help="Path to raster(s)", required=True
    )
    parser.add_argument(
        "-od", "--output-dir", type=str, help="Path to output folder", required=True
    )
    parser.add_argument(
        "-crs", type=int, help="EPSG code for desired crs", default=3857
    )

    # Parse args
    args = parser.parse_args()

    # Set up variables
    training_path = args.training_path
    output_dir = args.output_dir
    crs = f"EPSG:{args.crs}"
    os.makedirs(output_dir, exist_ok=True)

    # Start message
    console.print(
        f"TILING TRAINING FOR {os.path.basename(training_path)}", style="bold cyan"
    )

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
            f.writelines(f"{tile}\n" for tile in planet_tile_list)

    else:
        tile_list = tile_training(training_path, output_dir, crs)
        tiles_txt = os.path.join(output_dir, "planet_tiles.txt")

        with open(tiles_txt, "w") as f:
            f.write("location\n")
            f.writelines(f"{tile}\n" for tile in tile_list)

    # End message
    console.print(f"\n\u2713 All tiles written to {output_dir}\n", style="bold green")


if __name__ == "__main__":
    main()
