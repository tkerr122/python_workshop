# Theo Kerr on 11/13/2025
# For use on linux cluster "gdalenv" conda env

# Imports/env settings
from osgeo import gdal, ogr
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import argparse, os

gdal.UseExceptions()
console = Console()

"""This script is a command-line utility to find which planet tiles a given raster or folder of rasters
intersect with.
================================================
-p option: path to the raster or folder of rasters to be processed
-t option: path to the output TXT file which will have the planet tiles
"""


# =========================
# Define custom functions
# =========================
def get_planet_tiles(raster_path):
    # Get raster footprint
    footprint_ds = gdal.Footprint("", raster_path, format="Memory", dstSRS="EPSG:3857")
    footprint_layer = footprint_ds.GetLayer(0)
    footprint_feature = footprint_layer.GetNextFeature()
    footprint_geom = footprint_feature.GetGeometryRef().Clone()

    # Open planet tiles and filter by footprint
    planet_tiles_path = (
        "/gpfs/glad1/Theo/Data/Planet_and_1_degree/Planet_tiles_and_degree.fgb"
    )
    tiles_ds = ogr.Open(planet_tiles_path)
    tiles_layer = tiles_ds.GetLayer(0)
    tiles_layer.SetSpatialFilter(footprint_geom)

    # Return the tiles
    tiles = sorted(set(f.GetField("location") for f in tiles_layer))

    return tiles


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Script for getting the planet tiles of a raster"
    )
    parser.add_argument(
        "-p",
        "--raster-path",
        type=str,
        help="Path to raster or folder of rasters to be processed",
    )
    parser.add_argument(
        "-t", "--txt-path", type=str, help="Path to txt for output planet tiles"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up variables
    raster_path = args.raster_path
    txt_path = args.txt_path

    # Loop through the folder and get the planet tiles for each raster
    if os.path.isdir(raster_path):
        # Load input rasters
        rasters = [
            os.path.join(raster_path, raster)
            for raster in os.listdir(raster_path)
            if raster.endswith(".tif")
        ]

        # Get tiles
        tiles_list = []
        workers = len(rasters) if len(rasters) < 100 else 100

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:

            task = progress.add_task(
                f"Getting planet tiles for rasters in {raster_path}", total=len(rasters)
            )

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(get_planet_tiles, raster): raster for raster in rasters
                }
                for future in as_completed(futures):
                    result = futures[future]
                    try:
                        tiles = future.result()
                        tiles_list.extend(tiles)
                    except Exception as exc:
                        console.print_exception(
                            f"{result} failed: {exc}", word_wrap=True
                        )
                    finally:
                        progress.update(task, advance=1)

        console.print(f"Got {len(tiles_list)} planet tiles for {raster_path}")
    else:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Getting planet tiles for {raster_path}")
            tiles_list = get_planet_tiles(raster_path)
        console.print(f"Got planet tiles for {raster_path}")

    # Write to txt file
    with open(txt_path, "w") as f:
        tiles_list = sorted(set(tiles_list))
        f.write("location\n")
        f.writelines(f"{tile}\n" for tile in tiles_list)
    console.print(f"Planet tiles written to {txt_path}")


if __name__ == "__main__":
    main()
