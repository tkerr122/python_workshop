# Imports/env settings 
from osgeo import gdal, ogr, osr
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import os, argparse, shutil
console = Console()
gdal.UseExceptions()

"""This script is a command-line utility to create footprints for a folder of rasters
================================================
-p option: path to the folder of rasters to be processed
-od option: path to the output directory
"""

def get_footprint(raster_path, output_dir):
    # Get footprint
    name = os.path.splitext(os.path.basename(raster_path))[0]
    footprint_path = os.path.join(output_dir, f"{name}_footprint.fgb")
    gdal.Footprint(footprint_path, raster_path, format="FlatGeobuf", dstSRS="EPSG:3857")

def merge_footprints(footprint_dir, output_path, progress, task):
    out_ds = ogr.GetDriverByName("FlatGeobuf").CreateDataSource(output_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    out_layer = out_ds.CreateLayer("footprints", srs=srs, geom_type=ogr.wkbUnknown)
    fieldname = ogr.FieldDefn("filename", ogr.OFTString)
    out_layer.CreateField(fieldname)

    for file in os.listdir(footprint_dir):
        if file.endswith(".fgb"):
            filepath = os.path.join(footprint_dir, file)
            ds = ogr.Open(filepath)
            lyr = ds.GetLayer()
            for feat in lyr:
                out_feat = ogr.Feature(out_layer.GetLayerDefn())
                out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                out_feat.SetField("filename", file.rsplit("_footprint.fgb")[0])
                out_layer.CreateFeature(out_feat)
                out_feat = None
            
            ds = None
        
        progress.update(task, advance=1)
    
    out_ds = None
    
def main():
    # Setup
    print("\nGETTING FOOTPRINTS...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for getting the footprints for a folder of rasters")
    parser.add_argument("-p", "--raster-dir", type=str, required=True, help="Path to folder of rasters to be processed")
    parser.add_argument("-od", "--output-dir", type=str, required=True, help="Path to output footprint folder")
    parser.add_argument("-c", "--cores", type=int, required=True, help="Number of CPUs to use")

    # Parse arguments
    args = parser.parse_args()
    
    # Set up variables
    raster_dir = args.raster_dir
    output_dir = args.output_dir
    temp = os.path.join(output_dir, "temp")
    cores = args.cores
    os.makedirs(temp, exist_ok=True)
    rasters = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]
    worker_args = [
        (os.path.join(raster_dir, file), temp)
        for file in rasters
    ]
        
    # Get footprints
    with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
        
        task = progress.add_task("Getting footprints...", total=len(rasters))
        
        with ProcessPoolExecutor(max_workers=cores) as pool:
            futures = {pool.submit(get_footprint, *args): args[0] for args in worker_args}
            for future in as_completed(futures):
                progress.update(task, advance=1)
    
    console.print(f"{len(rasters)} footprints written")
    
    # Merge footprints
    with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
        
        task = progress.add_task("Merging footprints...", total=len(rasters))
        output_file = os.path.join(output_dir, f"{os.path.basename(raster_dir)}_footprints.fgb")
        merge_footprints(temp, output_file, progress, task)
        
        # Remove footprints dir
        shutil.rmtree(temp)
    
    console.print("All footprints merged")
    
        
if __name__ == "__main__":
    main()