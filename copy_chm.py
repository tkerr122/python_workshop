# Imports
from tqdm import tqdm
import os, shutil, argparse

"""This script is a command-line utility to copy CHMs from their parent folder to a new directory
based on a provided text file
================================================
-s option: lidar survey to copy CHMs from
-t option: path to the txt file
"""

def load_paths(txt_file: str) -> list:
    with open(txt_file, "r") as f:
        next(f)  # skip header
        paths = [f"{line.strip().split(',')[1]}.tif" for line in f if line.strip()]
    
    return paths

def copy_chm(paths: list, input_dir: str, output_dir: str) -> None:
    progress_bar = tqdm(total= len(paths), desc="Progress", unit="file")
    for path in paths:
        chm_path = os.path.join(input_dir, path)
        shutil.copy2(chm_path, output_dir)
        
        progress_bar.update(1)

    progress_bar.close()
    
def main():
    # Start message
    print("\nCOPYING CHMs...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for copying CHMs based on a text file")
    parser.add_argument("-s", "--survey", type=str, required=True, help="Lidar survey to copy CHMs from")
    parser.add_argument("-t", "--text-file", type=str, required=True, help="Path to text file for copying")
    
    # Parse arguments
    args = parser.parse_args()

    # Set up variables
    footprints_txt = args.text_file
    input_dir = f"/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/{args.survey}_CHM"
    output_dir = f"{input_dir}_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Load the footprint paths
    print("Reading filepaths...")
    paths = load_paths(footprints_txt)

    # Copy the relevant CHMs to a new folder
    print("Moving CHMs...")
    copy_chm(paths, input_dir, output_dir)
    
    print("Done")
    
if __name__ == "__main__":
    main()