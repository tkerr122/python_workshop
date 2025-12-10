import os
import time

# Get block number
block = input("What is the block number? ").strip()
folder = r"Y:\HLSDIST\Validation\2024_10kmblock\kml"

# Build the target filenames in numeric order
kmls = [f"unit{block}_{i}.kml" for i in range(1, 21)]

# One scan of the folder for existence checks
files_in_folder = set(os.listdir(folder))

first_open = True
for kml in kmls:
    if kml in files_in_folder:
        print(f"Opening {kml}...")
        os.startfile(os.path.join(folder, kml))
        if first_open:
            input("Press 'y' then Enter once Google Earth is launched: ")
            first_open = False
        else:
            time.sleep(0.05)

