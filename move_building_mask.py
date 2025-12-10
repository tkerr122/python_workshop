import shutil, os

lat_max = 20
lat_min = -10

lon_max = 90
lon_min = 30

lat_dir = "S"
lon_dir = "W"

output_dir = "/gpfs/glad1/Exch/for_Mikus/Planet_building_mask"
os.makedirs(output_dir, exist_ok=True)

for lat in range(lat_max, lat_min, -10):
    for lon in range(lon_max, lon_min, -10):
        # Format lat and lon for padding
        lat_str=f"{lat:02d}{lat_dir}"
        lon_str=f"{lon:03d}{lon_dir}"
        
        # Construct filepath
        path = f"/gpfs/glad1/Exch/Andres_2023/by_Theo/new_binary_2000/{lat_str}_{lon_str}.tif"
        if os.path.isfile(path) == False:
            continue
        
        newpath = os.path.join(output_dir, f"{lat_str}_{lon_str}.tif")
        shutil.copyfile(path, newpath)
        print(f"Moved {path}")
