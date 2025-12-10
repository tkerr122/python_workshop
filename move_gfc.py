import shutil, os

lat_max = 100
lon_max = 200

lat_dir = "N"
lon_dir = "E"

output_dir = "/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_3"
os.makedirs(output_dir, exist_ok=True)

for lat in range(lat_max, -10, -10):
    for lon in range(lon_max, -10, -10):
        # Format lat and lon for padding
        lat_str=f"{lat:02d}{lat_dir}"
        lon_str=f"{lon:03d}{lon_dir}"
        
        # Construct filepath
        path = f"/gpfs/glad1/Theo/Data/Global_Forest_Change/gfc_tiles/Hansen_GFC-2024-v1.12_lossyear_{lat_str}_{lon_str}.tif"
        if os.path.isfile(path) == False:
            continue
        
        newpath = os.path.join(output_dir, f"Hansen_GFC-2024-v1.12_lossyear_{lat_str}_{lon_str}.tif")
        shutil.copyfile(path, newpath)
