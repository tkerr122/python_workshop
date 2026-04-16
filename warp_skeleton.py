from extract_polygons import *

# Load cutline and raster to warp
skeleton_path = "/gpfs/glad1/Theo/Data/Pastures_test/block42/skeleton.tif"
cutline = "/gpfs/glad1/Theo/Data/Pastures_test/warp_cutline/cutline.geojson"
# output_tiff = "/gpfs/glad1/Theo/Data/Pastures_test/warp_cutline/block42_lines.tif"
output_tiff =  "/gpfs/glad1/Theo/Data/Pastures_test/block42/skeleton.tif"

# if os.path.isfile(output_tiff) == False:
#     warp_options = gdal.WarpOptions(format="GTiff",
#                                     cutlineDSName=cutline,
#                                     cropToCutline=True)
#     gdal.Warp(output_tiff, skeleton_path, options=warp_options)
#     print("warped")
# else:
#     print("already warped")

    
# Load new skeleton 
skel_info = get_raster_info(output_tiff)
skel_ds = gdal.Open(output_tiff)
skeleton = skel_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# Find endpoints
kernel = np.ones((3, 3), dtype=np.uint8)
neighbor_count = ndimage.convolve(skeleton, kernel, mode='constant', cval=0)

# Remove pixels with no neighbors
skeleton[neighbor_count == 1] = 0

# Store endpoints
endpoints = (skeleton == 1) & (neighbor_count == 2) # Count: itself & neighbor
ep_coords = list(zip(*np.where(endpoints)))
result = skeleton.copy()

# Get all other pixels
skel_coords = np.array(list(zip(*np.where(skeleton == 1))))
tree = cKDTree(skel_coords)

# Find and label segments
structure = ndimage.generate_binary_structure(2, 2)
labeled, _ = ndimage.label(skeleton, structure=structure)

# Close gaps with shortest distance
gap_threshold = 30
for i, (r1, c1) in enumerate(ep_coords):
    best_dist = np.inf
    best_coord = None

    indices = tree.query_ball_point([r1, c1], gap_threshold)
    for idx in indices:
        r2, c2 = skel_coords[idx]
        
        # Skip if already a part of the skeleton
        if labeled[r1, c1] == labeled[r2, c2]:
            continue

        dist = np.hypot(r2 - r1, c2 - c1)
        if dist < best_dist:
            best_dist = dist
            best_coord = (r2, c2)

    if best_coord is not None:
        r2, c2 = best_coord
        rr, cc = draw_line(int(r1), int(c1), int(r2), int(c2))
        # Clip to array bounds just in case
        valid = (rr >= 0) & (rr < skeleton.shape[0]) & \
                (cc >= 0) & (cc < skeleton.shape[1])
        result[rr[valid], cc[valid]] = 1
    
# Write out
output_closed_lines_path = "/gpfs/glad1/Theo/Data/Pastures_test/warp_cutline/42_new_cropped_lines.tif"
output_ds = gdal.GetDriverByName("GTiff").Create(
    output_closed_lines_path, skel_info.cols, skel_info.rows, 1, gdal.GDT_Byte
)
output_ds.SetGeoTransform(skel_info.transform)
output_ds.SetProjection(skel_info.projection.ExportToWkt())
output_band = output_ds.GetRasterBand(1)
output_band.SetNoDataValue(0)
output_band.WriteArray(result)

print("done")