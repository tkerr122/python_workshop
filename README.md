# Python workshop - All my work in progress files
---

## Scripts

| Script | Purpose |
|---|---|
| `copy_chm.py` | CLI utility to copy CHM files from one folder to another |
| `copy_dtm.py` | CLI utility to copy DTM files from one folder to another |
| `extract_polygons.py` | Extracts polygons from a linear features raster |
| `footprint.py` | CLI utility that gets footprints from a folder of rasters |
| `get_planet_tiles.py` | CLI utility that finds which planet tiles a given raster/folder of rasters intersect with |
| `gfc_create_training.py` | v1 of the gfc_extractor.py script, loops through polygons in shapefiles and extracts the GFC pixel values for training |
| `gfc_extractor.py` | Uses parallel processing and block indexing to extract gfc pixel values under a training data shapefile and write to merged raster |
| `pastures_create_training.py` | Loops through training data shapefiles and rasterizes each feature for further processing |
| `slope.py` | CLI utility that creates a slope raster from a specified folder of DTMs |
| `tile_training.py` | CLI utility that splits a given raster/folder of rasters into tiles matching the planet tile scheme|

---