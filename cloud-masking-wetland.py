#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:18:31 2026

@author: z3547080
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:06:51 2025

@author: z3547080
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

# === Paths ===
shapefile_folder = '/Users/z3547080/Documents/Vegetation cover melbourne project/Shapefile'
raw_images_root = '/Users/z3547080/Documents/Vegetation cover melbourne project/Raw images'
output_root = '/Users/z3547080/Documents/Vegetation cover melbourne project/clipped_images 2'

# Ensure output root exists
os.makedirs(output_root, exist_ok=True)

# Loop through each shapefile in the shapefile folder
for shapefile_name in os.listdir(shapefile_folder):
    if shapefile_name.endswith('.shp'):
        wetland_name = os.path.splitext(shapefile_name)[0]
        shapefile_path = os.path.join(shapefile_folder, shapefile_name)
        print(f"\n--- Processing {wetland_name} ---")

        # Read shapefile
        try:
            shapefile = gpd.read_file(shapefile_path)
        except Exception as e:
            print(f"❌ Error reading shapefile {shapefile_name}: {e}")
            continue

        if shapefile.empty:
            print(f"⚠️  Shapefile {shapefile_name} is empty. Skipping.")
            continue
        if shapefile.crs is None:
            print(f"⚠️  Shapefile {shapefile_name} has no CRS. Skipping.")
            continue

        # Define corresponding raw image folder
        raw_folder = os.path.join(raw_images_root, wetland_name)
        if not os.path.exists(raw_folder):
            print(f"⚠️  Raw image folder not found for {wetland_name}, skipping.")
            continue

        # Define output folder
        output_folder = os.path.join(output_root, wetland_name)
        os.makedirs(output_folder, exist_ok=True)

        # List all tif files in raw image folder
        tif_files = [f for f in os.listdir(raw_folder) if f.endswith('.tif')]
        if not tif_files:
            print(f"⚠️  No .tif files found in {raw_folder}")
            continue

        # Identify harmonized and udm2 pairs
        harmonized_files = {}
        udm2_files = {}
        for file in tif_files:
            parts = file.split('_')
            if len(parts) >= 3:
                date = parts[0]
                time_ = parts[1]
                key = (date, time_)

                if file.endswith('SR_harmonized_clip.tif'):
                    harmonized_files[key] = file
                elif file.endswith('udm2_clip.tif'):
                    udm2_files[key] = file

        matching_pairs = [(harmonized_files[k], udm2_files[k]) for k in harmonized_files if k in udm2_files]
        if not matching_pairs:
            print(f"⚠️  No matching harmonized + udm2 pairs for {wetland_name}")
            continue

        print(f"✅ Found {len(matching_pairs)} matching pairs in {wetland_name}")

        # Process each pair
        for harmonized_file, udm2_file in matching_pairs:
            harmonized_path = os.path.join(raw_folder, harmonized_file)
            udm2_path = os.path.join(raw_folder, udm2_file)

            try:
                with rasterio.open(harmonized_path) as harmonized_src:
                    harmonized_data = harmonized_src.read()
                    harmonized_meta = harmonized_src.meta
                    harmonized_crs = harmonized_src.crs

                    # Reproject shapefile if needed
                    if shapefile.crs != harmonized_crs:
                        print("🔄 Reprojecting shapefile to match raster CRS...")
                        shapefile = shapefile.to_crs(harmonized_crs)

                    # === Check for overlap ===
                    raster_bounds = harmonized_src.bounds
                    raster_box = box(*raster_bounds)
                    shapefile_union = shapefile.geometry.unary_union

                    if not shapefile_union.intersects(raster_box):
                        print(f"⚠️  Skipping {harmonized_file} – shapefile does not overlap raster.")
                        continue

                    # === Clip harmonized image ===
                    geometries = [geom for geom in shapefile.geometry]
                    clipped_harmonized, clipped_transform = mask(harmonized_src, geometries, crop=True)
                    harmonized_meta.update({
                        "transform": clipped_transform,
                        "height": clipped_harmonized.shape[1],
                        "width": clipped_harmonized.shape[2]
                    })

                # === Clip udm2 image ===
                with rasterio.open(udm2_path) as udm2_src:
                    clipped_udm2, _ = mask(udm2_src, geometries, crop=True)

                # === Apply UDM2 mask ===
                clear_mask = clipped_udm2[0] == 1
                masked_harmonized_data = np.where(clear_mask, clipped_harmonized, np.nan)

                # === Write masked output ===
                output_file = os.path.join(output_folder, f"{os.path.splitext(harmonized_file)[0]}_masked.tif")
                harmonized_meta.update(dtype=rasterio.float32, nodata=np.nan)

                with rasterio.open(output_file, 'w', **harmonized_meta) as dst:
                    dst.write(masked_harmonized_data.astype(rasterio.float32))

                print(f"✅ Saved: {output_file}")

            except Exception as e:
                print(f"❌ Error processing {harmonized_file}: {e}")
                continue
