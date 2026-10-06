#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:14:57 2026

@author: z3547080
"""

#%% final code for vegetation cover( %)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from datetime import datetime

# === Input paths ===
shapefile_folder = '/Users/z3547080/Documents/Vegetation cover melbourne project/Shapefile'
clipped_folder = '/Users/z3547080/Documents/Vegetation cover melbourne project/clipped_images'

# === Parameters ===
percentage_threshold = 85  # Threshold to keep high-quality images

# === Output containers ===
all_results = []

# === Function to count non-zero pixels in raster ===
def count_non_zero_pixels(src):
    image = src.read()
    return np.count_nonzero(image)

# === Process each wetland ===
for shapefile_name in os.listdir(shapefile_folder):
    if shapefile_name.endswith('.shp'):
        wetland_id = os.path.splitext(shapefile_name)[0]
        shapefile_path = os.path.join(shapefile_folder, shapefile_name)
        wetland_folder = os.path.join(clipped_folder, wetland_id)

        if not os.path.exists(wetland_folder):
            print(f"Warning: No clipped folder found for {wetland_id}. Skipping.")
            continue

        # Load shapefile and reproject later to raster CRS
        gdf = gpd.read_file(shapefile_path)
        total_area_shapefile = gdf.geometry.area.sum()

        # List all TIFF files in the folder
        masked_files = [f for f in os.listdir(wetland_folder) if f.endswith('.tif')]
        non_zero_pixel_counts = []

        # Count non-zero pixels for filtering
        for file in masked_files:
            file_path = os.path.join(wetland_folder, file)
            with rasterio.open(file_path) as src:
                non_zero_pixel_counts.append(count_non_zero_pixels(src))

        # Filter based on threshold
        if not non_zero_pixel_counts:
            continue
        max_non_zero_pixels = max(non_zero_pixel_counts)
        files_to_keep = [
            f for f, count in zip(masked_files, non_zero_pixel_counts)
            if count / max_non_zero_pixels * 100 >= percentage_threshold
        ]

        # === Per-file processing ===
        for file in files_to_keep:
            file_path = os.path.join(wetland_folder, file)
            with rasterio.open(file_path) as src:
                blue = src.read(1)
                green = src.read(2)
                red = src.read(3)
                nir = src.read(4)

                # Reproject shapefile to raster CRS
                gdf_proj = gdf.to_crs(src.crs)
                total_area_km2 = gdf_proj.geometry.area.sum() / 1e6

                # Clean up values
                nir = np.where(nir < 0, 0, nir)
                red = np.where(red < 0, 0, red)
                green = np.where(green < 0, 0, green)

  
                ndvi = (nir - red) / (nir + red)

                # Masking
                
                ndvi_masked = np.where(ndvi > 0.5, ndvi, np.nan)

                # Get pixel size
                pixel_size_x, pixel_size_y = src.res
                pixel_area_km2 = abs(pixel_size_x * pixel_size_y) / 1e6

                # Count pixels
                vegetation_pixel_count = np.count_nonzero(~np.isnan(ndvi_masked))


                # Cover percentage
                vegetation_cover = (vegetation_pixel_count * pixel_area_km2 / total_area_km2) * 100
                

                # Extract date from filename (assumes format like '20220202_104530_something.tif')
                try:
                    base = os.path.basename(file)
                    date_part, time_part = base.split('_')[:2]
                    timestamp = datetime.strptime(date_part + time_part, '%Y%m%d%H%M%S')
                except Exception:
                    timestamp = None

                # Append results
                all_results.append({
                    'Wetland': wetland_id,
                    'Filename': file,
                    'Date': timestamp,
                    'VegetationCover(%)': vegetation_cover,
                    
                })

                print(f"Wetland: {wetland_id} | File: {file} | Vegetation: {vegetation_cover:.2f}%")

# === Save all results to CSV ===
df_results = pd.DataFrame(all_results)
output_csv_path = '/Users/z3547080/Documents/Vegetation cover melbourne project/vegetation_turbidity_cover_summary.csv'
df_results.to_csv(output_csv_path, index=False)
print(f"\nSaved results for all wetlands to {output_csv_path}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df = pd.read_csv('vegetation_turbidity_cover_summary.csv')

# === Filter years and drop missing values ===
df = df[df['Year'].between(2017, 2024)]
df.dropna(subset=['Year', 'VegetationCover(%)', 'Wetland'], inplace=True)

# === Generate 'ShortWetland' if not already there ===
if 'ShortWetland' not in df.columns:
    def shorten(name):
        parts = name.replace('-', '_').split('_')
        return '_'.join(parts[:2]) if len(parts) >= 2 else name
    df['ShortWetland'] = df['Wetland'].apply(shorten)

# === ✅ 1. Count files per ShortWetland per year ===
file_counts = df.groupby(['ShortWetland', 'Year']).size().reset_index(name='FileCount')

print("\n📊 Number of files processed per wetland per year (short name):")
print(file_counts.sort_values(['ShortWetland', 'Year']))

# Optional: Save the summary to CSV
file_counts.to_csv('file_count_per_shortwetland_year.csv', index=False)
print("✅ File counts saved to 'file_count_per_shortwetland_year.csv'")

# === ✅ 2. Boxplot of all individual vegetation cover values grouped by ShortWetland ===
# === ✅ Sort ShortWetland alphabetically before plotting ===
sorted_wetlands = sorted(df['ShortWetland'].unique())

plt.figure(figsize=(16, 10))
sns.boxplot(
    data=df,
    x='VegetationCover(%)',
    y='ShortWetland',
    order=sorted_wetlands,  # 👈 Enforce alphabetical order on y-axis
    palette='viridis',
    showfliers=True
)
plt.title('Distribution of Vegetation Cover (%) per Wetland (All Years)')
plt.xlabel('Vegetation Cover (%)')
plt.ylabel('Wetland)')
plt.tight_layout()
plt.grid(True)
plt.show()

