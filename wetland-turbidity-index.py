#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:12:53 2026

@author: z3547080
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore invalid value warnings during division

# === Paths ===
shapefile_folder = '/Volumes/T9/Vegetation cover melbourne project/Shapefile'
clipped_folder = '/Volumes/T9/Vegetation cover melbourne project/clipped_images'

# === Storage for results ===
records = []

# === Loop over each wetland ===
for shapefile_name in os.listdir(shapefile_folder):
    if shapefile_name.endswith('.shp'):
        wetland_name = os.path.splitext(shapefile_name)[0]
        shapefile_path = os.path.join(shapefile_folder, shapefile_name)
        wetland_folder = os.path.join(clipped_folder, wetland_name)

        if not os.path.exists(wetland_folder):
            print(f"Clipped folder not found for {wetland_name}, skipping.")
            continue

        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty or gdf.crs is None:
                print(f"Invalid shapefile for {wetland_name}, skipping.")
                continue
        except Exception as e:
            print(f"Error reading {shapefile_name}: {e}")
            continue

        tif_files = [f for f in os.listdir(wetland_folder) if f.endswith('.tif')]
        if not tif_files:
            print(f"No TIFFs found for {wetland_name}, skipping.")
            continue

        # === Step 1: Filter based on non-zero pixels ===
        non_nan_pixel_counts = []
        for file in tif_files:
            try:
                filepath = os.path.join(wetland_folder, file)
                with rasterio.open(filepath) as src:
                    image_data = src.read()
                    non_nan_pixel_count = np.sum(~np.isnan(image_data))
                    non_nan_pixel_counts.append((file, non_nan_pixel_count))
            except Exception as e:
                print(f"Skipping file {file} due to read error: {e}")

        if not non_nan_pixel_counts:
            print(f"No valid TIFFs found for {wetland_name}")
            continue

        max_pixels = max([count for _, count in non_nan_pixel_counts])
        files_to_keep = [
            file for file, count in non_nan_pixel_counts
            if count / max_pixels * 100 >= 85
        ]
        
        if not files_to_keep:
            print(f"All TIFFs below 85% quality threshold for {wetland_name}, skipping.")
            continue

        # === Step 2: Process only filtered files ===
        for file in files_to_keep:
            try:
                filepath = os.path.join(wetland_folder, file)
                with rasterio.open(filepath) as src:
                    data = src.read()
                    # Assuming band order: Blue=0, Green=1, Red=2, NIR=3
                    blue, green, red, nir = data[0], data[1], data[2], data[3]
                    pixel_size_x, pixel_size_y = src.res
                    pixel_area_km2 = abs(pixel_size_x * pixel_size_y) / 1e6
                    gdf = gdf.to_crs(src.crs)
                    total_area_km2 = gdf.geometry.area.sum() / 1e6

                # Clean negative values
                nir = np.where(nir < 0, 0, nir)
                red = np.where(red < 0, 0, red)
                green = np.where(green < 0, 0, green)

                # Calculate NDVI
                ndvi = (nir - red) / (nir + red)
                # Calculate NDTI
                ndti = (red - green) / (red + green)

                # Mask NDTI where NDVI >= 0.5 (keep only where NDVI < 0.5)
                ndti_masked = np.where(ndvi < 0.5, ndti, np.nan)
                mean_ndti = np.nanmean(ndti_masked)

                # Extract year from filename
                # Extract date from filename
                parts = file.split('_')
                if len(parts) >= 2:
                    date_str = parts[0]  # e.g. 20210315
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    year = date_obj.year
                else:
                   date_obj = pd.NaT
                   year = np.nan


                # Shorten wetland name (first two parts)
                short_name = '_'.join(wetland_name.split('_')[:2])

                records.append({
    'Wetland': wetland_name,
    'ShortWetland': short_name,
    'Date': date_obj,
    'Year': year,
    'NDTI': mean_ndti,
})
                print(f"✅ {wetland_name} | {file} | Mean NDTI: {mean_ndti:.2f} ")

            except Exception as e:
                print(f"❌ Failed to process {file} in {wetland_name}: {e}")
                continue
# === Convert to DataFrame ===
df = pd.DataFrame(records)

# === Save or Analyze ===
df.to_csv('NDTI_summary final.csv', index=False)
# === Convert records to DataFrame ===
df.to_csv('/Volumes/T9/Vegetation cover melbourne project/NDTI_summary final.csv', index=False)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df = pd.read_csv('/Volumes/T9/Vegetation cover melbourne project/NDTI_summary final.csv')
df.columns = df.columns.str.strip()

df = df[df['Year'].between(2017, 2024)]
df = df.dropna(subset=['Year', 'NDTI', 'Wetland'])

# === Year-wise average NDTI per wetland ===
df_avg = df.groupby(['Wetland', 'Year'])['NDTI'].mean().reset_index()

# === Short label: initials after the first word (e.g. HARPLEY_WWQS8 -> WWQS8) ===
def shorten_initials(name):
    parts = name.replace('-', '_').split('_')
    parts = [p for p in parts if p.strip()]
    if len(parts) > 1:
        return ''.join(p[0] for p in parts[1:])
    return parts[0] if parts else 'Unknown'

# === Rank by FULL wetland name, so wetlands sharing a short code stay separate ===
wetland_order = (df_avg.groupby('Wetland')['NDTI']
                 .mean().sort_values(ascending=False).index)

# === Plot using full names ===
plt.figure(figsize=(14, 20))
ax = sns.boxplot(data=df_avg, x='NDTI', y='Wetland',
                 order=wetland_order, color='brown')

# === Replace tick labels with the short code only ===
ax.set_yticklabels([shorten_initials(w) for w in wetland_order])

print(f"Wetlands plotted: {len(wetland_order)}")

plt.xlabel('Wetland Turbidity Index', fontsize=28)
plt.ylabel('Wetland', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=22)
plt.tight_layout()
plt.savefig('/Volumes/T9/Vegetation cover melbourne project/NDTI_boxplot.png',
            dpi=300, bbox_inches='tight')
plt.show()