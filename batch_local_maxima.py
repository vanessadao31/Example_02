# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:13:52 2023

@author: daov1
"""
from pathlib import Path
import numpy as np
import csv

from skimage import feature
import napari_segment_blobs_and_things_with_membranes as nsbatwm

from example_2_functions import load_file, redirect_segmentation, binary_threshold, local_maxima

data_folder = Path.cwd() / "Data"
channels = 2
columns = ['File', 'Skimage Blobs', 'Clesperanto Blobs']
total_rows = []

for folder_path in data_folder.glob("A375M2_NUP96*"):

    csv_folder = folder_path / "csv"
    csv_folder.mkdir(exist_ok=True)
    
    for file_path in folder_path.glob("*.ome.tif"):
        
        pores = load_file(folder_path, file_path, 0)
        nucleus = load_file(folder_path, file_path, 1)
         
        nucleus_filtered = nsbatwm.median_filter(nucleus)
        segmented_pores = redirect_segmentation(nucleus_filtered, pores)

        points_data = feature.blob_dog(segmented_pores.astype(float), min_sigma=1, max_sigma=10, threshold=0.7)
        points = points_data[:, :3].astype(int)

        binary = binary_threshold(segmented_pores)
        pos_points = []

        for i in range(points.shape[0]):
            if binary[points[i, 0], points[i, 1], points[i, 2]] == 1:
               pos_points = np.append(pos_points, points[i], axis=0)

        # saving skimage localmaxima
        final_pores = np.reshape(pos_points, (-1, 3))
        skimage_name = f"{folder_path}_skimage.csv"
        np.savetxt(skimage_name, final_pores, delimiter=',')
        
        # saving clesperanto locamaxima
        clesperanto_name = f"{folder_path}_clesperanto.csv"
        final_pores2 = local_maxima(segmented_pores, binary)
        np.savetxt(clesperanto_name, final_pores2, delimiter=',')
        
        # saving actual dataset
        flat_3D = segmented_pores.ravel()
        data_name = f"{folder_path}_napari.csv"
        np.savetxt(data_name, flat_3D, delimiter=',')
        
        row = [folder_path.stem, final_pores.shape[0], final_pores2.shape[0]]
        total_rows = np.append(total_rows, row, axis=0) 
        
    for csv_path in data_folder.glob("*_*.csv"):
        new_path = csv_folder / csv_path.name
        csv_path.replace(new_path)
     
total_rows = np.reshape(total_rows, (-1, len(columns)))

# saving results
with open("summaryfile.csv", mode='w') as summary_file:
    summary_writer = csv.writer(summary_file, delimiter=',')
    summary_writer.writerow(columns)        
    summary_writer.writerows(total_rows)