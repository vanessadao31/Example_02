# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:13:52 2023

@author: daov1
"""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import csv

from skimage import filters, feature
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pyclesperanto_prototype as cle

from Draft import redirect_segmentation, load_file, binary_threshold, local_maxima

directory = Path.cwd() / "Data"
channels = 2
channel_names = ["pores", "nucleus"]

for folder_path in directory.glob("A375M2_NUP96*"):
    
    for file_path in folder_path.glob("*.ome.tif"):
        
        pores = load_file(file_path, 0)
        nucleus = load_file(file_path, 1)
         
        nucleus_filtered = nsbatwm.median_filter(nucleus)
        segmented_pores = redirect_segmentation(nucleus_filtered, pores)

        points_data = feature.blob_dog(segmented_pores.astype(float), min_sigma=1, max_sigma=10, threshold=0.7)
        points = points_data[:, :3].astype(int)

        binary = binary_threshold(segmented_pores)
        pos_points = []

        for i in range(points.shape[0]):
            if binary[points[i, 0], points[i, 1], points[i, 2]] == 1:
               pos_points = np.append(pos_points, points[i], axis=0)

        final_pores = np.reshape(pos_points, (-1, 3))
        final_pores2 = local_maxima(segmented_pores, binary)
        
        skimage_name = f"{folder_path}_skimage.csv"
        clesperanto_name = f"{folder_path}_clesperanto.csv"
        
    np.savetxt(skimage_name, final_pores, delimiter=',')
    np.savetxt(clesperanto_name, final_pores2, delimiter=',')
        
    for csv_path in directory.glob("*.csv"):
        new_path = folder_path / csv_path.name
        csv_path.replace(new_path)
     
            