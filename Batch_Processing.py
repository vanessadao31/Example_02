# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:13:52 2023

@author: daov1
"""
from pathlib import Path
import numpy as np
import napari
import imageio.v2 as imageio

from skimage import filters, feature
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pyclesperanto_prototype as cle

from Draft import redirect_segmentation, load_file, binary_threshold, local_maxima

directory = Path.cwd() / "Data"
channels = 2
channel_names = ["pores", "nucleus"]

for folder_path in directory.glob("A375M2_NUP96*"):
    for file_path in folder_path.glob("*.ome.tif"):
        # for channel in range(channels):
        #     channel_names[channel] = load_file(file_path, channel)
        
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

        viewer = napari.Viewer()
        CH1 = viewer.add_image(segmented_pores, name='nucleus')
        CH2 = viewer.add_points(final_pores, name='blobs skimage', size=5)
        CH3 = viewer.add_points(final_pores2, name='blobs clesp', size=5)
            
     
        
     
            