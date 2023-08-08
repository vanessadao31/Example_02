# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:05:43 2023

@author: daov1
"""
from pathlib import Path
import napari
import numpy as np
import argparse

from napari_blob_detection import points_to_labels
from napari_skimage_regionprops import regionprops_table

parser = argparse.ArgumentParser()
parser.add_argument("parent_directory", help="parent search directory")
parser.add_argument("folder_pattern", help="folders matching this pattern in the parent directory will be searched")
args = parser.parse_args()

data_folder = Path(args.parent_directory)
folder_pattern = args.folder_pattern

for folder_path in data_folder.glob(folder_pattern):
    print('Processing all files in ' + str(folder_path))

    for file_path in folder_path.glob("*_napari.csv"):
        print('Processing ' + str(file_path))
        flat_segmented_pores = np.loadtxt(file_path, delimiter=',')
        segmented_pores = flat_segmented_pores.reshape(-1, 1060, 1520)
        
    for file_path in folder_path.glob("*_clesperanto.csv"):
    # for file_path in data_directory.glob("*_skimage.csv"):
        print('Processing ' + str(file_path))
        final_pores = np.loadtxt(file_path, delimiter=',')

    print('Opening napari')
    viewer = napari.Viewer()
    CH1 = viewer.add_image(segmented_pores, name='Nucleus')
    CH2 = viewer.add_points(final_pores, name='Blobs', size=5)
    # CH3 = viewer.add_points(final_pores2, name='blobs clesp', size=5)
    
    data, state, Labels = points_to_labels(viewer.layers[1], viewer.layers[0])
    CH3 = viewer.add_labels(data, name='Labels')

    print('Adding region properties')
    # region properties
    regionprops_table(
        viewer.layers[0].data,
        viewer.layers[2].data,
        intensity=True,
        size=False,
        napari_viewer=viewer)
    
    napari.run()
    
    # napari.save_layers(folder_path, viewer.layers)
