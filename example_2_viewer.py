# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:05:43 2023

@author: daov1
"""
from pathlib import Path
import napari
import numpy as np
import argparse

from skimage.io import imread

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

    for file_path in folder_path.glob("*_napari.png"):
        print('Processing ' + str(file_path))
        segmented_pores = imread(file_path)
        
    for file_path in folder_path.glob("*_clesperanto.csv"):
    # for file_path in data_directory.glob("*_skimage.csv"):
        print('Processing ' + str(file_path))
        final_pores = np.loadtxt(file_path, delimiter=',')

    for file_path in folder_path.glob("*_properties.csv"):
        print(f"Processing {file_path.name}")
        voxel_sizes = np.loadtxt(file_path, delimiter=',')

    print('Opening napari')
    viewer = napari.Viewer()
    CH1 = viewer.add_image(segmented_pores, name='Nucleus', scale=(voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]))
    CH2 = viewer.add_points(final_pores, name='Blobs', size=5, scale=(voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]))
    # CH3 = viewer.add_points(final_pores2, name='blobs clesp', size=5)
    
    data, state, Labels = points_to_labels(viewer.layers[1], viewer.layers[0])
    CH3 = viewer.add_labels(data, name='Labels', scale=(voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]))

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
