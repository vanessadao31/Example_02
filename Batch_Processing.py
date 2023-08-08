# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:13:52 2023

@author: daov1
"""
import argparse
import csv
from pathlib import Path

import napari_segment_blobs_and_things_with_membranes as nsbatwm
import numpy as np
from skimage import feature

from example_2_functions import load_file, redirect_segmentation, binary_threshold, local_maxima

channels = 2
columns = ['File', 'Skimage Blobs', 'Clesperanto Blobs']
total_rows = []

parser = argparse.ArgumentParser()
parser.add_argument("parent_directory", help="parent search directory")
parser.add_argument("folder_pattern", help="folders matching this pattern in the parent directory will be searched")
parser.add_argument("file_pattern", help="files matching this pattern will be analysed")
args = parser.parse_args()

data_folder = Path(args.parent_directory)
folder_pattern = args.folder_pattern
file_pattern = args.file_pattern

for folder_path in data_folder.glob(folder_pattern):
    print('Processing all files in ' + str(folder_path))

    for file_path in folder_path.glob(file_pattern):
        print('Processing ' + str(file_path))
        pores = load_file(folder_path, file_path.name, 0)
        nucleus = load_file(folder_path, file_path.name, 1)

        print('Smoothing nuclear channel')
        nucleus_filtered = nsbatwm.median_filter(nucleus)
        print('Segmenting nucleus')
        segmented_pores = redirect_segmentation(nucleus_filtered, pores)
        print('Finding local maxima using skimage DOG filter')
        points_data = feature.blob_dog(segmented_pores.astype(float), min_sigma=1, max_sigma=10, threshold=0.7)
        points = points_data[:, :3].astype(int)

        binary = binary_threshold(segmented_pores)
        pos_points = []

        for i in range(points.shape[0]):
            if binary[points[i, 0], points[i, 1], points[i, 2]] == 1:
                pos_points = np.append(pos_points, points[i], axis=0)

        final_pores = np.reshape(pos_points, (-1, 3))
        print('Finding local maxima using pyclesperanto')
        final_pores2 = local_maxima(segmented_pores, binary)
        flat_3D = segmented_pores.ravel()

        skimage_name = f"{folder_path}_skimage.csv"
        clesperanto_name = f"{folder_path}_clesperanto.csv"
        data_name = f"{folder_path}_napari.csv"

        row = [folder_path.stem, final_pores.shape[0], final_pores2.shape[0]]
        total_rows = np.append(total_rows, row, axis=0)

    print('Saving results...')
    np.savetxt(skimage_name, final_pores, delimiter=',')
    np.savetxt(clesperanto_name, final_pores2, delimiter=',')
    np.savetxt(data_name, flat_3D, delimiter=',')

    for csv_path in data_folder.glob("*.csv"):
        new_path = folder_path / csv_path.name
        csv_path.replace(new_path)

total_rows = np.reshape(total_rows, (-1, len(columns)))

print('Summarising results...')
with open("summary_file.csv", mode='w') as summary_file:
    summary_writer = csv.writer(summary_file, delimiter=',')
    summary_writer.writerow(columns)
    summary_writer.writerows(total_rows)
