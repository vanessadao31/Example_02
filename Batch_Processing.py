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

data_folder = Path.cwd() / "Data"
channels = 2
channel_names = ["pores", "nucleus"]
columns = ['File', 'Skimage Blobs', 'Clesperanto Blobs']
total_rows = []

def load_file(file, channel):
    filename = data_folder / file
    image = imageio.imread(filename)
    return image[channel, :, :, :]

def redirect_segmentation(mask, signal):
    image = np.asarray(mask)
    
    # blur and threshold
    blurred = filters.gaussian(image, 2)
    thresh = filters.threshold_li(image)
    binary_li = blurred >= thresh
    
    return binary_li * signal

def binary_threshold(data):
    image = np.asarray(data)
    
    # blurred = filters.gaussian(image, 2)
    thresh = 0.4 * image.max()
    binary = image >= thresh
    
    return binary.astype(int)

def local_maxima(image, binary_image):
    preprocessed = filters.gaussian(image, sigma=1, preserve_range=True)
    local_maxima_image = cle.detect_maxima_box(preprocessed)
    all_labeled = cle.label_spots(local_maxima_image)

    final_spots = cle.exclude_labels_with_map_values_out_of_range(
        binary_image,
        all_labeled,
        minimum_value_range=1,
        maximum_value_range=1)
    
    points = np.argwhere(final_spots)
    
    return points

for folder_path in data_folder.glob("A375M2_NUP96*"):
    
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
        flat_3D = segmented_pores.ravel()
        
        skimage_name = f"{folder_path}_skimage.csv"
        clesperanto_name = f"{folder_path}_clesperanto.csv"
        data_name = f"{folder_path}_napari.csv"
        
        row = [folder_path.stem, final_pores.shape[0], final_pores2.shape[0]]
        total_rows = np.append(total_rows, row, axis=0)
        
    np.savetxt(skimage_name, final_pores, delimiter=',')
    np.savetxt(clesperanto_name, final_pores2, delimiter=',')
    np.savetxt(data_name, flat_3D, delimiter=',')

    for csv_path in data_folder.glob("*.csv"):
        new_path = folder_path / csv_path.name
        csv_path.replace(new_path)
     
total_rows = np.reshape(total_rows, (-1, len(columns)))

# saving results
with open("summary_file.csv", mode='w') as summary_file:
    summary_writer = csv.writer(summary_file, delimiter=',')
    summary_writer.writerow(columns)        
    summary_writer.writerows(total_rows)