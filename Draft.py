# -*- coding: utf-8 -*-
"""
Trying to open the file 
"""
from pathlib import Path
import numpy as np
import napari
import imageio.v2 as imageio

from skimage import filters, feature
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pyclesperanto_prototype as cle

def load_file(file, channel):
    filename = data_folder / file
    image = imageio.imread(filename)
    return image[channel, :, :, :]

data_folder = Path("A375M2_NUP96_1")
files = ["A375M2_NUP96_1_MMStack_Pos0.ome.tif"]

pores = load_file(files[0], 0)
nucleus = load_file(files[0], 1)

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
CH1 = viewer.add_image(segmented_pores, name='nucleus pores')
CH2 = viewer.add_points(final_pores, name='blobs skimage', size=5)
CH3 = viewer.add_points(final_pores2, name='blobs clesp', size=5)
