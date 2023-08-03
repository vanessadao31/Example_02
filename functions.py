# -*- coding: utf-8 -*-
"""
functions
"""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import csv

from skimage import filters, feature
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import pyclesperanto_prototype as cle

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