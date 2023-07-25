# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:56:15 2023

@author: daov1
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import napari
import imageio.v2 as imageio

from skimage.morphology import local_maxima
from skimage import data, measure, exposure, filters, feature
from skimage.segmentation import watershed
from scipy.ndimage import binary_erosion

import napari_segment_blobs_and_things_with_membranes as nsbatwm
import napari_blob_detection as nbd
import pyclesperanto_prototype as cle

from detect_blobs import _detect_blobs, difference_of_gaussian


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
    
    # blur and detect local maxima
    blurred_spots = filters.gaussian(image, 2)
    spot_centroids = local_maxima(blurred_spots)
    
    # blur and threshold
    blurred = filters.gaussian(image, 2)
    thresh = filters.threshold_li(image)
    binary_li = blurred >= thresh
    
    return binary_li * signal

nucleus_filtered = nsbatwm.median_filter(nucleus)
segmented_pores = redirect_segmentation(nucleus_filtered, pores)

nucleus_filtered = nsbatwm.median_filter(nucleus)
segmented_pores = redirect_segmentation(nucleus_filtered, pores)
pores_labels, state, Points = difference_of_gaussian(segmented_pores)

viewer = napari.Viewer()
CH1 = viewer.add_image(segmented_pores, name='nucleus pores')
CH2 = viewer.add_points(pores_labels, name='blobs', size=5)

# # Edge blob detection
# def edge_segmentation(mask, signal):
#     image = np.asarray(mask)
    
#     # blur and detect local maxima
#     blurred_spots = filters.gaussian(image, 2)
#     spot_centroids = local_maxima(blurred_spots)
    
#     # blur and threshold
#     blurred = filters.gaussian(image, 2)
#     thresh = filters.threshold_li(image)
#     binary_li = blurred >= thresh
    
#     eroded = binary_erosion(binary_li, iterations=5)
    
#     return eroded * signal

# nucleus_edge = edge_segmentation(nucleus_filtered, pores)
# CH3 = viewer.add_image(nucleus_edge, name='edge')