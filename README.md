# Example_2

A set of scripts which will identify high intensity blobs by [scikit-image's blob detection algorithms](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) and/or [pyclesperacto_prototype's](https://github.com/clEsperanto/pyclesperanto_prototype/tree/master)`detect_maxima_box` to detect local maxima. The point data is written onto .csv files and can be viewed on napari by the `napari_viewer.py` script producing an image, points and label layer.

## Pre-requisites
- Create a new conda/mamba environment and install [devbio-napari](https://github.com/haesleinhuepf/devbio-napari#installation) using mamba (already on OnDemand).
```
mamba create --name devbio-napari-env python=3.9 devbio-napari -c conda-forge -c pytorch
```

- Afterwards, activate the environment.
```
conda activate devbio-napari-env
```

Then navigate to the repo and run the desired scripts from the command line.

## Batch Processing
Iterates through .ome.tif files in `Data` folder and performs difference of gaussian assuming blobs are light on a darker backgorund and `detect_maxima_box` to locate the points of the blobs. Both methods work multi-dimensionally. 

Outputs .csv files into the `Data` folder.

## Napari Viewer
Reads .csv files in the `Data` folder and opens up napari viewer with the segmented nucleus as the image layer and detected blobs as a points and label layer. The [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops/tree/master) plugin measure properties of the blobs such as mean, min and max intensity. 


<img src="./images/regionprops.png" width="300">

Blobs can be individually selected from the viewer and the region properties can be saved as a .csv file.

## Using napari-skimage-regionprops
Running the `example_1_viewer.py` will open napari with a table measuring the properties of the `segmented_nuclei` layer using [regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops/tree/master). To interact with the labels and see which index corresponds to which region, 
1. activate `pick mode`
2. tick `show selected`
3. select any row/label in the table


