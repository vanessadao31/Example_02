# Example_2

A set of scripts which will identify high intensity blobs by [scikit-image's blob detection algorithms](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) and/or [pyclesperacto_prototype's](https://github.com/clEsperanto/pyclesperanto_prototype/tree/master)`detect_maxima_box` to detect local maxima. The point data is written onto .csv files and can be viewed on napari by the `napari_viewer.py` script producing an image, points and label layer.

## Pre-requisites
Create a new conda/mamba environment and install [devbio-napari](https://github.com/haesleinhuepf/devbio-napari#installation) using mamba (already on OnDemand).
```
mamba create --name devbio-napari-env python=3.9 devbio-napari -c conda-forge -c pytorch
```

Afterwards, activate the environment.
```
conda activate devbio-napari-env
```

Then navigate to the repo and run the desired scripts from the command line.

## Batch Processing

## Napari Viewer
