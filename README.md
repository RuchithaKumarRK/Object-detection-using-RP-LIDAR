# Yolo3d
## Overview
This repository contains a PyTorch implementation of Complex Yolo. You can download weights here: 
* [Model Weights](http://iki.hs-weingarten.de/elser/ModelWeights.pth).

The network is trained to detect cars in a representation of the pointcloud from the KITTI dataset. It was trained for 150 epochs with samples 0 - 3000 as input.

## Data
You can download the KITTI dataset for Object detection here: 
* [Velodyne point clouds (29 GB)](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip)
* [Left color images of object data set (12 GB)](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
* [Camera  calibration  matrices  of  object  data  set  (16
MB)](http://www.cvlibs.net/download.php?file=data_object_calib.zip)
* [Training labels of object data set (5 MB)](http://www.cvlibs.net/download.php?file=data_object_label_2.zip).

## Pytorch
You get Pytorch from here https://pytorch.org/. Since this network will work on a CPU you can choose CUDA None. If you are using PyCharm on a Windows you can also watch this video: https://www.youtube.com/watch?v=geyFTCBJHy4 where it is explained how to install pytorch.

## Libraries
To run the program you need to install those libraries with dependencies:
* torch
* matplotlib (only for visualization)
* cv2 (only for visualization)
* json
* numpy
