# Social Distancing Detection with Python, OpenCV, and TensorFlow

## Objective

The purpose of this project is to build an algorithm to detect whether people are obeying the social distancing rule to avoid the further spread of Coronavirus. This project is just for fun purpose only and how we can utilize the feature of OpenCV and TensorFlow framework to create such a social distancing detection algorithm.

In this project, the pre-trained TensorFlow model, which is Faster R-CNN ResNet trained on MS COCO 2017 dataset will be applied to predict the coordinate of the bounding boxes as well as to predict which object that each bounding box represents. The video used for this project comes from PETS dataset.

Link to PETS dataset: http://www.cvg.reading.ac.uk/PETS2009/a.html#s0

Link to model zoo object detection TensorFlow: https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md

Below is the video output of this project.

<p align="center">
  <img width="700" height="500" src=https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Social_Distancing_Detection/image/video_gif.gif>
</p>

## Files

There are three folders and three files in this project:
- config folder: consists of a configuration file defining the corner points for social distancing detection.
- image folder: consits of images for configuration file and the gif of output result.
- video folder: consists of the original video and the final output video.
- function.py: Python file containing the functions used for this project.
- main.py: the main Python file for this project.
- Social_Distancing_Detection.ipynb: The Jupyter Notebook which explains the step-by-step method applied in the project.
