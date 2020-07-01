# Pneumonia Detection Based on X-Ray Images with Transfer Learning and ImageGenerator

## Objective
In this project, a convolutional neural network (CNN) in order to predict whether a patient suffers from pneumonia or not based on the X-ray image will be built. In addition to the default model that will be applied, a famous pre-trained CNN architecture, InceptionV3, will also be applied. Then, the accuracy between these two models will be compared against one another.

The dataset that is used for this project was taken from https://data.mendeley.com/datasets/rscbjbr9sj/2. It contains around a thousand of x-ray images of healthy people and people with pneumonia.

After applying transfer learning, below is the example of the result obtained in this project:

<p align="center">
  <img width="800" height="1000" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Pneumonia_Detection_Using%20_Transfer_Learning/pneumonia_predict.jpeg">
</p>


## Files

There are three different files in this project, which are:

- Pneumonia_Detection.ipynb - The Jupyter Notebook file of this project which contains step-by-step approach applied in this project.
- main.py - collection of Python functions used in this project.
- test_images - this folder contains 15 images used for predictions to test the accuracy of the models applied in this project.
