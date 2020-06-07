# Convolutional Neural Networks for Hand Sign Prediction

## Objective
The purpose of this project is to predict the hand sign using convolutional neural networks (CNN) architecture. First, 'vanilla' CNN, i.e CNN without any hyperparameter or regularization tuning, will be applied and the corresponding accuracy to predict the training set will be investigated. 

Then, different kinds of regularization technique will be applied to see the difference in accuracy compared to vanilla CNN. Below is the architecture of vanilla CNN that will be applied in this project.


<p align="center">
  <img width="1100" height="80" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/CNN_for_Hand_Sign_Prediction/images/CNN.png">
</p>

The dataset consists of 1200 images of hand sign indicates the number of 0 until 5. The visualization of the some images in the dataset can be seen below:

<p align="center">
  <img width="1000" height="300" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/CNN_for_Hand_Sign_Prediction/images/hand.jpg">
</p>

## Files

There are four files in this project, which are:

- CNN_for_Hand_Sign_Prediction.ipynb -  Jupyter notebook file which contains step-by-step approach used in this project.
- function.py - list of functions used in this project.
- test_signs.h5 - test set of the data.
- training_signs.h5 - training set of the data.
- images - folder contains phone camera images to use for model prediction.
