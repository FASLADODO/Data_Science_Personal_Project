# Neural Networks Algorithm for Handwritten Digits Recognition

## Problem
The purpose of this project is to use neural networks algorithm to predict the output of handwritten digits between 0 until 9. The following is the example of the handwritten digits.
<p align="center">
  <img width="800" height="400" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Neural_Networks_for_Handwritten_Digits_Recognition/number.png">
</p>

The data being used in this project was taken from Prof. Andrew Ng online Machine Learning course on Coursera, in which the file has M-extension. The file contains 5000 handwritten digits, each digit is represented by 20 x 20 pixel image.

The neural networks model being used consists of an input layer with 4000 units ( because 20 x 20 pixel image will be unrolled into 4000 rows of 1D vector), a hidden layer with 25 units, and an output layer with 10 units (because the possible output is between 0 until 9).
The representation of the neural networks model can be seen as follows:
<p align="center">
  <img width="700" height="400" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Neural_Networks_for_Handwritten_Digits_Recognition/Schema.png">
</p>

At the end the accuracy of the neural networks model will be examined.

## Files

There is one mat file, one Jupyter Notebook file, two Python files, and two csv files in this project.
- Data.mat - the data being used in this project.
- data.csv - the csv file containing the predicted response value from neural networks algorithm.
- yData.csv - the csv file containing the actual value response variable.
- Neural Networks for Handwritten Digits Recognition.ipynb - The Jupyter notebook file which contains a walkthrough step-by-step report for this project.
- main.py - main Python file being used for this project.
- function.py - a Python file consists of list of functions being used in main.py.
