# -*- coding: utf-8 -*-
"""
@author: Marcellus Ruben Winastwan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import function as f

os.getcwd()

# read the image

image = plt.imread('mutu.jpg')

if image[0,0,0] > 1:
    image = image/255
else:
    image


# Store the image size
    
imageSize = np.shape(image)


# Initialize xData as our feature set for k-means clustering with size number of pixel*3

xData = image.reshape(imageSize[0]*imageSize[1],3)


# Define max number of iterations and number of centroids K

K= 16
maxIterations = 2

# Initialize centroid 

initialCentroids = f.initializeCentroids(K, xData)


finalCentroids = f.runKMeans(xData, initialCentroids, maxIterations)

print(finalCentroids)

# Next, assign each of the pixel into its closest final centroids.

idx = f.findClosestCentroids(xData, finalCentroids)

# Then, assign the value of final centroids (representation of the colors) to form a compressed image

xCompressed = finalCentroids[idx]

print(np.shape(xCompressed))

# Next, reshape the xCompressed array back to the shape of its original image

xCompressed = xCompressed.reshape(imageSize[0],imageSize[1],3)
print(xCompressed)

plt.subplot(1,2,1)
plt.imshow(image)

plt.subplot(1,2,2)
plt.imshow(xCompressed)
plt.show()
#

# 

    