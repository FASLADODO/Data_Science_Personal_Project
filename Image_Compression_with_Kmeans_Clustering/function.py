# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def initializeCentroids (numOfCentroids, xData):
    
    centroids = np.zeros((numOfCentroids, len(xData[1])))
    
    idx = np.random.permutation(len(xData))
    
    centroids = xData[idx[0:K],:]
    
    return centroids

def findClosestCentroids (x, centroids):
    
    centroidMatrices = np.zeros((len(x), len(centroids)))
    
    for i in range (len(x)):
        
        for j in range (len(centroids)):
            
            #calculate the least square error of each data points wrt initial centroids
            
            centroidMatrices[i,j]=np.sum((x[i]-centroids[j])**2)
        
     # Store the index of centroid each datapoints is assigned to
     
    index = np.argmin(centroidMatrices,1)
    
    return index

def computeCentroids (xData, idx, numCentroids, initialCentroids):
    
    newCentroids = np.zeros((numCentroids,len(initialCentroids[0])))
    
    for i in range (numCentroids):
        
        indices = np.where(idx == i)
        
        newCentroids[i,:] = np.mean(xData[indices],0)
        
    return newCentroids
    
    
def runKMeans (xData, initialCentroids, maxIterations):
    
    numCentroids = len(initialCentroids)
    
    for i in range (maxIterations):
        
        # Find the closest centroids of each data points
        
        idx = findClosestCentroids(xData, initialCentroids)
        
        # compute the new mean of each centroid
        
        newCentroids = computeCentroids(xData, idx, numCentroids, initialCentroids)
        
    return newCentroids