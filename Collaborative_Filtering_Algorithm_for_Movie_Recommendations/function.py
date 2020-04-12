# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan
"""

import numpy as np

def initializeWeight(outputVal, inputVal):
    
    epsilon = 2 #It can be adjusted according to your need
    weight = np.zeros((outputVal, inputVal))
    weight = np.random.rand(outputVal, inputVal) * 2*epsilon-epsilon
    
    return weight

def loadMovieList(strings):
    
    
    with open (strings,'r') as data:
        
        readData = data.readlines()
        
    readData = [x.strip() for x in readData]
    movieData = []
    
    for i in range (len(readData)):
        
        readData[i] = readData[i].split(' ',1)
        movieData.append(readData[i][1])
        
    return movieData

def computeCostFunctionCollaborationFiltering (X,Y,theta,lambdaReg, R):
    
    thetaGrad = np.zeros((np.shape(theta)))
    XGrad = np.zeros((np.shape(X)))
    
    costTerm = ((np.matmul(X, theta.transpose()))-Y)**2
    regularizedJ = (((lambdaReg/2))*(np.sum(np.sum(theta**2))))+((lambdaReg/2)*(np.sum(np.sum(X**2))))
    finalJ = ((np.sum(np.sum(np.multiply(costTerm,R))))*0.5) + regularizedJ
    
    XTemp = (np.multiply(((np.matmul(X,theta.transpose()))-Y),R))
    XGrad = (np.matmul(XTemp,theta))+(lambdaReg*X)
    
    thetaTemp = (np.multiply(((np.matmul(X, theta.transpose()))-Y),R))
    thetaGrad = (np.matmul(thetaTemp.transpose(),X))+(lambdaReg*theta)
    
    return finalJ, XGrad, thetaGrad

def computeGradientDescentCollaborationFiltering(x, y, theta, alpha, iterations, lambdaReg, R):
    
    JHistory = np.zeros((iterations,1))
    
    for i in range (iterations):
        
        JHistory[i], xGrad, thetaGrad = computeCostFunctionCollaborationFiltering(x, y, theta, lambdaReg, R)

        x = x - (alpha*xGrad)
        theta = theta - (alpha*thetaGrad)
        
    return JHistory, x, theta