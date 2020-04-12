# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

def computeCostFunctionRegression(x, y, theta, noOfExperiments):
    
    hypothesisFunct = np.matmul(x,theta)
    error = np.subtract(hypothesisFunct,y) 
    errorSquared = error.transpose()**2 
    costFunct = (0.5/noOfExperiments)*(np.sum(errorSquared))
    
    return costFunct

def gradientDescentRegression(x, y, theta, alpha, noOfExperiments, iterations):
    
    JHistory = np.zeros((iterations,1))
    
    for i in range (iterations):
        
        hypothesisFunct= np.matmul(x,theta)
        error = np.subtract(hypothesisFunct,y) 

        derivSlope = np.matmul(x.transpose(),error)
        theta = theta - (alpha*(1/noOfExperiments)*derivSlope)
        
        JHistory[i] = computeCostFunctionRegression(x, y, theta, noOfExperiments)
        
    return theta, JHistory

def normalEquation(x, y):
    
    xTranspose = x.transpose()
    xmatMult = np.matmul(xTranspose,x)
    xmatMultInv = np.linalg.inv(xmatMult)
    xmatMult2 = np.matmul(xmatMultInv,xTranspose)
    
    theta = np.matmul(xmatMult2,y)
    
    return theta

def sigmoidFunction(x):
    
    hypothesisFunct = 1/(1+np.exp(-x))
    
    return hypothesisFunct


def computeCostFunctionClassification(x, y, theta, noOfExperiments):
    
    hypothesis = sigmoidFunction(np.matmul(x,theta))
    
    yPos = -y*np.log(hypothesis)
    yNeg = (1-y)*np.log(1-hypothesis)

    cost = yPos-yNeg;
    
    costFunction = (1/noOfExperiments)*np.sum(cost)
    
    return costFunction
    
def gradientDescentClassification(x, y, theta, alpha, noOfExperiments, iterations):
    
    
    JHistory = np.zeros((iterations,1))
    
    for i in range (iterations):
        
        hypothesis = sigmoidFunction(np.matmul(x,theta))
        gradient = (1/noOfExperiments)*(np.matmul(x.transpose(),(hypothesis-y)))
        
        theta = theta - (alpha*gradient)
        
        JHistory[i] = computeCostFunctionClassification(x, y, theta, noOfExperiments)
    
    return theta, JHistory

def plotContour(theta1Val, theta2Val, JVal, x, y, noOfTraining):
    
    for i in range (len(theta1Val)):
        for j in range (len(theta2Val)):
        
            thetaAll = np.array([[theta1Val[i]],[theta2Val[j]]])

        
            JVal[i,j] = computeCostFunctionRegression(x, y, thetaAll, noOfTraining)

            
    fig = plt.figure(figsize=(12,8))
    ax = Axes3D(fig)
    ax.plot_surface(theta1Val, theta2Val, JVal, cmap=cm.jet, linewidth=0.1)
    ax.view_init(30,90)
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.show()