# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan 

"""

import pandas as pd
import numpy as np
import os
import function as f

os.getcwd()

#Read the dataframe with pandas

dataFrame = pd.read_csv('RestaurantData.txt',names=["CityPop", "RestaurantProfit"])

xData = dataFrame["CityPop"].tolist()
yData = dataFrame["RestaurantProfit"].tolist()

### ---------------------------------------------------------------------------
### ---------------------------------------------------------------------------

### ---------------------------------------------------------------------------

#Insert one more column for intercept coefficient.

intercept = np.ones((len(xData),1))
xDataNew = np.column_stack((intercept,xData))
yDataNew = np.asarray(yData).reshape((len(yData),1))

#Insert alpha for learning rate and other necessary variables

alpha = 0.01
numOfIterations = 10000
theta = np.zeros((len(xDataNew[0]),1))
noOfTraining = len(yData)

#### ---------------------------------------------------------------------------
J = f.computeCostFunctionRegression(xDataNew, yDataNew, np.array([[-10],[-1]]), noOfTraining)
print(J)

### ---------------------------------------------------------------------------
#Fetch optimum theta with gradient descent method

thetaGradientDesc, JHistory = f.gradientDescentRegression(xDataNew, yDataNew, theta, alpha, noOfTraining, numOfIterations)    
print(thetaGradientDesc)
### ---------------------------------------------------------------------------
#Fetch optimum theta with normal equation

thetaNormEq = f.normalEquation(xDataNew, yDataNew)
print(thetaNormEq)

#Plot gradient descent with contour plot
theta1Val = np.linspace(-10,10, num=100)
theta2Val = np.linspace(-1,4, num=100)
Jval = np.zeros((len(theta1Val),len(theta2Val)))

f.plotContour(theta1Val, theta2Val, Jval, xDataNew, yDataNew, noOfTraining)
        
    


