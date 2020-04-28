# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan
"""


import numpy as np
import os
import function as f

os.getcwd()



from scipy.io import loadmat

################################## Read Data ######################

handwritingData = loadmat('Data.mat')        
        
xData = handwritingData['X']
yData  =handwritingData['y']
################################ Visualize the Number ############
m = [1, 600, 1000, 1620, 2000, 2640, 3102, 3700, 4600]
f.visualizeTheNumber(xData,m)

################################ Define Necessary Variables #######

intercept = np.ones((len(xData),1))
xDataNew = np.column_stack((intercept,xData))

alpha = 3
numOfIterations = 2000
noOfTraining = len(yData)
noOfLabels = 10
initialTheta = np.zeros((len(xDataNew[0]),1))

for i in range(len(yData)):
    
    if yData[i] == 10:
        
        yData[i] = 0

np.savetxt('yData.csv',yData)

############################ One vs ALL Classification #########################################
#h = sigmoidFunction(np.matmul(xDataNew,initialTheta))
#
#theta = gradientDescentOneVsAllClassification(xDataNew, yData, alpha, noOfTraining, numOfIterations, noOfLabels)
#
#accProbability = predictResultOneVsAllClassification(xDataNew,theta)

############################# Neural Networks ##################################################
inputLayerSize = len(xDataNew[0])
hiddenLayerSize = 25
lambdaRegularization = 1

# Initialize Random Weight of Theta
Theta1 = f.initializeWeight(hiddenLayerSize,inputLayerSize)
Theta2 = f.initializeWeight(noOfLabels,hiddenLayerSize+1)

costJ, theta1Grad, theta2Grad = f.computeCostFunctionNeuralNetworks(Theta1, Theta2, xDataNew, yData, noOfLabels, noOfTraining, lambdaRegularization)

J, finTheta1, finTheta2 = f.gradientDescentNeuralNetworks(xDataNew, yData, Theta1, Theta2, alpha, noOfTraining, numOfIterations, noOfLabels, lambdaRegularization)
print(J[-1])

accProbability = f.predictResultNeuralNetworks(finTheta1, finTheta2, xDataNew)

######################################################################################
accNew = np.asarray(accProbability.reshape((len(yData),1)))
    

modelAccuracy = np.mean((accNew == yData))*100
print(modelAccuracy)
#np.savetxt('data.csv',accNew)