# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoidFunction(x):
    
    hypothesisFunct = 1/(1+np.exp(-x))
    
    return hypothesisFunct

def sigmoidGradientFunction(x):
    
    hypothesisFunct = np.multiply((1/(1+np.exp(-x))),(1-(1/(1+np.exp(-x)))))
    
    return hypothesisFunct

def computeCostFunctionClassification(x, y, theta, noOfExperiments):
    
    hypothesis = sigmoidFunction(np.matmul(x,theta))
    
    yPos = -y*np.log(hypothesis)
    yNeg = (1-y)*np.log(1-hypothesis)

    cost = yPos-yNeg;
    
    costFunction = (1/noOfExperiments)*np.sum(cost)
    
    return costFunction

def gradientDescentOneVsAllClassification(x, y, alpha, noOfExperiments, iterations, numOfLabels):
    
    theta = np.zeros((numOfLabels,len(x[0])))
    
    for i in range (numOfLabels):
        
        initialTheta = np.zeros((len(x[0]),1))
        JHistory = np.zeros((iterations,1))
        yList = (y == i)
       
        for j in range (iterations):
            
            hypothesis = sigmoidFunction(np.matmul(x,initialTheta))
            gradient = (1/noOfExperiments)*(np.matmul(x.transpose(),(hypothesis-yList)))
#        
            initialTheta = initialTheta - (alpha*gradient)
#        
            JHistory[i] = computeCostFunctionClassification(x, y, initialTheta, noOfExperiments)
#        
        theta[i,:] = initialTheta.transpose()
#    
    return theta

def gradientDescentNeuralNetworks(x, y, theta1, theta2, alpha, noOfExperiments, iterations, labels, lambdaReg):
    
    JHistory = np.zeros((iterations,1))
    
    for i in range (iterations):
        
        JHistory[i], theta1Grad, theta2Grad = computeCostFunctionNeuralNetworks(theta1, theta2, x, y, labels, noOfExperiments, lambdaReg)
        

        theta1 = theta1 - (alpha*theta1Grad)
        theta2 = theta2 - (alpha*theta2Grad)
        
    return JHistory, theta1, theta2

def computeCostFunctionNeuralNetworks(theta1, theta2, x, y, labels, experiments, lambdaReg):
    
    # Convert yData into a matrix ----------------------------------------------
    
    yMatrix = np.zeros((len(y),labels))
    I = np.identity(labels)
    for i in range (len(y)):
        yMatrix[i,:] = I[y[i],:]
    
    # Forward Propagation Algorithm-------------------------------------------------------
    
    z2 = np.matmul(x,theta1.transpose())
    a2 = sigmoidFunction(z2)
    
    a2Bias = np.ones((len(a2),1))
    a2PlusBias = np.column_stack((a2Bias,a2))
    
    z3 = np.matmul(a2PlusBias,theta2.transpose())
    a3 = sigmoidFunction(z3)
    
    #Compute Unregularized Cost -----------------------------------------------
    
    frontEq = np.multiply(-yMatrix,np.log(a3))
    tailEq = np.multiply(1-yMatrix,np.log(1-a3))
    concatEq = frontEq-tailEq
    
    rowSummation = np.sum(concatEq,axis=1)
    colSummation = np.sum(rowSummation)
    
    J = (1/experiments)*colSummation
    
    # Compute Regularized Cost -------------------------------------------------
    
    # Convert first column to 0 because bias coefficient shouldn't be regularized
    
    theta1[:,0] = 0
    theta2[:,0] = 0
    
    theta1Reg = theta1**2
    theta2Reg = theta2**2
    
    rowSummationTheta1 = np.sum(theta1Reg,axis=1)
    colSummationTheta1 = np.sum(rowSummationTheta1)
    
    rowSummationTheta2 = np.sum(theta2Reg,axis=1)
    colSummationTheta2 = np.sum(rowSummationTheta2)
    
    J = J+((lambdaReg/(2*experiments)))*(colSummationTheta1+colSummationTheta2)
    
    # Backpropagation Algorithm ----------------------------------------------------------
    
    d3 = a3-yMatrix
    theta2WithoutBias = theta2[:,1:]
    a2Derivative = sigmoidGradientFunction(z2)
    d2 = np.multiply(np.matmul(d3,theta2WithoutBias),a2Derivative)
    
    delta1 = np.matmul(d2.transpose(),x)
    delta2 = np.matmul(d3.transpose(),a2PlusBias)
    
    theta1Gradient = (1/experiments)*delta1
    theta2Gradient = (1/experiments)*delta2
    
    scaledTheta1 = (lambdaReg/experiments)*theta1Gradient
    scaledTheta2 = (lambdaReg/experiments)*theta2Gradient
    
    theta1Gradient = theta1Gradient+scaledTheta1
    theta2Gradient = theta2Gradient+scaledTheta2
    
    return J, theta1Gradient, theta2Gradient
    
def predictResultOneVsAllClassification(x, theta):
    
    probabilityList = np.zeros((len(x),1))
    probabilityHypothesis = sigmoidFunction(np.matmul(x,theta.transpose()))
    
    probabilityList = np.argmax(probabilityHypothesis, axis=1)
    
    return probabilityList

def predictResultNeuralNetworks(theta1, theta2, x):
    
    probabilityList = np.zeros((len(x),1))
    
    h1 = sigmoidFunction(np.matmul(x,theta1.transpose()))
    h1Bias = np.ones((len(x),1))
    h1PlusBias = np.column_stack((h1Bias,h1))
    h2 = sigmoidFunction(np.matmul(h1PlusBias,theta2.transpose()))
    
    probabilityList = np.argmax(h2, axis=1)
    
    return probabilityList

def initializeWeight(outputVal, inputVal):
    
    epsilon = 0.12 #It can be adjusted according to your need
    thetaWeight = np.zeros((outputVal, inputVal))
    thetaWeight = np.random.rand(outputVal, inputVal) * 2*epsilon-epsilon
    
    return thetaWeight

def visualizeTheNumber (xData, matrixRow):
    
    for row in range (len(matrixRow)):
        
        matrixData = xData[matrixRow[row],:]
        matrixData = matrixData.reshape(20,20)
        
        plt.subplot(3,3,row+1)
        plt.imshow(matrixData.transpose(), cmap='gray_r')