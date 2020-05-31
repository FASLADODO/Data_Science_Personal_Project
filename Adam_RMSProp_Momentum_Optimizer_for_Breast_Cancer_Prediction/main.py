# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def initializeParameters (layerDimension):
    
    np.random.seed(3)
    parameters = {}
    L = len(layerDimension)     

    for l in range(1, L):
       
        parameters['W' + str(l)] = np.random.randn(layerDimension[l],layerDimension[l-1])*np.sqrt(2 / layerDimension[l-1])
        parameters['b' + str(l)] = np.zeros((layerDimension[l],1))

        
        assert(parameters['W' + str(l)].shape == (layerDimension[l], layerDimension[l-1]))
        assert(parameters['b' + str(l)].shape == (layerDimension[l], 1))

        
    return parameters

def linearForward(A, W, b):

    Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def tanh(Z):
    
    A = np.tanh(Z)
    cache = Z 
    
    return A, cache

def activationForward(A_prev, W, b, activation):

    if activation == "sigmoid":

        Z, linear_cache = linearForward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
     
    elif activation == "relu":

        Z, linear_cache = linearForward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    elif activation == "tanh":
        
        Z, linear_cache = linearForward(A_prev,W,b)
        A, activation_cache = tanh(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def modelForwardProp(X, parameters,activation):
    
    caches = []
    A = X
    L = len(parameters) // 2                 
 
    for l in range(1, L):
        A_prev = A 
        
        A, cache = activationForward(A_prev,parameters["W" + str(l)],parameters["b" + str(l)],activation)
        
        caches.append(cache)
        
    AL, cache = activationForward(A,parameters["W" + str(L)],parameters["b" + str(L)],'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def computeCost(AL, Y):
    
    m = Y.shape[1]
    
    cost = -1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))

    cost = np.squeeze(cost) 
    #assert(cost.shape == ())
    
    return cost

def computeCost_mini_batch(AL,Y):
    
    logProbs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    costTotal =  np.sum(logProbs)
    
    return costTotal

def linearBackward(dZ, cache):
   
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def reluBackward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoidBackward(dA, cache):

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanhBackward(dA, cache):
    
    Z = cache
    
    dZ = dA*(1-np.power(Z,2))
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def activationBackward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        dZ = reluBackward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
        
    elif activation == "sigmoid":
       
        dZ = sigmoidBackward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
     
    elif activation == "tanh":
       
        dZ = tanhBackward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
        
    return dA_prev, dW, db

def modelBackProp(AL, Y, caches, activation):   
    
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activationBackward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activationBackward(grads["dA"+str(l+1)], current_cache, activation)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp 
        grads["db" + str(l + 1)] = db_temp

    return grads

def updateParameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate*grads["dW"+str(l + 1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db"+str(l + 1)])

    return parameters

def predict(X, y, parameters, activation):
    
    m = X.shape[1] 
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = modelForwardProp(X, parameters, activation)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    truePositive = []
    trueNegative = []
    falsePositive = []
    falseNegative = []
    
    for i in range (0, probas.shape[1]):
        truePositive.append((p[0,i] == 1 and y[0,i] == 1))
        trueNegative.append(p[0,i] == 0 and y[0,i] == 0)
        falsePositive.append(p[0,i] == 1 and y[0,i] == 0)
        falseNegative.append(p[0,i] == 0 and y[0,i] == 1)
    
    epsilon = 10e-8
    recall = truePositive.count(True)/(truePositive.count(True)+falseNegative.count(True)+epsilon)
    precision = truePositive.count(True)/(truePositive.count(True)+falsePositive.count(True)+epsilon)
    
    F1_score = 2*(precision*recall/(precision+recall+epsilon))
        
    return F1_score

def gradientDescentNN(X, Y, layerDimension, learningRate, numIterations, activation):

    np.random.seed(1)
    costs = []                  

    parameters = initializeParameters(layerDimension)
 
    
    # Loop gradient descent
    for i in range(0, numIterations):

        AL, caches = modelForwardProp(X, parameters, activation)
  
        cost = computeCost(AL, Y)
        
        grads = modelBackProp(AL, Y, caches, activation)
 
        parameters = updateParameters(parameters, grads, learningRate)
                
        # Print the cost every 100 training example
        if i % 10000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 10000 == 0:
            costs.append(cost)

    return parameters

###############################################################################
