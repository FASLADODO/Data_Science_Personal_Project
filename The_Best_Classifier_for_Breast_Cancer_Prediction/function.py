# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:36:13 2020

@author: ASUS
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def TreeClassifier(xTrain, xVal, yTrain, yVal, depth):
    
    np.random.seed(1)
    F1_Tree_train = []
    F1_Tree_test = []
    
    for i in range(len(depth)):
    
        tree = DecisionTreeClassifier(criterion='entropy',max_depth=depth[i]).fit(xTrain,yTrain)
    
        yHat_Tree = tree.predict(xVal)
        yHat_train = tree.predict(xTrain)
    
        F1_Tree_train.append(metrics.f1_score(yTrain, yHat_train, average='weighted'))    
        F1_Tree_test.append(metrics.f1_score(yVal, yHat_Tree, average='weighted'))
    
    return F1_Tree_train, F1_Tree_test 

def SVMClassifier(xTrain, xVal, yTrain, yVal, kernel):
    
    np.random.seed(1)
    F1_SVM_train = []
    F1_SVM_test = []
    
    for i in range(len(kernel)):
        clf = svm.SVC(kernel=kernel[i])
        clf.fit(xTrain, yTrain)
    
        yHat_SVM = clf.predict(xVal)
        yHat_train = clf.predict(xTrain)
    
        F1_SVM_train.append(metrics.f1_score(yTrain, yHat_train, average='weighted'))
        F1_SVM_test.append(metrics.f1_score(yVal, yHat_SVM, average='weighted'))
    
    return F1_SVM_train,F1_SVM_test

def LogisticClassifier(xTrain, xVal, yTrain, yVal, C):
    
    np.random.seed(1)
    
    F1_Log_train = []
    F1_Log_test = []
    
    for i in range(len(C)):
        
        LR_model = LogisticRegression(C=C[i]).fit(xTrain,yTrain)
    
    
        yHat_Log = LR_model.predict(xVal)
        yHat_train = LR_model.predict(xTrain)
    
        F1_Log_train.append(metrics.f1_score(yTrain, yHat_train, average='weighted'))
        F1_Log_test.append(metrics.f1_score(yVal, yHat_Log, average='weighted'))
    
    return F1_Log_train, F1_Log_test


def NBClassifier(xTrain, xVal, yTrain, yVal):
    
    np.random.seed(1)
        
    F1_NB_train = []
    F1_NB_test = []
    
    clf = GaussianNB().fit(xTrain,yTrain)
    
    
    yHat_Log = clf.predict(xVal)
    yHat_train = clf.predict(xTrain)
    
    F1_NB_train.append(metrics.f1_score(yTrain, yHat_train, average='weighted'))
    F1_NB_test.append(metrics.f1_score(yVal, yHat_Log, average='weighted'))
    
    return F1_NB_train, F1_NB_test
