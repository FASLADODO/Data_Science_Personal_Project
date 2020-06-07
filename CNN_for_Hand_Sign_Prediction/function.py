# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import cv2

def load_dataset():
    
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def visualizeSign(xData, index):
    
    plt.figure(figsize=(12,5))
    
    for i in range (len(index)):
        
        matrixData = xData[index[i]]
        
        plt.subplot(3,7,i+1)
        plt.imshow(matrixData)
        plt.axis('off')
        
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def get_vanilla_model(input_shape):

    inputs = Input(shape=input_shape)
    A1 = Conv2D(16, (3,3), padding='SAME', activation='relu')(inputs)
    P1 = MaxPooling2D((3,3))(A1)
    A2 = Conv2D(32, (3,3), padding='SAME', activation='relu')(P1)
    P1 = MaxPooling2D((3,3))(A2)
    F = Flatten()(P1)
    FC = Dense(100, activation='relu')(F)
    FC = Dense(20, activation='relu')(FC)
    outputs = Dense(6, activation='softmax')(FC)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_regularized_model(input_shape):
    
    inputs = Input(shape=input_shape)
    A1 = Conv2D(16, (3,3), padding='SAME', activation='relu')(inputs)
    P1 = MaxPooling2D((3,3))(A1)
    A2 = Conv2D(32, (3,3), padding='SAME', activation='relu')(P1)
    P1 = MaxPooling2D((3,3))(A2)
    F = Flatten()(P1)
    FC = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(F)
    FC = Dense(20, activation='relu')(FC)
    outputs = Dense(6, activation='softmax')(FC)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_batchnorm_model(input_shape):
    
    inputs = Input(shape=input_shape)
    A1 = Conv2D(16, (3,3), padding='SAME', activation='relu')(inputs)
    P1 = MaxPooling2D((3,3))(A1)
    A2 = Conv2D(32, (3,3), padding='SAME', activation='relu')(P1)
    P1 = MaxPooling2D((3,3))(A2)
    F = Flatten()(P1)
    FC = Dense(100, activation='relu')(F)
    FC = BatchNormalization()(FC)
    FC = Dense(20, activation='relu')(FC)
    outputs = Dense(6, activation='softmax')(FC)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_dropout_model(input_shape):
    
    inputs = Input(shape=input_shape)
    A1 = Conv2D(16, (3,3), padding='SAME', activation='relu')(inputs)
    P1 = MaxPooling2D((3,3))(A1)
    A2 = Conv2D(32, (3,3), padding='SAME', activation='relu')(P1)
    P1 = MaxPooling2D((3,3))(A2)
    F = Flatten()(P1)
    FC = Dense(100, activation='relu')(F)
    FC = Dropout(0.3)(FC)
    FC = Dense(20, activation='relu')(FC)
    outputs = Dense(6, activation='softmax')(FC)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def import_and_resize(image_data):
    
    img = cv2.imread(image_data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_resize = (cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC))/255.
    
    return img_resize