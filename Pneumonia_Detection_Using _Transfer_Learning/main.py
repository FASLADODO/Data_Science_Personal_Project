# -*- coding: utf-8 -*-
"""
"""
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg

def get_data(path_to_data):
    
    directory = os.path.join(path_to_data)
    data_names = os.listdir(directory)
    
    return data_names

def visualize_images(image):
    
    n_row = 4
    n_cols = 3
    color_pixel = 150
    
    plt.figure(figsize=(30,30))
    
    for i in range (len(image)):
        
        img = mpimg.imread(image[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(color_pixel, color_pixel), interpolation=cv2.INTER_CUBIC))/255.
    
        plt.subplot(n_row,n_cols,i+1)
        plt.imshow(img_resize)
        plt.axis('off')
        
def get_default_model():
    
    model = Sequential([
    
    # First convolution
        Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
    
    # Second convolution
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
    
    # Third convolution
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
    
    # Fourth convolution
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
    
    # Fifth convolution
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
    
    # Flatten the results
        Flatten(),
    
    # Dense hidden layer
        Dense(512, activation='relu'),
        Dropout(0.2),
    
    # Output neuron. 
        Dense(1, activation='sigmoid')
    ])
    
    return model

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      featurewise_center=False, 
                                      samplewise_center=False, 
                                      featurewise_std_normalization=False,  
                                      samplewise_std_normalization=False, 
                                      zca_whitening=False, 
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.1)
    
  
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (150,150),
                                                       batch_size = 512,
                                                       class_mode = 'binary',
                                                       subset='training')
    
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                          target_size = (150,150),
                                                          batch_size = 52,
                                                          class_mode = 'binary',
                                                          subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(150,150),
                                                     batch_size = 62,
                                                     class_mode = 'binary')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model

def import_and_predict(image_data, model):
    
    predictions = []
    values = []
    
    for i in range (len(image_data)):
        
        sigmoid = "None"
        pred = "None"
        
        img = mpimg.imread(image_data[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_CUBIC))/255.
    
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
      
        if prediction[0][0] > 0.5:
            
            sigmoid = "Sigmoid: "+ str(round(prediction[0][0],3))
            pred = "Prediction: Pneumonia"
            
        else:
            sigmoid = "Sigmoid: "+ str(round(prediction[0][0],3))
            pred = "Prediction: Normal"
        
        values.append(sigmoid)
        predictions.append(pred)
        
    return values, predictions

def visualize_xRay(image_test, confidence, prediction):
    
    n_row = 5
    n_cols = 3
    color_pixel = 150
    
    plt.figure(figsize=(40,40))
    
    for i in range (len(image_test)):
        
        img = mpimg.imread(image_test[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(color_pixel, color_pixel), interpolation=cv2.INTER_CUBIC))/255.
    
        plt.subplot(n_row,n_cols,i+1)
        plt.title(image_test[i],fontsize=30, color='r')
        plt.imshow(img_resize)
        plt.axis('off')
        
        plt.annotate(prediction[i],((color_pixel/2)-40,color_pixel-20),fontsize=20, color='r')
        plt.annotate(confidence[i],((color_pixel/2)-40,color_pixel-10), fontsize=20, color='r')
        
