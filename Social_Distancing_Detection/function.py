# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import tensorflow as tf
import cv2
import time
import yaml
import imutils
from os import listdir
from os.path import isfile, join
import itertools
import math
import glob
import os
import matplotlib as plt


RED = (0, 0, 255)
GREEN = (0, 255, 0)

class Model:
   
    def __init__(self, model_path):

        
        self.detection_graph = tf.Graph()
        
        # Load the model into the tensorflow graph
        with self.detection_graph.as_default():
            
            od_graph_def = tf.compat.v1.GraphDef()
            
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                
                serialized_graph = file.read()
                print(serialized_graph)
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create a session from the detection graph
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def predict(self,img):
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        img_exp = np.expand_dims(img, axis=0)
        
        # Pass the inputs and outputs to the session to get the results 
        (boxes, scores, classes) = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'), self.detection_graph.get_tensor_by_name('detection_scores:0'), self.detection_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'): img_exp})
        
        return (boxes, scores, classes)  
    
def compute_perspective_transform(corner_points,width,height,image):

    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
    
    matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
    img_transformed = cv2.warpPerspective(image,matrix,(width,height))
    
    return matrix,img_transformed

def compute_point_perspective_transformation(matrix,list_centroids):

 
    list_points_to_detect = np.float32(list_centroids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    
    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
        
    return transformed_points_list

def get_human_box_detection(boxes,scores,classes,height,width):

    array_boxes = list() 
    
    for i in range(boxes.shape[1]):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.75
        if int(classes[i]) == 1 and scores[i] > 0.75:
            
            box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
            
            array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
            
    return array_boxes

def get_centroids(array_boxes_detected):

    array_centroids = list() # Initialize empty centroid and ground point lists 
    for index,box in enumerate(array_boxes_detected):
    
        center_x = int(((box[1]+box[3])/2))
        center_y = int(((box[0]+box[2])/2))
        
        array_centroids.append((center_x, center_y))
       
    return array_centroids

def draw_rectangle(corner_points, frame):

    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), GREEN, thickness=1)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), GREEN, thickness=1)
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), GREEN, thickness=1)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), GREEN, thickness=1)