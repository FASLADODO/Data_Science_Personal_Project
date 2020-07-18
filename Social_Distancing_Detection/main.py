# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import cv2
import yaml
import itertools
import math
import os
import function as f


RED = (0, 0, 255)
GREEN = (0, 255, 0)

########################### LOAD CONFIG FILE ##################################

print("[ Loading config file for the bird view transformation ] ")

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []

for section in cfg:
    corner_points.append(cfg["image_parameters"]["p1"])
    corner_points.append(cfg["image_parameters"]["p2"])
    corner_points.append(cfg["image_parameters"]["p3"])
    corner_points.append(cfg["image_parameters"]["p4"])
    
    width_og = int(cfg["image_parameters"]["width_og"])
    height_og = int(cfg["image_parameters"]["height_og"])
    
    img_path = cfg["image_parameters"]["img_path"]
   
    
print(" Done ..." )

################### LOAD PRE_TRAINED TENSORFLOW MODEL #########################

model_names_list = [name for name in os.listdir("C:/Users/ASUS/models/.") if name.find(".") == -1]
for index,model_name in enumerate(model_names_list):
    print(" - {} [{}]".format(model_name,index))

model_path="C:/Users/ASUS/models/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb" 

print( " [ Loading TensorFlow Model ... ]")
model = f.Model(model_path)
print("Done ...")


################### LOAD VIDEO ################################################

video_names_list = [name for name in os.listdir("C:/Users/ASUS/models/video/.") if name.endswith(".mp4") or name.endswith(".avi")]
for index,video_name in enumerate(video_names_list):
    print(" - {} [{}]".format(video_name,index))

video_path="C:/Users/ASUS/models/video/PETS2009.avi"  

################### DEFINE MIN DISTANCE #######################################

distance_minimum = "110"

################### COMPUTE TRANSFORMATION MATRIX #############################

matrix,imgOutput = f.compute_perspective_transform(corner_points,width_og,height_og,cv2.imread(img_path))
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1] 
dim = (width, height)

################### PREDICT THE VIDEO ########################################

vs = cv2.VideoCapture(video_path)
output_video_1 = None
# Loop until the end of the video stream
while True:
    # Load the image of the ground and resize it to the correct size
    img = cv2.imread("C:/Users/ASUS/models/image/chemin_1.png")
    bird_view_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # Load the frame
    (frame_exists, frame) = vs.read()
    
    print(frame_exists)
   
    if not frame_exists:
        break
    else:

        # Make the predictions for a frame
        (boxes, scores, classes) =  model.predict(frame)
       

        # Return only boundix boxes that represent humans 
        array_boxes_detected = f.get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])

        # Compute the centroids of each bounding boxes
        array_centroids= f.get_centroids(array_boxes_detected)

        # Use the transform matrix to get the transformed coordinates
        transformed_centroids = f.compute_point_perspective_transformation(matrix, array_centroids)

        # Check if 2 or more people have been detected (otherwise no need to detect)
        if len(transformed_centroids) >= 2:
            for index, centroid in enumerate(transformed_centroids):
                if not (centroid[0] > width or centroid[0] < 0 or centroid[1] > height+200 or centroid[1] < 0 ):
                    cv2.rectangle(frame,(array_boxes_detected[index][1],array_boxes_detected[index][0]),(array_boxes_detected[index][3],array_boxes_detected[index][2]),GREEN,2)

            # Iterate over every possible permutations of the transformed centroids
            list_indexes = list(itertools.combinations(range(len(transformed_centroids)), 2))
            
            for i,pair in enumerate(itertools.combinations(transformed_centroids, r=2)):
                
                # Check if the distance between each combination of points is less than the minimum distance
                if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
                    
                    # Change the colors of the points that are too close from each other to red
                    if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
                       
                        index_pt1 = list_indexes[i][0]
                        index_pt2 = list_indexes[i][1]
                        
                        cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),RED,2)
                        cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),RED,2)


    # Draw the rectangle of the area in the images where the detection is considered
    f.draw_rectangle(corner_points, frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Write the both outputs video to a local folders
    if output_video_1 is None:
        fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
        output_video_1 = cv2.VideoWriter("C:/Users/ASUS/models/video/output_video_i.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
       
    elif output_video_1 is not None:
        output_video_1.write(frame)

    # Break the loop
    if key == ord("q"):
        break