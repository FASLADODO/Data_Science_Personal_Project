# -*- coding: utf-8 -*-
"""

@author: Marcellus Ruben Winastwan
"""

import numpy as np
import pandas as pd
import os
import function as f

########################## Data Wrangling #####################################

os.getcwd()

fullDataFrame = pd.read_csv('RawData.txt', delimiter="\t", names=["UserID","ItemID","Rating","Timestamp"])

# Remove "Timestamp" column as it is not needed for this project.

dataFrame = fullDataFrame.drop(['Timestamp'], axis= 1)

# Get the total number of user and movie in the dataframe.

totalUser = len(dataFrame.UserID.value_counts())
totalMovie = len(dataFrame.ItemID.value_counts())
#print((totalUser, totalMovie))

# Create a complete matrix of the training set.

dataArray = dataFrame.to_numpy()

Y = np.zeros((totalMovie, totalUser))
R = np.zeros((totalMovie, totalUser))

for i in range (len(dataArray)):
        
        # -1 because Python indexing starts from 0.
        
        Y[(dataArray[i,1]-1), (dataArray[i,0]-1)] = dataArray[i,2]

# Initialize R matrix, which has a value of 1 if a user has rated a movie, or 0 otherwise.
        
for i in range (totalMovie):
    for j in range (totalUser):

        if Y[i,j] != 0:
            R[i,j] = 1


# Initialize movie list.

movieList = f.loadMovieList('movie_ids.txt')

# Initialize my movie rating.

myMovieRating = np.zeros((totalMovie,1))

# Movie I like
myMovieRating[169] = 5 #Cinema Paradiso
myMovieRating[215] = 5 #When Harry Met Sally
myMovieRating[245] = 4 #Chasing Amy
myMovieRating[250] = 5 #Shall We Dance?
myMovieRating[353] = 4 #The Wedding Singer

# Movie I dislike
myMovieRating[352] = 2 #Deep Rising
myMovieRating[372] = 1 #Judge Dredd
myMovieRating[478] = 2 #Vertigo
myMovieRating[591] = 2 #True Crime
myMovieRating[602] = 2 #Rear Window
myMovieRating[690] = 1 #Dark City

# Create R Matrix based on movie that I rated
myR = np.zeros((len(myMovieRating),1))
for i in range (len(myMovieRating)):
    
    if myMovieRating[i] != 0:
        myR[i] = 1
        
# Print my rating and the corresponding movie title.

for i in range (len(myMovieRating)):
    
    if myMovieRating[i] != 0:
        print('I rated', movieList[i], int(myMovieRating[i]),' stars')
        
Y = np.column_stack((myMovieRating,Y))
R = np.column_stack((myR,R))
##################### Start of Collaborative Filtering ########################
            
# Initialize random values for X (weights of genres in a movie) and Theta (user's preference of movie genre)

totalFeatures = 5
X = f.initializeWeight(totalMovie, totalFeatures)
# Theta +1 because I add my rating 
Theta = f.initializeWeight(totalUser+1, totalFeatures) 


# Set regularization parameter, learning rate, and number of iterations
lambdaReg = 0
alpha = 0.0001
numIterations = 20

# Perform optimization process for collaboration filtering algorithm
J, finX, finTheta = f.computeGradientDescentCollaborationFiltering(X, Y, Theta, alpha, numIterations, lambdaReg, R)


# Fetch my movie recommendation.
moviePredictions = np.matmul(finX,finTheta.transpose())
recommendationForMe = moviePredictions[:,0]


indices = np.argsort(recommendationForMe)[::-1]

# Print my movie recommendation.
for i in range (10):
    
    idx = indices[i]
    
    if i == 0:
        print('Top movie recommendations for you:')
    print('No',[i+1],':',movieList[idx])