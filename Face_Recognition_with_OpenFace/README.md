# Static Face Recognition: Leonardo DiCaprio, Emma Stone, Sandra Bullock, and Brad Pitt

## Objective

The purpose of this project is to build a face recognition algorithm to detect whether the person in an image is Leonardo DiCaprio, Emma Stone, Sandra Bullock, or Brad Pitt. We can expand the scope of the project by adding more people rather than just these four, but for simplicity only these four people will be considered.

Since it will be very expensive to train the model, then transfer learning approach is applied. The model that will be used in this project was taken from OpenFace pre-trained model, in which this model was trained using triplet loss cost function. 

Below is the example of the result obtained in this project.

<p align="center">
  <img width="1000" height="500" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Face_Recognition_with_OpenFace/pict.png">
</p>


## Files
There two files and three folders in this project, which are:

- test folder: contains tests images used in this project.
- train folder: contains the original example of images for face recognition.
- train_face folder: contains the face-isolated example of images for face recognition.
- Face_Recognition_Openface: the Jupyter Notebook file of this project which contains a step-by-step method applied in this project.
- openface_weights.h5: the pre-trained weight of pre-trained Inceprion ResNet V1 trained with triplet loss function.
