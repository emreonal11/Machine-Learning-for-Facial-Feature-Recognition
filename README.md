# Machine-Learning-for-Facial-Feature-Recognition
A logistic regression or convolutional neural network that is used to detect certain facial features from images, including gender, hair color, eye color, etc. The model is trained on the CelebFaces attributes dataset.

I am currently working with another Princeton student on programming (in Python) an app that uses machine learning techniques to classify human faces across a variety of labels
like gender, age, hair color, eye color, etc. We first pre-process the images with histogram equalization and intensity normalization. Then, we crop around the face using the 
MTCNN face detection algorithm and feed the resulting image to the selected machine learning model. Thus far we have implemented a logistic regression model 
(using principal component analysis) and a convolutional neural network (CNN). The logistic regression model had an average accuracy of 90% across all labels, while the CNN 
performed at about 93% accuracy.


