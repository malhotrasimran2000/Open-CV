# Face Detection and Recognition

## Introduction
This project uses Open CV and the Haarcascade Classifier for face detection and the collected data is trained using KNN classification model.

## Abstract 
Object Detection using Haar feature-based classifiers is an effective object detection method.It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images.It is then used to detect objects in other images.

The K-nearest neighbours (KNN) algorithm is a type of supervised machine learning algorithm.It simply calculates the distance of a new data point to all other training data points. The distance can be of any type e.g Euclidean or Manhattan etc. Once the distances of the new data points (to be classified) wrt to all the other data points are calculated, they are tabulated and then sorted in an ascending order. Then an input of the K value by a user is used to select that K number of nearest data points and a majority vote is considered and depending on the distance of the new data points from the existing ones, they are classified accordingly. 

## Prerequisites
* A lot of modules used in the code are part of OpenCV's contrib package (link [here](https://github.com/opencv/opencv_contrib))  
* Follow the steps for installation [here](https://pypi.org/project/opencv-contrib-python/)  
    * Make sure to first uninstall cv2 ``` pip uninstall cv2 ```  
    * Then install contrib package ``` pip install opencv-contrib-python ```  
*Download the required XML files [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## Files
* ```Data folder``` contains face data
* ```collect.py``` file detects the face and collects face data 
* ```test.py``` file detects the face and recognises the person

## Resources
* Face Detection using Haar Cascasdes(link[Click here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html))
* KNN Classification(link[Click here](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761))
