#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:52:39 2020

@author: emreonal & liamparker
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
import numpy as np
import time

class model:
    def __init__(self, train, test, train_label, test_label):
        """
        :param train->array: represents the training data set
        :param test->array: represents the validation data set
        :param train_label->list: represents the training label list
        :param test_label->list: represents the validation label list
        """
            
        self.train     = train
        self.test      = test
        self.trl       = train_label
        self.tel       = test_label
    

    @ignore_warnings(category=ConvergenceWarning)
    def logistic_regression(self):
        """Logistic regressor model from SKLearn""" 
        clf = LogisticRegressionCV(penalty = 'l2', max_iter = 500, solver = 'saga')
        clf.fit(self.train, self.trl)
        score = clf.score(self.test, self.tel)
        return clf, score

    def CNN(self, loadModel, feature):
        """Input CNN code"""
        print(self.train.shape)
        
        N = self.train.shape[0] + self.test.shape[0]
        
        train = np.reshape(self.train, (self.train.shape[0], 178, 218, 3))
        test = np.reshape(self.test, (self.test.shape[0], 178, 218, 3))

        y_train = to_categorical(self.trl, num_classes = 2)
        y_test = to_categorical(self.tel, num_classes = 2)
        
        if loadModel:
            print('loading model')
            model = keras.models.load_model('C:/Users/onale/Desktop/Project/'+feature)
        else:
            #create model
            model = Sequential()
    
            #convolutional layers
            model.add(Conv2D(filters = 32, kernel_size=3, strides = 1, padding = 'same', 
                         input_shape=(178,218,3), activation ='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
        
            model.add(Conv2D(filters = 64, kernel_size=3, strides = 1, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
        
            model.add(Conv2D(filters = 128, kernel_size=3, strides=1, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
        
            model.add(Conv2D(filters = 256, kernel_size=3, strides=1, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
  
        
            #fully connected layers
            model.add(Flatten())
            model.add(Dense(units = 64))
            model.add(Activation('relu'))
            model.add(Dropout(0.05))
        
            model.add(Dense(units = 32))
            model.add(Activation('relu'))
            model.add(Dropout(0.05))
    
            model.add(Dense(2, activation='softmax'))
    
            #compile model using accuracy as a measure of model performance
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
        #train model
        model.fit(train, y_train,validation_data=(test, y_test), epochs=3)
        model.save('C:/Users/onale/Desktop/Project/'+feature)
        score = clf.score(self.test, self.tel)  #change
        return model, score


