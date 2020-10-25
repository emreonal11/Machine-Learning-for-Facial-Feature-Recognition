#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:52:39 2020

@author: liamparker & emreonal
"""

import numpy as np
from numpy import linalg
from sklearn.random_projection import SparseRandomProjection


class PCA:
    def __init__(self, variance_retained, train, test, grayscale, create_load_PCA):
        """
        :param variance_retained->int: represents the variance retention of PCA
        :param train->array: represents the training data set
        :param test->array: represents the validation data set
        :param test_img->array: represents the test image, if there is one
        :param grayscale->boolean: represents grayscale preference
        :param N->int: represents the total number training and validation images
        :param clpca->boolean: represents the desire to create PCA load file
        """
            
        self.vr        = variance_retained
        self.train     = train
        self.test      = test
        self.gr        = grayscale
        self.N         = self.train.shape[0] + self.test.shape[0]
        self.clpca     = create_load_PCA
      
        
    def analysis(self):
        """Performs PCA with given variance retained"""
        train_processed, test_processed = self.projection()
        train_covariance = np.dot(train_processed.T, train_processed)
        train_val, train_vec = linalg.eigh(train_covariance)
        
        idx = train_val.argsort()[::-1]
        train_val = train_val[idx]
        train_vec = train_vec[:, idx]       
        
        train_variances = []
        for i in range(len(train_val)):
            train_variances.append(train_val[i] / np.sum(train_val))  
        
        sum_var=0
        train_idx = 0
        for i in range(len(train_variances)):
            if sum_var >= self.vr:
                break
            else:
                sum_var += train_variances[i]
                train_idx += 1
        
        big_train_eigenvectors = train_vec[:,:train_idx]    
        train_data = np.dot(train_processed, big_train_eigenvectors)
        test_data = np.dot(test_processed, big_train_eigenvectors)
        
        if self.clpca == True:
            self.store_PCA(train_data, test_data)
        
        return train_data, test_data#, test_img_new 
        
    def projection(self):
        """Performs sparse random projection to reduce dimensionality of data"""
        transformer = SparseRandomProjection()
        train_new = transformer.fit_transform(self.train)
        test_new = transformer.transform(self.test)
        return train_new, test_new
        
    def store_PCA(self, train_data, test_data):
        """Stores the PCA file"""
        gry = ''
        if self.gr:
            gry += 'gray' 
        filename = 'PCA'+str(self.N)+gry+'Train'
        np.save(filename, train_data)
        filename = 'PCA'+str(self.N)+gry+'Test'
        np.save(filename, test_data)
        
        
        

