#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:52:39 2020

@author: liamparker & emreonal
"""

import numpy as np

class data_loading:
    def __init__(self, N, split, grayscale):
        """
        :param N->int: represents the total number training and validation images
        :param split->double: represents the desired training split
        :param grayscale->boolean: represents desired grayscale option
        """
        
        training_number = N*split
        if training_number != int(training_number):
            raise ValueError("N*split must be an integer")
            
        self.N         = N
        self.split     = split
        self.gr        = grayscale
        self.tr        = int(training_number)
        
    def data_cutoff(self, centered, no_face_array):
        """Generates cutoff between training and validation data sets"""
        train_no_face = 0
        test_no_face = 0
        for val in no_face_array:
            if val < self.tr:
                train_no_face += 1
            else:
                test_no_face += 1
        
        cutoff = self.tr - train_no_face
        end = self.N - test_no_face
        
        train = centered[:cutoff]
        test = centered[cutoff:end]
        
        return train, test
    
    def load_data(self):
        string = ''
        if self.gr == True:
            string += 'gray'
        filename ='centered'+str(self.N)+string+'.npy'
        nofacename='noface'+str(self.N)+string+'.npy'
        centered = np.load(filename)
        no_face_array = np.load(nofacename)
            
        train, test = self.data_cutoff(centered, no_face_array)

        return train, test, no_face_array
    
    def load_PCA(self):
        gry = ''
        if self.gr == True:
            gry += 'gray'
            
        filename = 'PCA'+str(self.N)+gry+'Train.npy'
        train = np.load(filename)
        filename = 'PCA'+str(self.N)+gry+'Test.npy'
        test = np.load(filename)
        
        return train, test

