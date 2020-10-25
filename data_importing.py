#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 22:34:44 2020

@author: liamparker & emreonal
"""

import numpy as np
from numpy import asarray
from numpy import empty
import os
from os import walk
import PIL
from PIL import Image, ImageDraw
import sklearn
from sklearn import preprocessing
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from skimage import exposure
import csv
import cv2
from csv import writer

"""Add to this dictionary if you want to add more possible features"""
DICTIONARY = dict([
    ('Male', 21),
    ('Attractive', 3),
    ('Black Hair', 9),
    ('Blonde Hair', 10),
    ('Brown Hair', 12),
    ('Chubby', 14),
    ('Pale', 27),
    ('Straight Hair', 33),
    ('Wavy Hair', 34)
    ])

class data_importing:
    def __init__(self, directory, N, split, grayscale, facial_recognition, attribute, attribute_file, create_load):
        """
        :param directory: represents the directory from which to pull training and validation images
        :param N->int: represents the total number training and validation images
        :param split->double: represents the desired training split
        :param grayscale->boolean: represents desired grayscale option
        :param facial_recognition->boolean: represents desired facial recognition option
        :param training_number->int: represents the number of files in the training set
        :param attribute->string: represents the desired attribute expressed as a string
        :param attribute_file: represents the attribute file location
        :param create_load->boolean: represents desire to create data file for future loading
        """
        
        training_number = N*split
        if training_number != int(training_number):
            raise ValueError("N*split must be an integer")
        
        self.dir       = directory
        self.N         = N
        self.split     = split
        self.gr        = grayscale
        self.fr        = facial_recognition
        self.tr        = int(training_number)
        self.attr      = attribute
        self.attr_file = attribute_file
        self.cl        = create_load
        
        #Create files
        self.import_files()
        
        #Find dimensions of training set images
        self.get_dimensions()
        
    def import_files(self):
        "Imports sorted image files from given directory"
        for root,dirs,files in os.walk(self.dir, topdown = True): 
            files.sort()
        
        self.files = files
    
    def create_array(self):
        "Initializes image array with given dimensions"
        if self.gr == True:
            img_array = np.empty((self.N, self.HEIGHT, self.WIDTH))
        else:
            img_array = np.empty((self.N, self.HEIGHT, self.WIDTH, self.DEPTH))
        
        return img_array
    
    def generate_face(self, test_img):
        """Generate a face from the given image using prebuilt CV2 Haar Classifier"""
        
        img = cv2.imread(test_img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        box = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 4)
        
        if box == ():
            return None
        
        x = box[0][0]
        y = box[0][1]
        w = box[0][2]
        h = box[0][3]
        
        test_img = Image.open(test_img)
        cropped_img = test_img.crop((x, y, x + w, y + h))
        cropped_img = cropped_img.resize((self.WIDTH, self.HEIGHT))
        img = cropped_img
            
        return img
    
    def get_dimensions(self):
        """Get the dimensions of training set images"""
        sample = asarray(Image.open(self.dir+self.files[0])).shape
        
        #Determines the height and width of training set images
        self.HEIGHT = int(sample[0])
        self.WIDTH = int(sample[1])
        self.DEPTH = 3
    
    def histogram_equalization(self, img):
        """Perform histogram equalization on the image"""
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        data = exposure.rescale_intensity(img, in_range=(p2, p98))
        return data
    
    def fill_array(self, data, img_array, label):
        if self.gr == True:
            for i in range(self.HEIGHT):
                for j in range(self.WIDTH):
                    img_array[label][i][j] = data[i][j]
        if self.gr == False:
            for i in range(self.HEIGHT):
                for j in range(self.WIDTH):
                    for k in range(self.DEPTH):
                        img_array[label][i][j][k] = data[i][j][k]
        return img_array
        
    def delete_rows(self, img_array, no_face_array):
        """Deletes the empty rows of the imageg array as instructed by the no face array"""
        i = 0
        for val in no_face_array:
            img_array = np.delete(img_array, val-i, axis=0)
            i=i+1
        return img_array
    
    def reshape_array(self, img_array):
        """Reshape array to 2 dimensions based off of grayscale optionality"""
        length = self.N - self.no_face_len
        
        if self.gr == True:
            img_array = np.reshape(img_array, (length, self.HEIGHT*self.WIDTH))
    
        if self.gr == False: 
            img_array = np.reshape(img_array, (length, self.HEIGHT*self.WIDTH*self.DEPTH))
        
        return img_array
    
    def scaler(self, img_array):
        """Scales and centers data using prebuilt StandardScaler from SKLearn"""
        centered = StandardScaler().fit_transform(img_array)
        return centered
        
    def create_data_file(self, centered, no_face_array):
        """Creates data file for future loading"""
        gry = ''
        if self.gr:
            gry += 'gray' 
        length = self.N
        filename = 'centered'+str(self.N)+gry
        nofacename = 'noface'+str(self.N)+gry
        np.save(filename, centered)
        np.save(nofacename, no_face_array)
        
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
        
    def extract_data(self):
        """Extracts data from image files and compiles to 2-dimensional array"""
        #Creates label for row labelling and array for outlier processing
        label = 0
        no_face_array = [] 
        
        img_array = self.create_array()
        
        for file in self.files[:self.N]:
            if self.fr == True:
                img = self.generate_face(self.dir+file)
                face = img
            if self.fr == False:
                img = Image.open(self.dir+file)
                face = self.generate_face(self.dir+file)
            if face == None:
                no_face_array.append(label)
                label = label + 1
                continue
            if self.gr == True:
                img = img.convert('L')

            data = self.histogram_equalization(asarray(img))
            self.fill_array(data, img_array, label)
            label = label + 1
            
        #Declare length of no face array
        self.no_face_len = len(no_face_array)
        
        #Delete the empty rows, then reshape array
        img_array = self.delete_rows(img_array, no_face_array)
        img_array = self.reshape_array(img_array)
        
        #Centers and scales the image array
        centered = self.scaler(img_array)
        
        #Create data files for image array and no face array
        if self.cl == True:
            self.create_data_file(centered, no_face_array)
        
        #Cutoff the data
        train, test = self.data_cutoff(centered, no_face_array)
        
        return train, test, no_face_array
    
    def cutoff_labels(self, no_face_array, labels):
        """Cutoff the label array based off of values in no face array"""
        train_no_face = 0
        test_no_face = 0
        
        for val in no_face_array:
            if val < self.tr:
                train_no_face += 1
            else:
                test_no_face += 1
        
        cutoff = self.tr - train_no_face
        end = self.N - test_no_face
        train_labels = labels[:cutoff]
        test_labels = labels[cutoff:end]
        return train_labels, test_labels
        
        
    def process_labels(self, no_face_array):
        """Process the labels for a given attribute from the provided file"""
        file = open(self.attr_file)
        st = file.readlines()
        
        #Initialize label array
        labels = []
        i = 2
        
        col = int(DICTIONARY[self.attr])
        tf = False
        
        while (i < 2 + self.N):
            if i-2 in no_face_array:
                i = i+1
                continue
            line = st[i].split()
            labels.append(line[col])
            i = i + 1
        
        file.close()

        return self.cutoff_labels(no_face_array, labels)
    
        

    
    
        

    
    
        
        
    
        
    