#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:52:39 2020

@author: emreonal & liamparker
"""
import time
from data_importing import data_importing
from data_loading import data_loading
from PCA import PCA
import numpy as np
from model import model

"""Edit these paths per user requirements"""
directory = 'C:/Users/onale/Desktop/Project/faces2/'
attributes = 'C:/Users/onale/Desktop/Project/attributes2.txt'
test_img = 'C:/Users/onale/Desktop/Project/Emre.jpeg'

"""Edit these variables to adjust model specifications"""
N                 = 2000     #Number of images in training + validation
split             = 0.8     #Split between training + validation
variance_retained = 0.9     #Variance retention for principal component analysis

load              = True   #Load data from prebuilt file
load_PCA          = True   #Load PCA data from prebuilt file
load_cnn          = True   #Load a previously trained CNN
recognize         = True    #Recognize faces
grayscale         = False    #Grayscale images
create_load       = False    #Create load file
create_load_PCA   = False    #Create PCA load file
epochs            = 3        #Number of epochs to train CNN

features          = ['Male']
model_type        = 'CNN'

#Features:     'Male', 'Attractive', 'Black Hair', 'Blonde Hair', 'Brown Hair', 'Chubby', 
#              'Pale', 'Straight Hair', 'Wavy Hair'


def print_settings():
    """Helper function to print settings"""
    print('Settings: ')
    string = ''
    for feature in features:
        if feature == features[len(features)-1]:
            string += feature
            break
        string+= feature + ', '
    print('features: ' + string)
    print('samples: '+str(N))
    print('load_data: '+str(load))
    print('load_cnn: '+str(load_cnn))
    print('train-split: '+str(split))
    print('grayscale: '+str(grayscale))
    print('recognize-face: '+str(recognize)) 
    print('create-load: '+str(create_load))
    print('create-load-PCA: '+str(create_load_PCA))

def convert_time(tsecs):
    """Helper function to convert time"""
    if (tsecs > 3600):
        hours = int(tsecs / 3600)
        timeleft = tsecs - ( hours * 3600)
        minutes = int(timeleft/60)
        seconds = round(timeleft - (minutes * 60))
        string = str(hours) + ' hour(s), ' + str(minutes) + ' minute(s), ' + str(seconds) + ' second(s)'
        
    elif (tsecs > 60):
        minutes = int(tsecs/60)
        seconds = round(tsecs - (minutes * 60))
        string = str(minutes) + ' minute(s), ' + str(seconds) + ' second(s)'
    else:
        string = str(round(tsecs))+' second(s)'
    return string

def print_time(string, tlast):
    """Prints time after each round of processing"""
    if string == 'Data Importing':
        print('')
        print('Runtimes: ')
    if string == 'Total Time Elapsed':
        print('')
    tnow = time.time()
    print(string + ': ' + str(convert_time(tnow-tlast)))
    tlast = tnow
    return tlast

def convert_labels(train_label, test_label):
    train_y = np.zeros(len(train_label))
    i = 0
    for label in train_label:
        if label == '-1':
            train_y[i] = 0
        else:
            train_y[i] = 1
        i += 1

    test_y = np.zeros(len(test_label))
    i = 0
    for label in test_label:
        if label == '-1':
            test_y[i] = 0
        else:
            test_y[i] = 1
        i += 1
    
    return train_y, test_y

def generate_scores(train, test, nfa, loadModel, feature):
    """Generate scores for each model of each feature"""
    all_scores = []
    
    for feature in features:
        data_object = data_importing(directory, N, split, grayscale, recognize, feature, attributes, create_load)
        train_label, test_label = data_object.process_labels(nfa)

        
        if model_type == 'CNN':
            train_label, test_label = convert_labels(train_label, test_label)
            
        model_object = model(train, test, train_label, test_label)
        
        if model_type == "Logistic Regression":
            _, score = model_object.logistic_regression()
        if model_type == "CNN":
            _, score = model_object.CNN(loadModel, feature)
        all_scores.append(score)
    
    return all_scores

if __name__ == '__main__':
    #Initialize time and print current settings
    print_settings()
    tstart = time.time()
    tlast = tstart
    
    #Load or import data per specifications
    if load==True:
        data_object = data_loading(N, split, grayscale)
        train, test, nfa = data_object.load_data()

    if load==False:
        data_object = data_importing(directory, N, split, grayscale, recognize, features[0], attributes, create_load)
        train, test, nfa = data_object.extract_data()
    
    #Print time after data processing has been completed
    tlast = print_time('Data Importing', tlast)
    
    
    if model_type == 'Logistic Regression':
        #Load or import PCA per specifications
        if load_PCA==True:
            data_object = data_loading(N, split, grayscale)
            train, test = data_object.load_PCA()
        
        if load_PCA==False:
            data_object = PCA(variance_retained, train, test, grayscale, create_load_PCA)
            train, test = data_object.analysis()
    
        #Print time after PCA has been completed
        tlast = print_time('Principal Component Analysis', tlast)
    
    #Generate scores of ML models on features
    scores = generate_scores(train, test, nfa, load_cnn, features[0])
    
    #Print time after model generation and scoring
    tlast = print_time('Model generation on all features', tlast)
    
    #Print out the model accuracies
    i = 0
    print('')
    print('Model Accuracies:')
    for feature in features:
        print(feature + ': ' + str(scores[i]))
        i += 1
    
    #Print out the total time elapsed
    print_time('Total Time Elapsed', tlast)


