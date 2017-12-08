#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After generate encoded metafeatures txt file + encoded model choice txt file + performance txt file
run this python script to integrate all encoded data to matrix as input for RNN

"""
import numpy as np

def integrate_encoded_data_for_one_dataset(data_set_index):
    metafeatures_vector = []
    model_choice_matrix = []
    performance_matrix = []
    
    # metadata vector
    metafeatures_vector_filename = './log/classifier_log' + str(data_set_index) + '/metafeatures_vector.txt' 
    metafeatures_vector = np.loadtxt(metafeatures_vector_filename)
    
    # encoded model choice
    models_hyperparameters_encode_filename = "./log/classifier_log" + str(data_set_index) + "/encoded_tried_models_hyperparameters_for_dataset" + str(data_set_index) + ".txt"
    model_choice_matrix = np.loadtxt(models_hyperparameters_encode_filename)
    
    # performance vector
    performance_matrix_filename = "./log/classifier_log" + str(data_set_index) + "/encode_reproduce_performance_matrix" + str(data_set_index) + ".txt"
    performance_matrix = np.loadtxt(performance_matrix_filename)
    
    row_num = performance_matrix.shape[0]
    if metafeatures_vector.shape[0] != 38:
        raise ValueError('the number of metafeatures of data set #{0} is not 38'.format(data_set_index))
    
    # convert metafeatures vector to matrix
    metafeatures_matrix = np.repeat(metafeatures_vector.reshape(1, len(metafeatures_vector)), row_num, axis=0)
    
    # select models reproduced with performace
    model_choice_matrix = model_choice_matrix[:row_num]
    
    return metafeatures_matrix, model_choice_matrix, performance_matrix
    

def integrate_encoded_data_for_datasets(dataset_range):
    metafeatures_matrix = None
    model_choice_matrix = None
    performance_matrix = None
    for i in dataset_range:
        try:
            single_metafeatures_matrix, single_model_choice_matrix, single_performance_matrix = integrate_encoded_data_for_one_dataset(i)
            if metafeatures_matrix:
                np.concatenate((np.array(metafeatures_matrix),single_metafeatures_matrix),axis=0)
            else:
                metafeatures_matrix = single_metafeatures_matrix
            
            if model_choice_matrix:
                np.concatenate((model_choice_matrix,single_model_choice_matrix),axis=0)
            else:
                model_choice_matrix = single_model_choice_matrix
                
            if performance_matrix:
                np.concatenate((performance_matrix,single_performance_matrix),axis=0)
            else:
                performance_matrix = single_performance_matrix
            
        except Exception as err:
            print("ERROR occuerred when process dataset #{0}".format(i))
            print("ERROR: {0}".format(err))
            pass
    print(np.array(performance_matrix).shape)
    return metafeatures_matrix, model_choice_matrix, performance_matrix

generate_range = range(0,10)
integrate_encoded_data_for_datasets(generate_range)

    