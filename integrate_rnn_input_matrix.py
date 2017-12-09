#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After generate encoded metafeatures txt file + encoded model choice txt file + performance txt file
run this python script to integrate all encoded data to matrix as input for RNN

"""
import numpy as np
import traceback
import sys

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
    if row_num != 30:
        raise Exception('No PERFORMANCE txt file in dataset #' + str(data_set_index))
    if metafeatures_vector.shape[0] != 38:
        raise Exception('the number of metafeatures of data set #{0} is not 38'.format(data_set_index))
    
    # convert metafeatures vector to matrix
    metafeatures_matrix = np.repeat(metafeatures_vector.reshape(1, len(metafeatures_vector)), row_num, axis=0)
    
    # select models reproduced with performace
    model_choice_matrix = model_choice_matrix[:row_num]
    
    return metafeatures_matrix, model_choice_matrix, performance_matrix
    

def integrate_encoded_data_for_datasets(dataset_range):
    metafeatures_matrix = []
    input_model_choice_matrix = []
    predict_model_choice_matrix = []
    input_performance_matrix = []
    for i in dataset_range:
        try:
            single_metafeatures_matrix, single_model_choice_matrix, single_performance_matrix = integrate_encoded_data_for_one_dataset(i)
            print('DATA_SET #' + str(i))
            
            metafeatures_matrix.append(single_metafeatures_matrix)
            
            single_model_choice_matrix = np.concatenate((single_model_choice_matrix[:, :14], np.log10(single_model_choice_matrix[:,14]).reshape(30,1) , single_model_choice_matrix[:, 15:]), axis=1)
            #print('single_model_choice_matrix shape : {0}'.format(single_model_choice_matrix.shape))
            input_model = np.concatenate((np.zeros((1,17)), single_model_choice_matrix[:-1]),axis=0)
            input_model_choice_matrix.append(input_model)
            predict_model_choice_matrix.append(single_model_choice_matrix)
                
            input_performance = np.concatenate((np.zeros((1,4)), single_performance_matrix[:-1]),axis=0)
            input_performance_matrix.append(input_performance)
            
            print('input_performance_matrix shape : {0}'.format(np.array(input_performance_matrix).shape))
            print('metafeatures_matrix shape: {0}'.format(np.array(metafeatures_matrix).shape))
            print('input_model_choice_matrix shape : {0}'.format(np.array(input_model_choice_matrix).shape))
            print('predict_model_choice_matrix shape : {0}'.format(np.array(predict_model_choice_matrix).shape))
            print('=============================================')
        except Exception as err:
            print("ERROR occuerred when process dataset #{0}".format(i))
            print("ERROR: {0}".format(err))
            pass
    print(np.array(input_performance_matrix).shape)
    return np.array(metafeatures_matrix), np.array(input_model_choice_matrix), np.array(predict_model_choice_matrix), np.array(input_performance_matrix)

if __name__ == '__main__':
    generate_range = range(int(sys.argv[1]), int(sys.argv[2]))
    integrate_encoded_data_for_datasets(generate_range)

    