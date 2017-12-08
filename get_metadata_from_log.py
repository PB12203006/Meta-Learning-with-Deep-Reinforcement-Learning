#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
import numpy as np

'''
dict_keys:
    'ClassEntropy',
    'SymbolsSum', 
    'SymbolsSTD', 
    'SymbolsMean', 
    'SymbolsMax', 
    'SymbolsMin', 
    'ClassProbabilitySTD', 
    'ClassProbabilityMean', 
    'ClassProbabilityMax', 
    'ClassProbabilityMin', 
    'InverseDatasetRatio', 
    'DatasetRatio', 
    'RatioNominalToNumerical', 
    'RatioNumericalToNominal', 
    'NumberOfCategoricalFeatures', 
    'NumberOfNumericFeatures', 
    'NumberOfMissingValues', 
    'NumberOfFeaturesWithMissingValues', 
    'NumberOfInstancesWithMissingValues', 
    'NumberOfFeatures', 
    'NumberOfClasses', 
    'NumberOfInstances', 
    'LogInverseDatasetRatio', 
    'LogDatasetRatio', 
    'PercentageOfMissingValues', 
    'PercentageOfFeaturesWithMissingValues', 
    'PercentageOfInstancesWithMissingValues', 
    'LogNumberOfFeatures', 
    'LogNumberOfInstances', 
    'LandmarkRandomNodeLearner', 
    'SkewnessSTD', 
    'SkewnessMean', 
    'SkewnessMax', 
    'SkewnessMin', 
    'KurtosisSTD', 
    'KurtosisMean', 
    'KurtosisMax', 
    'KurtosisMin'
'''

metadata_dict_keys = ['ClassEntropy', 'SymbolsSum', 'SymbolsSTD', 'SymbolsMean', 'SymbolsMax', 'SymbolsMin', 'ClassProbabilitySTD', 'ClassProbabilityMean', 'ClassProbabilityMax', 'ClassProbabilityMin', 'InverseDatasetRatio', 'DatasetRatio', 'RatioNominalToNumerical', 'RatioNumericalToNominal', 'NumberOfCategoricalFeatures', 'NumberOfNumericFeatures', 'NumberOfMissingValues', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues', 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio', 'LogDatasetRatio', 'PercentageOfMissingValues', 'PercentageOfFeaturesWithMissingValues', 'PercentageOfInstancesWithMissingValues', 'LogNumberOfFeatures', 'LogNumberOfInstances', 'LandmarkRandomNodeLearner', 'SkewnessSTD', 'SkewnessMean', 'SkewnessMax', 'SkewnessMin', 'KurtosisSTD', 'KurtosisMean', 'KurtosisMax', 'KurtosisMin']

def parse_num_from_str(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def get_metadata_from_log(data_set_num):
    model_hyperparameters_filename = "./log/classifier_log" + str(data_set_num) + "/AutoML(1):simulated" + str(data_set_num) + ".log"
    print("Parsing metadata/metafeatures from " + model_hyperparameters_filename)
    metafeatures_filename = "./log/classifier_log" + str(data_set_num) + "/metafeatures.json"
    with open(model_hyperparameters_filename) as f:
        parse_line = False
        count = 0
        metafeatures = {}
        for line in f:
            if not parse_line and 'Metafeatures for dataset' in line and '[INFO]' in line:
                count += 1
                parse_line = True
            elif parse_line:
                l = line.split(':')
                if len(l) == 2:
                    metafeatures[l[0].strip()] = parse_num_from_str(l[1].strip().replace('\n', ''))
                else:
                    #print(metafeatures)
                    break
        with open(metafeatures_filename, 'w') as fp:
            json.dump(metafeatures, fp)
    return metafeatures

def encode_metadata_vector(data_set_index):
    
    #np.savetxt('Data_Set/X_' + str(data_set_index), X)
    #np.savetxt('Data_Set/y_' + str(data_set_index), y)
    metafeatures_filename = "./log/classifier_log" + str(data_set_index) + "/metafeatures.json"
    metafeatures = {}
    with open(metafeatures_filename) as mf:
        metafeatures = json.load(mf)
    metafeatures_vector = []
    for key in metadata_dict_keys:
        if key in metafeatures:
            metafeatures_vector.append(metafeatures[key])
    metafeatures_vector_filename = './log/classifier_log' + str(data_set_index) + '/metafeatures_vector.txt' 
    #with open(metafeatures_vector_filename, 'w') as mv:
    np.savetxt(metafeatures_vector_filename, metafeatures_vector)
    return metafeatures_vector
        
    
if __name__ == "__main__":
    generate_range = range(int(sys.argv[1]), int(sys.argv[2]))
    for data_set_index in generate_range:
        try:
            encode_metadata_vector(data_set_index)
        except Exception as err:
            print("can not encode metadata of index: " + str(data_set_index))
            print("EXCEPTION {0}".format(err))
            pass
    