#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json

"""
balancing:strategy : none/weighting

imputation:strategy : not mean / mean
imputation:strategy : not median / median
imputation:strategy : not most_frequent / most_frequent

rescaling:__choice__ : not none / none
rescaling:__choice__ : not minmax/minmax
rescaling:__choice__ : not normalize/normalize
rescaling:__choice__ : not standardize/standardize

preprocessor:__choice__ : no_preprocessing/PCA
classifier:__choice__ : bernoulli_nb/qda

one_hot_encoding:use_minimum_fraction: True/False
one_hot_encoding:minimum_fraction

preprocessor:pca:keep_variance
preprocessor:pca:whiten : True/False

classifier:bernoulli_nb:alpha
classifier:bernoulli_nb:fit_prior : True/False
classifier:qda:reg_param

"""


model_hyperparameters_list = [{"balancing:strategy" : {"none": 0, "weighting" : 1}},
                              {"imputation:strategy" : {"mean": 0, "median": 1, "most_frequent" : 2}},
                              {"rescaling:__choice__" : {"none": 0, "minmax": 1, "normalize": 2, "standardize": 3}},
                              {"preprocessor:__choice__" : {"no_preprocessing" : 0, "pca": 1}},
                              {"classifier:__choice__" : {"bernoulli_nb": 0, "qda": 1}},
                              
                              {"one_hot_encoding:use_minimum_fraction" : {"True" : 0, "False" : 1}},
                              {"one_hot_encoding:minimum_fraction" : [0.0001, 0.5]},
                              
                              {"preprocessor:pca:keep_variance" : [0.5, 0.9999]},
                              {"preprocessor:pca:whiten" : {"True": 0, "False": 1}},
                              
                              {"classifier:bernoulli_nb:alpha" : [0.01, 100.0]},
                              {"classifier:bernoulli_nb:fit_prior" : {"True" : 0, "False" : 1}},
                              {"classifier:qda:reg_param" : [0.0, 1.0]}
                              ]
                              #{"classifier:libsvm_svc:C" : [0.03125, 32768.0]},
                              #{"classifier:libsvm_svc:gamma" : [3.0517578125e-05, 8.0]},
                              #{"classifier:libsvm_svc:kernel" : {"rbf" : 0, "poly" : 1, "sigmoid" :2}},
                              #{"classifier:libsvm_svc:max_iter" : [-1, 5000]},
                              #{"classifier:libsvm_svc:shrinking" : {"True" : 0, "False" : 1}},
                              #{"classifier:libsvm_svc:tol" : [1e-05, 0.1]},
                              #{"classifier:libsvm_svc:coef0" : [-1.0, 1.0]},
                              #{"classifier:libsvm_svc:degree" : [1, 5]}
                              

model_hyperparameters_dict = {"balancing:strategy" : {"none", "weighting"},
                              "imputation:strategy" : {"mean", "median", "most_frequent" },
                              "rescaling:__choice__" : {"none", "minmax", "normalize", "standardize"},
                              "preprocessor:__choice__" : {"no_preprocessing", "pca"},
                              "classifier:__choice__" : {"bernoulli_nb", "qda"},
                              
                              "one_hot_encoding:use_minimum_fraction" : {"True", "False"},
                              "one_hot_encoding:minimum_fraction" : [0.0001, 0.5],
                              
                              "preprocessor:pca:keep_variance" : [0.5, 0.9999],
                              "preprocessor:pca:whiten" : {"True", "False"},
                              
                              "classifier:bernoulli_nb:alpha" : [0.01, 100.0],
                              "classifier:bernoulli_nb:fit_prior" : {"True" , "False"},
                              "classifier:qda:reg_param" : [0.0, 1.0]
                              }
                              #"classifier:libsvm_svc:C" : [0.03125, 32768.0],
                              #"classifier:libsvm_svc:gamma" : [3.0517578125e-05, 8.0],
                              #"classifier:libsvm_svc:kernel" : {"rbf", "poly", "sigmoid"},
                              #"classifier:libsvm_svc:max_iter" : [-1,5000],
                              #"classifier:libsvm_svc:shrinking" : {"True", "False"},
                              #"classifier:libsvm_svc:tol" : [1e-05, 0.1],
                              #"classifier:libsvm_svc:coef0" : [-1.0, 1.0],
                              #"classifier:libsvm_svc:degree" : [1, 5]
                             

def one_hot_encode(dim, num):
    """
    Converts each scaling index in configspace to vector with one_hot encoding
    """
    if dim > 2:
        y_one_hot = np.zeros(dim)
        if num != -1:
            y_one_hot[int(num)] = 1
    else:
        y_one_hot = np.zeros(1)
        if num != -1:
            y_one_hot[0] = int(num)
    return y_one_hot

def encode_model(model_hyperparameters):
    """
    Encode model and hyperparameters into number list
    """
    global model_hyperparameters_list
    encoded_model = []
    for param_dict in model_hyperparameters_list:
        for key, value in param_dict.items():
            if key not in model_hyperparameters and type(value) is dict:
                encoded_model.extend(one_hot_encode(len(value), -1)) # this parameter does not show up in this model selection
            elif key not in model_hyperparameters and type(value) is list:
                encoded_model.append(value[0]) # this parameter does not show up in this model selection
            elif key in model_hyperparameters and type(value) is dict:
                encoded_model.extend(one_hot_encode(len(value), value[model_hyperparameters[key]]))
            elif key in model_hyperparameters and type(value) is list:
                encoded_model.append(float(model_hyperparameters[key]))
    return encoded_model
            
            

def get_encoded_models_for_data_set(data_set_num):
    """
    Parse AutoML tried models and hyperpatameters log and encode information into matrix
    """
    global model_hyperparameters_dict
    tried_models_hyperparameters_encode = []
    tried_models = {}
    model_hyperparameters_filename = "./log/classifier_log" + str(data_set_num) + "/AutoML(1):simulated" + str(data_set_num) + ".log"
    print("Parsing models and hyperparameters from " + model_hyperparameters_filename)
    with open(model_hyperparameters_filename) as f:
        parse_line = False
        count = 0
        for line in f:
            if "Function called with argument: ()," in line and "'config': Configuration:" in line:
                count += 1
                tried_models[count] = {}
                parse_line = True
            elif parse_line and ", 'num_run':" in line:
                parse_line = False
                tried_models_hyperparameters_encode.append(encode_model(tried_models[count]))
            elif parse_line and len(line.split(", Value:")) == 2:
                key = line.split(", Value:")[0].strip()
                val = line.split(", Value:")[1].strip().replace("'","").replace("\n", "")
                if type(model_hyperparameters_dict[key]) is list: # and key != "classifier:libsvm_svc:degree":
                    val = float(val)
                tried_models[count][key] = val

        # print(tried_models)
        # print(tried_models_hyperparameters_encode)
    tried_models_filename = "./log/classifier_log" + str(data_set_num) + "/tried_models_for_dataset" + str(data_set_num) + ".json"
    with open(tried_models_filename, 'w') as fp:
        json.dump(tried_models, fp)
    tried_models_hyperparameters_encode_filename = "./log/classifier_log" + str(data_set_num) + "/encoded_tried_models_hyperparameters_for_dataset" + str(data_set_num) + ".txt"
    np.savetxt(tried_models_hyperparameters_encode_filename, tried_models_hyperparameters_encode)
    print("Saved AutoML tried models and hyperparameters in file: " + tried_models_filename)
    print("Saved encoded AutoML tried models and hyperparameters matrix in file: " + tried_models_hyperparameters_encode_filename)
    return tried_models_hyperparameters_encode, tried_models
                
def test():
    # generate model selections and encoded models for each dataset
    for i in range(2):
        tried_models_hyperparameters_encode, tried_models = get_encoded_models_for_data_set(i)
        
        
#if __name__ == "__main__":
#    test()