#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
import sys
from get_autosklean_tried_models_hyperparameters import model_hyperparameters_list
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, KFold

performance_keys = ["train_accuracy_score", "test_accuracy_score", "train_log_loss", "test_log_loss"]

def reverse_model_hyperparameters_list():
    # global model_hyperparameters_list
    reversed_model_hyperparameters_list = []
    for param_dict in model_hyperparameters_list:
        for key, value in param_dict.items():
            if type(value) is dict:
                reversed_model_hyperparameters_list.append({key : {v : k for k, v in value.items()}})
            else:
                reversed_model_hyperparameters_list.append({key : value})
    return reversed_model_hyperparameters_list

def decode_model(encoded_model_hyperparameters):
    model_hyperparameters_decode_list = reverse_model_hyperparameters_list()
    #print(encoded_model_hyperparameters)
    decoded_model = {}
    idx = 0
    for i in range(6):
        param_dict = model_hyperparameters_decode_list[i]
        for key, values in param_dict.items():
            if type(values) is dict and len(values) > 2:
                for k in range(len(values)):
                    if encoded_model_hyperparameters[idx + k] == 1:
                        decoded_model[key] = values[k]
                        break
                idx += len(values)
            else:
                decoded_model[key] = values[int(encoded_model_hyperparameters[idx])]
                idx += 1
    one_hot_encoding_min_fraction = decoded_model['one_hot_encoding:use_minimum_fraction']
    if one_hot_encoding_min_fraction == 'True':
        decoded_model['one_hot_encoding:minimum_fraction'] = float(encoded_model_hyperparameters[idx])
        assert(decoded_model['one_hot_encoding:minimum_fraction']>0.0001)
        assert(decoded_model['one_hot_encoding:minimum_fraction']<0.5)
        assert(isinstance(decoded_model['one_hot_encoding:minimum_fraction'], float))
        #print(decoded_model)
    idx += 1

    #decoded_model['preprocessor:__choice__'] = values[int(encoded_model_hyperparameters[idx])]
    #idx += 1
    preprocessor = decoded_model['preprocessor:__choice__']
    start = 7
    end = start + 2
    idx, decoded_model = parse_encode_field(preprocessor, idx, start, end, model_hyperparameters_decode_list, encoded_model_hyperparameters, decoded_model)
    #decoded_model['classifier:__choice__'] = values[int(encoded_model_hyperparameters[idx])]
    #idx += 1
    classifier = decoded_model['classifier:__choice__']
    start = 9
    end = start + 3
    idx, decoded_model = parse_encode_field(classifier, idx, start, end, model_hyperparameters_decode_list, encoded_model_hyperparameters, decoded_model)
    if 'preprocessor:pca:keep_variance' in decoded_model:
        decoded_model['preprocessor:pca:keep_variance'] = float(decoded_model['preprocessor:pca:keep_variance'])
        assert(isinstance(decoded_model['preprocessor:pca:keep_variance'], float))
    if 'classifier:bernoulli_nb:alpha' in decoded_model:
        decoded_model['classifier:bernoulli_nb:alpha'] = float(decoded_model['classifier:bernoulli_nb:alpha'])
        assert(isinstance(decoded_model['classifier:bernoulli_nb:alpha'], float))
    if 'classifier:qda:reg_param' in decoded_model:
        decoded_model['classifier:qda:reg_param'] = float(decoded_model['classifier:qda:reg_param'])
        assert(isinstance(decoded_model['classifier:qda:reg_param'], float))
    return decoded_model
    
def parse_encode_field(keyword, idx, start, end, model_hyperparameters_decode_list, encoded_model_hyperparameters, decoded_model):
    for i in range(start, end):
        param_dict = model_hyperparameters_decode_list[i]
        for key, values in param_dict.items():
            if type(values) is dict and len(values) > 2:
                for k in range(len(values)):
                    if keyword in key and encoded_model_hyperparameters[idx + k] == 1:
                        decoded_model[key] = values[k]
                        break
                idx += len(values)
            elif type(values) is dict:
                if keyword in key :
                    decoded_model[key] = values[int(encoded_model_hyperparameters[idx])]
                idx += 1
            else:           
                if keyword in key :
                    decoded_model[key] = float(encoded_model_hyperparameters[idx])
                idx += 1
    return idx, decoded_model
    #decoded_model['one_hot_encoding:use_minimum_fraction'] = False
    '''
    
    if 'classifier:libsvm_svc:kernel' in decoded_model:
        kernel = decoded_model['classifier:libsvm_svc:kernel']
        if kernel == 'poly':
            decoded_model['classifier:libsvm_svc:coef0'] = float(encoded_model_hyperparameters[idx])
            decoded_model['classifier:libsvm_svc:degree'] = int(encoded_model_hyperparameters[idx + 1])
        elif kernel == 'sigmoid':
            decoded_model['classifier:libsvm_svc:coef0'] = float(encoded_model_hyperparameters[idx])
    #print(decoded_model)
    '''
    
def get_performance_of_encoded_model(data_set, encoded_model, verbose=False):
    """
    Get model performance from encoded vector
    """
    train_accuracy_score = []
    test_accuracy_score = []
    train_log_loss = []
    test_log_loss = []
    X, y = data_set
    #kf = KFold(n_splits=5, random_state=1, shuffle=True)
    model = decode_model(encoded_model)
    if verbose:
        print('Model choice: {0}'.format(model))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        
    p = SimpleClassificationPipeline(config=model)
    p.fit(X_train, y_train)
    #scores = sklearn.model_selection.cross_validate(p, X, y, scoring=scoring, cv=5, return_train_score=True)
    #print(scores)
    y_train_pred = p.predict(X_train)
    y_test_pred = p.predict(X_test)
    train_accuracy_score.append(accuracy_score(y_train, y_train_pred))
    test_accuracy_score.append(accuracy_score(y_test, y_test_pred))
    train_log_loss.append(log_loss(y_train, y_train_pred))
        test_log_loss.append(log_loss(y_test, y_test_pred))
    model_performance = np.array([np.mean(train_accuracy_score), np.mean(test_accuracy_score), np.mean(train_log_loss), np.mean(test_log_loss)])
    #print('Model Performance: {o}'.format(model_performance))
    return model_performance
    
    

def get_performance_of_range_encoded_models(data_set_idx, encoded_all_model_hyperparameters, json_model, verbose=False):
    """
    Get models performance from encoded matrix
    """
    X = np.loadtxt('Data_Set/X_' + str(data_set_idx))
    y = np.loadtxt('Data_Set/y_' + str(data_set_idx))
    probas = np.loadtxt('Data_Set/probas_' + str(data_set_idx))
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    models_performance = {}
    #get_performance_of_encoded_model([X, y], encoded_all_model_hyperparameters[0])
    for i in range(len(encoded_all_model_hyperparameters)):
        #model = models[str(i)]
        encoded_model = encoded_all_model_hyperparameters[i]
        model = decode_model(encoded_model)
        if verbose:
            print('Original json model: ', json_model[str(i+1)])
            print('Encoded model: ', encoded_model)
            print('Decoded model:' , model)
            print("==========================================================")
        train_accuracy_score = []
        test_accuracy_score = []
        train_log_loss = []
        test_log_loss = []
        #kf = KFold(n_splits=5, random_state=1)
        time_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        
        p = SimpleClassificationPipeline(config=model)
        p.fit(X_train, y_train)
        #scores = sklearn.model_selection.cross_validate(p, X, y, scoring=scoring, cv=5, return_train_score=True)
        #print(scores)
        y_train_pred = p.predict(X_train)
        y_test_pred = p.predict(X_test)
        train_accuracy_score.append(accuracy_score(y_train, y_train_pred))
        test_accuracy_score.append(accuracy_score(y_test, y_test_pred))
        train_log_loss.append(log_loss(y_train, y_train_pred))
        test_log_loss.append(log_loss(y_test, y_test_pred))
        time_end = time.time()
        duration = time_end - time_start
        models_performance[i] = {"train_accuracy_score": np.mean(train_accuracy_score),
                         "test_accuracy_score": np.mean(test_accuracy_score),
                         "train_log_loss" : np.mean(train_log_loss),
                         "test_log_loss" : np.mean(test_log_loss),
                         "duration" : duration}
           
    performance_json_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
    with open(performance_json_filename, 'w') as fp:
        json.dump(models_performance, fp)
    return models_performance

def save_json_performance_of_encoded_model(i):
    tried_models_hyperparameters_encode_filename = "./log/classifier_log" + str(i) + "/encoded_tried_models_hyperparameters_for_dataset" + str(i) + ".txt"
    encoded_all_model_hyperparameters = np.loadtxt(tried_models_hyperparameters_encode_filename)
    tried_models_filename = "./log/classifier_log" + str(i) + "/tried_models_for_dataset" + str(i) + ".json"
    json_model = {}
    with open(tried_models_filename) as fp:
        json_model = json.load(fp)
    get_performance_of_range_encoded_models(i, encoded_all_model_hyperparameters, json_model)
    
def encode_performance_of_model(data_set_idx):
    performance_json_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
    performance_json = {}
    with open(performance_json_filename) as fp:
        performance_json = json.load(fp)
    performance_metrics_matrix = []
    #reproduce_num_act = min(len(performance_json), reproduce_num)
    for i in range(len(performance_json)):
        performance_single_json = performance_json[str(i+1)]
        performance_vector = []
        for key in performance_keys:
            performance_vector.append(performance_single_json[key])
        performance_metrics_matrix.append(performance_vector)
    print('Encoded model performance: ', np.array(performance_metrics_matrix))
    performance_matrix_filename = "./log/classifier_log" + str(data_set_idx) + "/encode_reproduce_performance_matrix" + str(data_set_idx) + ".txt"
    np.savetxt(performance_matrix_filename, performance_metrics_matrix)

  
if __name__ == "__main__":
    for i in range(int(sys.argv[1]), int(sys.argv[2])):
        save_json_performance_of_encoded_model(i)
        
