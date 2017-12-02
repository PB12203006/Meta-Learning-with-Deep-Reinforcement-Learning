#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
from get_autosklean_tried_models_hyperparameters import model_hyperparameters_list
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold


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
    """
    Encode model and hyperparameters into number list
    """
    model_hyperparameters_decode_list = reverse_model_hyperparameters_list()
    #print(model_hyperparameters_decode_list)
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
        decoded_model['one_hot_encoding:minimum_fraction'] = encoded_model_hyperparameters[idx]
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

def get_performance_of_encoded_model(data_set_idx, encoded_all_model_hyperparameters, json_model):
    """
    Get model performance from encoded matrix
    """
    X = np.loadtxt('Data_Set/X_' + str(data_set_idx))
    y = np.loadtxt('Data_Set/y_' + str(data_set_idx))
    probas = np.loadtxt('Data_Set/probas_' + str(data_set_idx))
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    models_performance = {}
    for i in range(len(encoded_all_model_hyperparameters)):
        #model = models[str(i)]
        encoded_model = encoded_all_model_hyperparameters[i]
        print(json_model[str(i+1)])
        print(encoded_model)
        model = decode_model(encoded_model)
        print(model)
        print("==========================================================")
        train_accuracy_score = []
        test_accuracy_score = []
        train_log_loss = []
        test_log_loss = []
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        time_start = time.time()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
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
                         "duration" : duration/5}
           
    performance_json_filename = "./log/classifier_log" + str(data_set_idx) + "/rnn_models_performance" + str(data_set_idx) + ".json"
    with open(performance_json_filename, 'w') as fp:
        json.dump(models_performance, fp)
    return models_performance

def test():
    for i in range(1,10):
        tried_models_hyperparameters_encode_filename = "./log/classifier_log" + str(i) + "/encoded_tried_models_hyperparameters_for_dataset" + str(i) + ".txt"
        encoded_all_model_hyperparameters = np.loadtxt(tried_models_hyperparameters_encode_filename)
        tried_models_filename = "./log/classifier_log" + str(i) + "/tried_models_for_dataset" + str(i) + ".json"
        json_model = {}
        with open(tried_models_filename) as fp:
            json_model = json.load(fp)
        get_performance_of_encoded_model(i, encoded_all_model_hyperparameters, json_model)
        
        
if __name__ == "__main__":
    test()