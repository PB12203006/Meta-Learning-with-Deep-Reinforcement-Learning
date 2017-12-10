#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
import sys
# from pprint import pprint
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

performance_keys = ["train_accuracy_score", "test_accuracy_score", "train_log_loss", "test_log_loss"]

def random_model(nb=0.5):
    model = dict()
    
    r = np.random.random()
    if r >0.5:
        model["balancing:strategy"] = "none"
    else:
        model["balancing:strategy"] = "weighting"
        
    r = np.random.random()
    if r<0.33:
        model["imputation:strategy"] = "mean"
    elif r<0.66:
        model["imputation:strategy"] = "median"
    else:
        model["imputation:strategy"] = "most_frequent"
    
    r = np.random.random()
    if r < 0.25:
        model["rescaling:__choice__"] = "none"
    elif r<0.5:
        model["rescaling:__choice__"] = "minmax"
    elif r < 0.75:
        model["rescaling:__choice__"] = "normalize"
    else:
        model["rescaling:__choice__"] = "standardize"
    
    r = np.random.random()
    if r > 0.5:
        model['preprocessor:__choice__'] = 'no_preprocessing'
    else:
        model['preprocessor:__choice__'] = 'pca'
        r1 = np.random.random()
        model['preprocessor:pca:keep_variance'] = 0.5 + r1*(0.9999-0.5)
        r2 = np.random.random()
        if r2>0.5:
            model['preprocessor:pca:whiten'] = 'True'
        else:
            model['preprocessor:pca:whiten'] = 'False'
        
    
    r = np.random.random()
    if r < nb:
        model['classifier:__choice__'] = 'bernoulli_nb'
        r1 = np.random.random()
        model["classifier:bernoulli_nb:alpha"] = 0.01 + r1*(100-0.01)
        r2 = np.random.random()
        if r2>0.5:
            model["classifier:bernoulli_nb:fit_prior"] = "True"
        else:
            model["classifier:bernoulli_nb:fit_prior"] = "False"
    else:
        model['classifier:__choice__'] = 'qda'
        r = np.random.random()
        model["classifier:qda:reg_param"] = r
        
    r = np.random.random()
    if r > 0.5:
        model["one_hot_encoding:use_minimum_fraction"] = "True"
        r1 = np.random.random()
        model["one_hot_encoding:minimum_fraction"] = 0.0001 + r1*(0.5-0.0001)
    else:
        model["one_hot_encoding:use_minimum_fraction"] = "False"
    return model
    

def get_models_performance_by_data(input):
    #X = np.loadtxt('Data_Set/X_' + str(data_set_idx))
    #y = np.loadtxt('Data_Set/y_' + str(data_set_idx))
    X = input[0]
    y = input[1]
    probas = np.loadtxt('Data_Set/probas_' + str(data_set_idx))
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    #tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/tried_models_for_dataset" + str(data_set_idx) + ".json"
    #models_performance = {}
    # duration = get_training_duration(data_set_idx)
    #with open(tried_models_filename) as fp:
    if 1:
        #models = json.load(fp)
        #reproduce_num_act = min(len(models), reproduce_num)
        #for i in range(1, reproduce_num_act + 1):
        if 1:
            #model = models[str(i)]
            #print(model)
            model = random_model()
            
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
            models_performance = {"train_accuracy_score": np.mean(train_accuracy_score),
                             "test_accuracy_score": np.mean(test_accuracy_score),
                             "train_log_loss" : np.mean(train_log_loss),
                             "test_log_loss" : np.mean(test_log_loss),
                             "duration" : duration/5}
            #if i in duration:
            #    models_performance[i]["duration"] = duration[i]
    #repreduce_performance_json_filename = tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
    #with open(repreduce_performance_json_filename, 'w') as fp:
     #   json.dump(models_performance, fp)
    return models_performance
        

def get_models_performance(data_set_idx,nb=0.5):
    X = np.loadtxt('Data_Set/X_' + str(data_set_idx))
    y = np.loadtxt('Data_Set/y_' + str(data_set_idx))
    probas = np.loadtxt('Data_Set/probas_' + str(data_set_idx))
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    #tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/tried_models_for_dataset" + str(data_set_idx) + ".json"
    #models_performance = {}
    # duration = get_training_duration(data_set_idx)
    #with open(tried_models_filename) as fp:
    if 1:
        #models = json.load(fp)
        #reproduce_num_act = min(len(models), reproduce_num)
        #for i in range(1, reproduce_num_act + 1):
        if 1:
            #model = models[str(i)]
            #print(model)
            model = random_model(nb=nb)
            
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
            models_performance = {"train_accuracy_score": np.mean(train_accuracy_score),
                             "test_accuracy_score": np.mean(test_accuracy_score),
                             "train_log_loss" : np.mean(train_log_loss),
                             "test_log_loss" : np.mean(test_log_loss),
                             "duration" : duration/5}
            #if i in duration:
            #    models_performance[i]["duration"] = duration[i]
    #repreduce_performance_json_filename = tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
    #with open(repreduce_performance_json_filename, 'w') as fp:
     #   json.dump(models_performance, fp)
    return models_performance
            
def get_training_duration(data_set_idx):
    model_hyperparameters_filename = "./log/classifier_log" + str(data_set_idx) + "/AutoML(1):simulated" + str(data_set_idx) + ".log"
    print("Geting training duration from " + model_hyperparameters_filename)
    duration = {}
    with open(model_hyperparameters_filename) as f:
        for line in f:
            if "duration" in line:
                info = json.loads(line.split("additional:")[1].replace('\n', '').replace('\'','\"'))
                duration[info['num_run']] = info['duration']
    return duration

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
    #print(np.array(performance_metrics_matrix))
    performance_matrix_filename = "./log/classifier_log" + str(data_set_idx) + "/encode_reproduce_performance_matrix" + str(data_set_idx) + ".txt"
    np.savetxt(performance_matrix_filename, performance_metrics_matrix)

def test():
    for i in range(1):
        get_models_performance(20, i)
        
'''        
if __name__ == "__main__":
    performance = []
    for i in range(3494,3495):
        datap = []
        for j in range(10):
            datap.append(get_models_performance(i)['test_accuracy_score'])
        performance.append(datap)
    print(performance)
'''