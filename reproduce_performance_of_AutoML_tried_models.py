#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
import sys
# from pprint import pprint
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, KFold

performance_keys = ["train_accuracy_score", "test_accuracy_score", "train_log_loss", "test_log_loss"]

def get_models_performance(reproduce_num, data_set_idx):
    '''
    reproduce_num: the number of model choices for the dataset to reproduce
    data_set_idx: generated data set index, will load tried models for the dataset json file
    return: reproduced models performance json file
    '''
    X = np.loadtxt('Data_Set/X_' + str(data_set_idx))
    y = np.loadtxt('Data_Set/y_' + str(data_set_idx))
    probas = np.loadtxt('Data_Set/probas_' + str(data_set_idx))
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/tried_models_for_dataset" + str(data_set_idx) + ".json"
    models_performance = {}
    # duration = get_training_duration(data_set_idx)
    with open(tried_models_filename) as fp:
        models = json.load(fp)
        reproduce_num_act = min(len(models), reproduce_num)
        for i in range(1, reproduce_num_act + 1):
            model = models[str(i)]
            #print(model)
            train_accuracy_score = []
            test_accuracy_score = []
            train_log_loss = []
            test_log_loss = []
            #kf = KFold(n_splits=5, random_state=1, shuffle=True)
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
            #if i in duration:
            #    models_performance[i]["duration"] = duration[i]
    repreduce_performance_json_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
    with open(repreduce_performance_json_filename, 'w') as fp:
        json.dump(models_performance, fp)
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
    '''
    Encode performance json to encoded performance matrix (reproduce_num * 4) 
    data_set_idx: generated data set index, will load reproduced models performance json file of this dataset
    '''
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
        
        
if __name__ == "__main__":
    for i in range(int(sys.argv[1]), int(sys.argv[2])):
        try:
            encode_performance_of_model(i)
        except Exception as err:
            print(" cannot encode performances json file to matrix txt file for data index:" + str(i))
            print("EXCEPTION {0}".format(err))
            pass