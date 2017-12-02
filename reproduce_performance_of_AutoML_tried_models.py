#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
# from pprint import pprint
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

def get_models_performance(reproduce_num, data_set_idx):
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
        for i in range(1, len(models) + 1):
            model = models[str(i)]
            #print(model)
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
            #if i in duration:
            #    models_performance[i]["duration"] = duration[i]
    repreduce_performance_json_filename = tried_models_filename = "./log/classifier_log" + str(data_set_idx) + "/reproduce_models_performance" + str(data_set_idx) + ".json"
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

def test():
    for i in range(1):
        get_models_performance(20, i)
        
        
#if __name__ == "__main__":
#    test()