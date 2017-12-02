#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from Generate_Data_Set import generate_data_set
import autosklearn.classification
import numpy as np
from get_autosklean_tried_models_hyperparameters import get_encoded_models_for_data_set
from reproduce_performance_of_AutoML_tried_models import get_models_performance
from get_metadata_from_log import get_metadata_from_log

data_set_count = 1
for data_set_index in range(data_set_count):
        # genetate data set
        N = int(np.random.randint(100) * 10 + 100)
        D = int(np.random.random() * 0.8 * N)
        X, y, probas = generate_data_set(N, D)
        np.savetxt('Data_Set/X_' + str(data_set_index), X)
        np.savetxt('Data_Set/y_' + str(data_set_index), y)
        np.savetxt('Data_Set/probas_' + str(data_set_index), probas)
        feature_types = (['numerical'] * D)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
        
        # train and log the process of auto-sklean
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120, per_run_time_limit=30,
            tmp_folder='./log/classifier_log' + str(data_set_index),
            output_folder='./log/classifier_out' + str(data_set_index),
            include_estimators=['qda', 'bernoulli_nb'],
            exclude_estimators = None,
            include_preprocessors=["no_preprocessing", "pca"], exclude_preprocessors=None,
            ml_memory_limit= 10240,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            ensemble_size=30, initial_configurations_via_metalearning=25)
        automl.fit(X_train, y_train, dataset_name='simulated'+str(data_set_index),)
        print(automl.show_models())
        
        # get metadata from log and save json
        get_metadata_from_log(data_set_index)
        # get and save model json and encoded model vectors
        tried_models_hyperparameters_encode, tried_models = get_encoded_models_for_data_set(data_set_index)
        # get and save reproduce model performance
        get_models_performance(10, data_set_index)
        
        # delete log file
        log_file_path = "./log/classifier_log" + str(data_set_index) + "/AutoML(1):simulated" + str(data_set_index) + ".log"
        try:
            os.remove(log_file_path)
        except:
            print("Remove file failed. File name: " + log_file_path)
            pass
