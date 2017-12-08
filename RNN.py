#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D, Add
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.optimizers import Adadelta
from keras import optimizers
from keras.models import Model, load_model
from keras import metrics
from integrate_rnn_input_matrix import integrate_encoded_data_for_datasets
#from Load_and_transform_data import *

#Dimension 0: 0 if none, 1 if weighting
#Dimension 1: bernoulli_nb
#Dimension 2: libsvm_svc
#Dimension 3: qda
#Dimension 4: imputation mean
#Dimension 5: imputation median
#Dimension 6: imputation mode
#Dimension 7: one_hot_use_minimum_fraction
#Dimension 8: preprocessor (always 0 here)
#Dimension 9: min-max
#Dimension 10: none
#Dimension 11: normalize
#Dimension 12: standardize
#Dimension 13: bernoulli, alpha
#Dimension 14: bernoulli, fit prior
#Dimension 15: svm, C
#Dimension 16: svm, gamma
#Dimension 17: svm, rbf
#Dimension 18: svm, poly
#Dimension 19: svm, sigmoid
#Dimension 20: svm, max_iter
#Dimension 21: svm, shrinking. 0 if True
#Dimension 22: svm, tol
#Dimension 23: qda, reg
#Dimension 24: one-hot mean fraction
#Dimension 25: svm, coef0
#Dimension 26: svm, degree
"""
METADATA:

0 - 'ClassEntropy',
1 - 'SymbolsSum',
2 - 'SymbolsSTD',
3 - 'SymbolsMean',
4 - 'SymbolsMax',
5 - 'SymbolsMin',
6 - 'ClassProbabilitySTD',
7 - 'ClassProbabilityMean',
8 - 'ClassProbabilityMax',
9 - 'ClassProbabilityMin',
10 - 'InverseDatasetRatio',
11 - 'DatasetRatio',
12 - 'RatioNominalToNumerical',
13 - 'RatioNumericalToNominal',
14 - 'NumberOfCategoricalFeatures',
15 - 'NumberOfNumericFeatures',
16 - 'NumberOfMissingValues',
17 - 'NumberOfFeaturesWithMissingValues',
18 - 'NumberOfInstancesWithMissingValues',
19 - 'NumberOfFeatures',
20 - 'NumberOfClasses',
21 - 'NumberOfInstances',
22 - 'LogInverseDatasetRatio',
23 - 'LogDatasetRatio',
24 - 'PercentageOfMissingValues',
25 - 'PercentageOfFeaturesWithMissingValues',
26 - 'PercentageOfInstancesWithMissingValues',
27 - 'LogNumberOfFeatures',
28 - 'LogNumberOfInstances',
29 - 'LandmarkRandomNodeLearner',
30 - 'SkewnessSTD',
31 - 'SkewnessMean',
32 - 'SkewnessMax',
33 - 'SkewnessMin',
34 - 'KurtosisSTD',
35 - 'KurtosisMean',
36 - 'KurtosisMax',
37 - 'KurtosisMin'

"""

"""
MODEL CHOICE

0 - balancing:strategy : none/weighting

1 - imputation:strategy : not mean / mean
2 - imputation:strategy : not median / median
3 - imputation:strategy : not most_frequent / most_frequent

4 - rescaling:__choice__ : not none / none
5 - rescaling:__choice__ : not minmax/minmax
6 - rescaling:__choice__ : not normalize/normalize
7 - rescaling:__choice__ : not standardize/standardize

8 - preprocessor:__choice__ : no_preprocessing/PCA
9 - classifier:__choice__ : bernoulli_nb/qda

10 - one_hot_encoding:use_minimum_fraction: True/False
11 - one_hot_encoding:minimum_fraction

12 - preprocessor:pca:keep_variance   [0.5, 0.9999]
13 - preprocessor:pca:whiten : True/False

14 - classifier:bernoulli_nb:alpha    [0.01, 100.0]
15 - classifier:bernoulli_nb:fit_prior : True/False
16 - classifier:qda:reg_param    [0.0, 1.0]

"""

"""
MODEL PERFORMANCE

0 - train_accuracy_score
1 - test_accuracy_score
2 - train_log_loss
3 - test_log_loss

"""

def cross_entropy_with_logits(y_true, y_pred, dimensions, weights):
    choice_true = tf.gather(y_true, dimensions, axis=2)
    choice_pred_logits = tf.gather(y_pred, dimensions, axis=2)
    return tf.losses.softmax_cross_entropy(choice_true, choice_pred_logits, weights=weights)

def sigmoid_binary_loss_with_logits(y_true, y_pred, dimension, weights):
    label_true = tf.gather(y_true, dimension, axis=2)
    label_pred_logits = tf.gather(y_pred, dimension, axis=2)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_true, logits=label_pred_logits) * weights)

def square_params_loss(y_true, y_pred, dimensions, weights):
    params_true = tf.gather(y_true, dimensions, axis=2)
    params_pred = tf.gather(y_pred, dimensions, axis=2)
    
    if 14 in dimensions:
        print(dimensions)
        return tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.square(tf.log(params_true)/tf.log(10.0) - tf.log(params_pred)/tf.log(10.0)), axis=2), weights)) / len(dimensions)
    else:
        return tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.square(params_true - params_pred), axis=2), weights)) / len(dimensions)

#Last Layer comes in w \out any activation
def customized_loss(y_true, y_pred):
    balance_loss_weight = 0.01
    rescale_loss_weight = 0.01
    imputation_choice_loss_weight = 0.05
    preprocessor_choice_loss_weight = 1
    model_choice_loss_weight = 1
    pca_loss_weight = 0.5
    bernoulli_nb_loss_weight = 0.5
    qda_loss_weight = 0.5
    clf_pca_used = tf.gather(y_true, 8, axis=2)
    clf_bernoulli_nb_used = 1 - tf.gather(y_true, 9, axis=2)
    clf_qda_used = tf.gather(y_true, 9, axis=2)
    #clf_svm_poly_used = tf.gather(y_true, 18, axis=2)

    #Dimension 0 balance
    balance_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 0, weights=balance_loss_weight)

    #Dimesnsion 1, 2, 3 imputation choice
    imputation_choice_loss = cross_entropy_with_logits(y_true, y_pred, [1,2,3], weights=imputation_choice_loss_weight)

    #Dimension 4,5,6,7 rescale choice
    rescale_loss = cross_entropy_with_logits(y_true, y_pred, [4,5,6,7], weights=rescale_loss_weight)

    #Dimension 8 preprocessor choice
    preprocessor_choice_loss = cross_entropy_with_logits(y_true, y_pred, [8], weights=preprocessor_choice_loss_weight)

    #Dimension 9 classifier choice
    model_choice_loss = cross_entropy_with_logits(y_true, y_pred, [9], weights=model_choice_loss_weight)

    #Dimension 10, 11 one hot - irrelevant

    #Dimension 12 preprocessor pca
    pca_variance_loss = square_params_loss(y_true, y_pred, [12], weights = clf_pca_used * pca_loss_weight)
    pca_whiten_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 13, weights=clf_pca_used * pca_loss_weight)
    pca_loss = pca_variance_loss + pca_whiten_loss

    #Dimension 14
    bernoulli_nb_params_loss = square_params_loss(y_true, y_pred, [14], weights=clf_bernoulli_nb_used*bernoulli_nb_loss_weight)
    #Dimension 15
    bernoulli_nb_fit_prior_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 15, weights=clf_bernoulli_nb_used * bernoulli_nb_loss_weight)
    bernoulli_nb_loss = bernoulli_nb_fit_prior_loss + bernoulli_nb_params_loss

    #Dimension 16
    qda_loss = square_params_loss(y_true, y_pred, [16], weights = clf_qda_used * qda_loss_weight)

    #Aggregating the losses
    loss = balance_loss + imputation_choice_loss + rescale_loss + preprocessor_choice_loss + model_choice_loss + pca_loss + bernoulli_nb_loss + qda_loss

    return loss

def preprocessor_choice_loss_(y_true, y_pred):
    preprocessor_choice_loss = cross_entropy_with_logits(y_true, y_pred, [8], weights=1)
    return preprocessor_choice_loss

def model_choice_loss_(y_true, y_pred):
    model_choice_loss = cross_entropy_with_logits(y_true, y_pred, [9], weights=1)
    return model_choice_loss

def preprocessor_choice_accuracy_(y_true, y_pred):
    true_labels = tf.gather(y_true, 8, axis=2)
    pred_labels = tf.gather(y_pred, 8, axis=2)
    accuracy, o = tf.metrics.accuracy(true_labels, pred_labels)
    return accuracy

def model_choice_accuracy_(y_true, y_pred):
    true_labels = tf.gather(y_true, 9, axis=2)
    pred_labels = tf.gather(y_pred, 9, axis=2)
    accuracy, o = tf.metrics.accuracy(true_labels, pred_labels)
    return accuracy

def pca_param_loss_(y_true, y_pred):
    clf_pca_used = tf.gather(y_true, 8, axis=2)
    pca_variance_loss = square_params_loss(y_true, y_pred, [12], weights = clf_pca_used )
    pca_whiten_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 13, weights=clf_pca_used)
    pca_loss = pca_variance_loss + pca_whiten_loss
    #pca_loss = square_params_loss(y_true, y_pred, [12], weights = clf_pca_used)
    return pca_loss

def qda_param_loss_(y_true, y_pred):
    clf_qda_used = tf.gather(y_true, 9, axis=2)
    qda_loss = square_params_loss(y_true, y_pred, [16], weights = clf_qda_used)
    return qda_loss

def bernoulli_nb_param_loss_(y_true, y_pred):
    clf_bernoulli_nb_used = 1 - tf.gather(y_true, 9, axis=2) # 0 - bernoulli_nb ?
    bernoulli_nb_params_loss = square_params_loss(y_true, y_pred, [14], weights=clf_bernoulli_nb_used)
    bernoulli_nb_fit_prior_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 15, weights=clf_bernoulli_nb_used)
    bernoulli_nb_loss = bernoulli_nb_fit_prior_loss + bernoulli_nb_params_loss
    #bernoulli_nb_params_loss = square_params_loss(y_true, y_pred, [14], weights=clf_bernoulli_nb_used)
    return bernoulli_nb_loss

max_length = 20 # sequence length
meta_statistics_input_layer = Input(shape=(None, 38)) # meta_statistics_input num
x_last_input_layer = Input(shape=(None, 17))
feedback_input_layer = Input(shape=(None, 4)) # 4: training testing loss accuracy
meta_statistics_in_layer = Dense(10, activation='sigmoid')(meta_statistics_input_layer)
x_last_in_layer = Dense(10, activation='sigmoid')(x_last_input_layer)
feedback_in_layer = Dense(10, activation='sigmoid')(feedback_input_layer)
input_agg = Add()([meta_statistics_in_layer, x_last_in_layer, feedback_in_layer])
input_agg = Dense(10, activation='sigmoid')(input_agg)

predictions = LSTM(10, recurrent_dropout=0.1, return_sequences=True)(input_agg) #LSTM: layer
# LSTM INPUT ONE STEP GET ONE OUT, not input all
predictions = Dense(10, activation='sigmoid')(predictions)
predictions = Dense(17, activation=None)(predictions)
model = Model(inputs=[meta_statistics_input_layer, x_last_input_layer, feedback_input_layer], outputs=predictions)
# add pca loss, continueous: square_params_loss
model.compile(loss=customized_loss, optimizer='Adam', metrics=[preprocessor_choice_loss_, model_choice_loss_, pca_param_loss_, qda_param_loss_, bernoulli_nb_param_loss_]) #preprocessor_choice_accuracy_, model_choice_accuracy_, 
model.summary()

generate_range = range(int(sys.argv[1]), int(sys.argv[2]))
metafeatures_matrix, input_model_choice_matrix, predict_model_choice_matrix, input_performance_matrix = integrate_encoded_data_for_datasets(generate_range)
model.fit([metafeatures_matrix, input_model_choice_matrix, input_performance_matrix], predict_model_choice_matrix, epochs=10, batch_size=max_length)