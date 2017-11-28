import tensorflow as tf
import keras
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
from Load_and_transform import *

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
    return tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.square(params_true - params_pred), axis=2), weights)) / len(dimensions)

#Last Layer comes in w \out any activation
def customized_loss(y_true, y_pred):
    balance_loss_weight = 0.01
    rescale_loss_weight = 0.01
    model_choice_loss_weight = 1
    bernoulli_nb_loss_weight = 0.5
    svm_loss_weight = 0.5
    qda_loss_weight = 0.5
    clf_bernoulli_nb_used = tf.gather(y_true, 1, axis=2)
    clf_svm_used = tf.gather(y_true, 2, axis=2)
    clf_qda_used = tf.gather(y_true, 3, axis=2)
    clf_svm_poly_used = tf.gather(y_true, 18, axis=2)
    
    #Dimension 0 balance
    balance_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 0, weights=balance_loss_weight)
    
    #Dimesnsion 1, 2, 3 classifier choice
    model_choice_loss = cross_entropy_with_logits(y_true, y_pred, [1,2,3], weights=model_choice_loss_weight)
    
    #Dimension 4,5,6,7,8 irrelevant
    
    #Dimension 9,10,11,12
    rescale_loss = cross_entropy_with_logits(y_true, y_pred, [9,10,11,12], weights=rescale_loss_weight)
    
    #Dimension 13
    bernoulli_nb_params_loss = square_params_loss(y_true, y_pred, [13], weights=clf_bernoulli_nb_used * bernoulli_nb_loss_weight)
    #Dimension 14
    bernoulli_nb_fit_prior_loss = sigmoid_binary_loss_with_logits(y_true, y_pred, 14, weights=clf_bernoulli_nb_used * bernoulli_nb_loss_weight)
    bernoulli_nb_loss = bernoulli_nb_fit_prior_loss + bernoulli_nb_params_loss
    
    #Dimension 15,16, 22,25
    svm_param_loss = square_params_loss(y_true, y_pred, [15, 16, 22, 25], weights=clf_svm_used * svm_loss_weight)
    #Dimension 17, 18,19
    svm_kernel_loss = cross_entropy_with_logits(y_true, y_pred, [17,18,19],weights=clf_svm_used * svm_loss_weight)
    #Dimension 26
    svm_degree_loss = square_params_loss(y_true, y_pred, [26], weights=clf_svm_used * svm_loss_weight * clf_svm_poly_used)
    svm_loss = svm_kernel_loss + svm_param_loss + svm_degree_loss
    
    #Dimension 20, 21, 24 irrelevant; in runs, always shrink
    
    #Dimension 23
    qda_loss = square_params_loss(y_true, y_pred, [23], weights = clf_qda_used * qda_loss_weight)
    
    #Aggregating the losses
    loss = balance_loss_weight + model_choice_loss + bernoulli_nb_loss + svm_loss + qda_loss
    
    return loss

def model_choice_loss_(y_true, y_pred):
    model_choice_loss = cross_entropy_with_logits(y_true, y_pred, [1,2,3], weights=1)
    return model_choice_loss

def model_choice_accuracy_(y_true, y_pred):
    true_one_hot = tf.gather(y_true, [1,2,3], axis=2)
    true_labels = tf.argmax(true_one_hot, axis=2)
    pred_one_hot = tf.gather(y_true, [1,2,3], axis=2)
    pred_labels = tf.argmax(pred_one_hot, axis=2)
    accuracy, o = tf.metrics.accuracy(true_labels, pred_labels)
    return accuracy

def kernel_choice_accuracy_(y_true, y_pred):
    clf_svm_used = tf.gather(y_true, 2, axis=2)
    true_one_hot = tf.gather(y_true, [17,18,19], axis=2)
    true_labels = tf.argmax(true_one_hot, axis=2)
    pred_one_hot = tf.gather(y_pred, [17,18,19], axis=2)
    pred_labels = tf.argmax(pred_one_hot, axis=2)
    accuracy, o = tf.metrics.accuracy(true_labels, pred_labels, weights=clf_svm_used)
    return accuracy

def svm_param_loss_(y_true, y_pred):
    clf_svm_used = tf.gather(y_true, 2, axis=2)
    svm_param_loss = square_params_loss(y_true, y_pred, [15, 16, 22, 25], weights=clf_svm_used)
    return svm_param_loss

def qda_param_loss_(y_true, y_pred):
    clf_qda_used = tf.gather(y_true, 3, axis=2)
    qda_loss = square_params_loss(y_true, y_pred, [23], weights = clf_qda_used)
    return qda_loss

def bernoulli_nb_param_loss_(y_true, y_pred):
    clf_bernoulli_nb_used = tf.gather(y_true, 1, axis=2)
    bernoulli_nb_params_loss = square_params_loss(y_true, y_pred, [13], weights=clf_bernoulli_nb_used)
    return bernoulli_nb_params_loss

max_length = 20
meta_statistics_input_layer = Input(shape=(max_length, 10))
y_last_input_layer = Input(shape=(max_length, 27))
feedback_input_layer = Input(shape=(max_length, 4))
meta_statistics_in_layer = Dense(10, activation='sigmoid')(meta_statistics_input_layer)
y_last_in_layer = Dense(10, activation='sigmoid')(y_last_input_layer)
feedback_in_layer = Dense(10, activation='sigmoid')(feedback_input_layer)
input_agg = Add()([meta_statistics_in_layer, y_last_in_layer, feedback_in_layer])
input_agg = Dense(10, activation='sigmoid')(input_agg)
predictions = LSTM(10, recurrent_dropout=0.1, return_sequences=True)(input_agg)
predictions = Dense(10, activation='sigmoid')(predictions)
predictions = Dense(27, activation=None)(predictions)
model = Model(inputs=[meta_statistics_input_layer, y_last_input_layer, feedback_input_layer], outputs=predictions)
model.compile(loss=customized_loss, optimizer='Adam', metrics=[model_choice_loss_, model_choice_accuracy_, svm_param_loss_, kernel_choice_accuracy_, qda_param_loss_, bernoulli_nb_param_loss_])
model.summary()

meta_data_dir = 'Generated_data/datafeature.npy'
meta_data = load_meta_data_from_dir(meta_data_dir)
performance =
