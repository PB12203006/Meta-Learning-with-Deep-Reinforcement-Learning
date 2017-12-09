import sys
import numpy as np

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
continueous_range = {11 : [0.0001, 0.5], 12 : [0.5, 0.9999], 14 : [0.01, 100.0], 16 : [0.0, 1.0]}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sample_binary(y_hat, dimension):
    p = sigmoid(y_hat[dimension])
    r = np.random.random()
    y_hat[dimension] = 1.0 if r>p else 0.0
    return y_hat
    
def sample_softmax(y_hat, dimensions):
    r = np.random.random()
    ps = softmax(y_hat[dimensions])
    p = 0
    choice = np.zeros(len(dimensions))
    for i in range(len(ps)):
        p += ps[i]
        if p > r:
            choice[i] = 1.0
            break
    y_hat[dimensions] = choice
    return y_hat

def sample_continues(y_hat, dimension):
    sample_range = continueous_range[dimension]
    EPSILON = 1e-7
    if dimension == 14:
        y_hat[dimension] = 10 ** y_hat[dimension]
    if y_hat[dimension] < sample_range[0]:
        y_hat[dimension] = sample_range[0] + EPSILON
    elif y_hat[dimension] > sample_range[1]:
        y_hat[dimension] = sample_range[1] - EPSILON
    return y_hat
        
def sample_model_prediction(model_dist):
    model_choice = sample_binary(model_dist, 0)
    
    model_choice = sample_softmax(model_choice, [1,2,3])
    
    model_choice = sample_softmax(model_choice, [4,5,6,7])
    
    model_choice = sample_binary(model_choice, 8)
    
    model_choice = sample_binary(model_choice, 9)
    
    model_choice = sample_binary(model_choice, 10)
    
    model_choice = sample_continues(model_choice, 11)
    model_choice[11] = 0.01 
    model_choice = sample_continues(model_choice, 12)
    
    model_choice = sample_binary(model_choice, 13)
    
    model_choice = sample_continues(model_choice, 14)
    
    model_choice = sample_binary(model_choice, 15)
    
    model_choice = sample_continues(model_choice, 16)
    
    return model_choice

if __name__ == '__main__':
    model_dist = np.array([-0.30685335,  0.13222617,  0.31084996, -0.09449814,  0.22122033,  0.01977987,
  0.07629474, -0.43703952, -0.15832511, -0.03690121, -0.08069695,  0.56416631,
  0.58806747, -0.2543076,   0.03425501, -0.34980315,  0.34526229,])
    print(sample_model_prediction(model_dist))
    
