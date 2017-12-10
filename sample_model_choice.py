import sys
import numpy as np
<<<<<<< HEAD
from get_performance_of_encoded_model import get_performance_of_encoded_model
from Generate_Data_Set import generate_data_set
=======
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7

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
<<<<<<< HEAD
11 - one_hot_encoding:minimum_fraction [0.0001, 0.5]
=======
11 - one_hot_encoding:minimum_fraction
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7

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

<<<<<<< HEAD
def sample_continuous(y_hat, dimension):
=======
def sample_continues(y_hat, dimension):
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7
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
<<<<<<< HEAD
    '''
    input : predict model choice from rnn 
    return : model_choice
    '''
=======
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7
    model_choice = sample_binary(model_dist, 0)
    
    model_choice = sample_softmax(model_choice, [1,2,3])
    
    model_choice = sample_softmax(model_choice, [4,5,6,7])
    
    model_choice = sample_binary(model_choice, 8)
    
    model_choice = sample_binary(model_choice, 9)
    
    model_choice = sample_binary(model_choice, 10)
    
<<<<<<< HEAD
    model_choice = sample_continuous(model_choice, 11)
    
    model_choice = sample_continuous(model_choice, 12)
    
    model_choice = sample_binary(model_choice, 13)
    
    model_choice = sample_continuous(model_choice, 14)
    
    model_choice = sample_binary(model_choice, 15)
    
    model_choice = sample_continuous(model_choice, 16)
    
    return model_choice

def sample_model_choice_from_prob(model_probability):
    '''
    input : model_probability (one hot dim : probability)
    return : model_chosen
    '''
    r = np.random.rand(17)
    
    model_choice = np.zeros(17)
    assert(0.0<= model_probability[0] <= 1.0) # 0 : balancing:strategy
    model_choice[0] = 1.0 if model_probability[0] >= r[0] else 0.0
    
    assert(model_probability[1] + model_probability[2] + model_probability[3] == 1.0) # 1,2,3 : imputation:strategy
    model_probability[2] = model_probability[1] + model_probability[2]
    model_probability[3] = model_probability[2] + model_probability[3]
    model_choice[1] = 1.0 if 0.0 <= r[1] < model_probability[1] else 0.0
    model_choice[2] = 1.0 if model_probability[1] <= r[1] < model_probability[2] else 0.0 # do not use r[2]
    model_choice[3] = 1.0 if model_probability[2] <= r[1] <= 1.0 else 0.0 # do not use r[3]
    
    assert(model_probability[4] + model_probability[5] + model_probability[6] + model_probability[7] == 1.0) # 4,5,6,7 : rescaling:__choice__
    model_probability[5] = model_probability[4] + model_probability[5]
    model_probability[6] = model_probability[5] + model_probability[6]
    model_probability[7] = model_probability[6] + model_probability[7]
    model_choice[4] = 1.0 if 0.0 <= r[4] < model_probability[4] else 0.0
    model_choice[5] = 1.0 if model_probability[4] <= r[4] < model_probability[5] else 0.0 # do not use r[5]
    model_choice[6] = 1.0 if model_probability[5] <= r[4] < model_probability[6] else 0.0 # do not use r[6]
    model_choice[7] = 1.0 if model_probability[6] <= r[4] <= 1.0 else 0.0 # do not use r[7]
    
    assert(0.0 <= model_probability[8] <= 1.0) # 8 : preprocessor:__choice__ : no_preprocessing/PCA
    model_choice[8] = 1.0 if model_probability[8] >= r[8] else 0.0
    
    assert(0.0 <= model_probability[9] <= 1.0) # 9 : classifier:__choice__ : bernoulli_nb/qda
    model_choice[9] = 1.0 if model_probability[9] >= r[9] else 0.0
    
    assert(0.0 <= model_probability[10] <= 1.0) # 10 : one_hot_encoding:use_minimum_fraction True/False
    model_choice[10] = 1.0 if model_probability[10] >= r[10] else 0.0
    
    # 11: one_hot_encoding:minimum_fraction [0.0001, 0.5]
    model_choice = sample_continuous(model_choice, 11)
    
    # 12: preprocessor:pca:keep_variance  [0.01, 100.0]
    model_choice = sample_continuous(model_choice, 12)
    
    assert(0.0 <= model_probability[13] <= 1.0) # 13 : preprocessor:pca:whiten True/False
    model_choice[13] = 1.0 if model_probability[13] >= r[13] else 0.0
    
    # 14: classifier:bernoulli_nb:alpha    [0.01, 100.0]
    model_choice = sample_continuous(model_choice, 14)
    
    assert(0.0 <= model_probability[15] <= 1.0) # 15 : classifier:bernoulli_nb:fit_prior : True/False
    model_choice[15] = 1.0 if model_probability[15] >= r[15] else 0.0
    
    # 16: classifier:qda:reg_param    [0.0, 1.0]
    model_choice = sample_continuous(model_choice, 16)
=======
    model_choice = sample_continues(model_choice, 11)
    model_choice[11] = 0.01 
    model_choice = sample_continues(model_choice, 12)
    
    model_choice = sample_binary(model_choice, 13)
    
    model_choice = sample_continues(model_choice, 14)
    
    model_choice = sample_binary(model_choice, 15)
    
    model_choice = sample_continues(model_choice, 16)
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7
    
    return model_choice

if __name__ == '__main__':
    model_dist = np.array([-0.30685335,  0.13222617,  0.31084996, -0.09449814,  0.22122033,  0.01977987,
  0.07629474, -0.43703952, -0.15832511, -0.03690121, -0.08069695,  0.56416631,
<<<<<<< HEAD
  0.58806747, -0.2543076,   0.03425501, -0.34980315,  0.34526229])
    #print(sample_model_prediction(model_dist))
    
    model_prob = np.array([0.5, 0.3, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4])
    model_choice = sample_model_choice_from_prob(model_prob)
    print(model_choice)
    N = int(np.random.randint(100) * 10 + 100)
    D = int(np.random.random() * 0.8 * N + 5)
    X, y, probas = generate_data_set(N, D)
    
    print(get_performance_of_encoded_model((X,y), model_choice))
=======
  0.58806747, -0.2543076,   0.03425501, -0.34980315,  0.34526229,])
    print(sample_model_prediction(model_dist))
>>>>>>> 5d99a96bdd372c357d6d4678c07065e394361ac7
    
