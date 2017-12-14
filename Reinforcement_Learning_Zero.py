#This is a very high level reinforcement framework
#It is mainly for reference purpose
import tensorflow as tf
import numpy as np
from Generate_Data_Set import generate_data_set
from sample_model_choice import sample_model_choice_from_prob
from get_performance_of_encoded_model import get_performance_of_encoded_model
from get_metadata import get_metadata
import multiprocessing
import multiprocessing
from generate_random_model import generate
from subprocess import call
import time
#how myopic is the model (0 myopic, 1 count all reward in the future)
GAMMA = 0.8
#The number of steps tried by the neural network
num_of_steps = 10
#the epsilon for the epsilon greedy exploration
EPSILON = 0.2

#METADATA_DIM: the dimension of meta data
#ACTION_DIM: the dimension of action, which is a probability distribution
#CHOICE_DIM: a one-hot encoding of the choice, which is a deterministic choice
#PERFORMANCE_DIM: the dimension of the returned performance
#DENSE_UNITS: the number of dense units for each of the inner units
#!!!!!! Note that in an action, all entries of probability corresponding to one choice adds up to one
#(probabilities instead of logits)
def create_Q_network(LSTM_HIDDEN, METADATA_DIM, ACTION_DIM, PERFORMANCE_DIM, CHOICE_DIM, DENSE_UNITS):
    from keras.models import Model
    from keras import models
    from keras.layers import Input, Dense, LSTM, Concatenate
    from keras import backend as K
    #All the inputs to the LSTM
    meta_data_input_layer = Input(shape=(None, METADATA_DIM))
    performance_input_layer = Input(shape=(None, PERFORMANCE_DIM))
    last_choice_input_layer = Input(shape=(None, CHOICE_DIM))
    
    #Concatenate the input
    input_agg = Concatenate()([meta_data_input_layer, performance_input_layer, last_choice_input_layer])
    
    #Input followed by 1 dense layer before feeding into the LSTM
    input_1 = Dense(DENSE_UNITS)(input_agg)
    
    #LSTM defined here, game_state corresponding to the state (s in Q(s, a))
    lstm_layer = LSTM(LSTM_HIDDEN, return_sequences=True)
    game_state= lstm_layer(input_1)

    #the action input a
    action_input = Input(shape=(None, ACTION_DIM))
    
    #Aggregating a and s, since we are learning Q(s, a)
    action_state_agg = Concatenate()([game_state, action_input])
    sa_dense1 = Dense(DENSE_UNITS)
    sa_dense_out1 = sa_dense1(action_state_agg)
    sa_dense2 = Dense(1, activation='relu')
    value_action_at_state = sa_dense2(sa_dense_out1)

    #Define the model that predicts Q(s,a)
    qsa_model = Model([meta_data_input_layer, performance_input_layer, last_choice_input_layer, action_input], value_action_at_state)
    qsa_model.summary()
    qsa_model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])
    qsa_model.save('model_0')
    return


#The training goes here
def train_network(iteration_idx, meta_data, model_choices, model_performances, actions, values):
    from keras.models import Model
    from keras import models
    from keras.layers import Input, Dense, LSTM, Concatenate
    from keras import backend as K
    qsa_model = models.load_model('model_%d' % iteration_idx)
    values = np.expand_dims(values, axis=2)
    qsa_model.fit([meta_data, model_performances, model_choices, actions], values, batch_size=32, epochs=10)
    qsa_model.save('model_' + str(iteration_idx + 1))
    return

def evaluate(iteration_idx,  num_threads, sample_count):
    num_threads = num_threads
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    meta = []
    performance = []
    choices = []
    actions = []
    q = []
    result_acc = []
    assert(sample_count % num_threads == 0)
    for _ in range(int(sample_count / num_threads)):
        for thread_idx in range(num_threads):
            p = multiprocessing.Process(target=evaluate_on_data_set, args=(thread_idx, return_dict,
                                                                           iteration_idx, 38, 17, 4, 17, 0.1,))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for _ in range(num_threads):
            if return_dict.get(_) is None:
                continue
            meta.append(return_dict[_][0][0])
            performance.append(return_dict[_][1][0])
            choices.append(return_dict[_][2][0])
            actions.append(return_dict[_][3][0])
            q.append(return_dict[_][4])
            result_acc.append(return_dict[_][5])
    for _ in range(len(result_acc)):
        for t in range(1, 10):
            result_acc[_][t] = result_acc[_][t - 1] if result_acc[_][t - 1] > result_acc[_][t] else result_acc[_][t]
    averaged_curve = np.mean(result_acc, axis=0)
    np.save('iteration ' + str(iteration_idx) + '_performance', averaged_curve)
    return np.array(meta), np.array(performance), np.array(choices), np.array(actions), np.array(q)

#The evaluation phase goes to here
#It is generating the data: meta_data, model_choices, model_performances, actions, values
#for the training
#It is better implemented as multi-thread process
def evaluate_on_data_set(thread_idx, return_dict, iteration_idx, METADATA_DIM, ACTION_DIM, PERFORMANCE_DIM, CHOICE_DIM,
                         EPSILON):
    from keras import models
    np.random.seed(seed=int(thread_idx * int(time.time() / 100) ))
    num_of_steps = 10
    #Generate Data set
    #Get metastatistics
    call(['rm', '-r', 'log_rnn' + str(thread_idx)])
    N = int(np.random.randint(100) * 10 + 100)
    D = int(np.random.random() * 0.8 * N + 5)
    X, y, probas = generate_data_set(N, D)
    dataset = (X, y)
    meta_data = None
    try:
        meta_data = get_metadata((X, y), thread_idx)
        assert(meta_data.shape==(38,))
    except:
        meta_data = np.zeros((38,))
    call(['rm', '-r', 'log_rnn' + str(thread_idx)])
    last_model_choice = np.zeros((17,))
    last_model_performance = np.zeros((4,))
    qsa_model = models.load_model('model_' + str( iteration_idx))
    #Maintain a history of the evaluation
    reward_history = []
    action_history = []
    performance_history = []
    model_choice_history = []
    model_choice_history.append(last_model_choice)
    performance_history.append(last_model_performance)
    result_acc = []
    max_acc_now = 0
    for step in range(num_of_steps):
        chosen_action = None
        best_value = -1
        next_state = None
        meta_data_history_ = np.tile(meta_data, (step + 1 , 1)).reshape((1, step + 1, 38))
        performance_history_ = np.array([performance_history])
        model_choice_history_ = np.array([model_choice_history])
        if np.random.random() > EPSILON:
            '''
            for _ in range(100):
                a_h = action_history[:]
                a = generate()
                a_h.append(a)
                action_history_ = np.array([a_h])
                assert(meta_data_history_.shape == (1, step + 1, 38))
                assert(model_choice_history_.shape == (1, step + 1, 17))
                assert(action_history_.shape == (1, step + 1, 17))
                assert(performance_history_.shape == (1, step + 1, 4))
                value = qsa_model.predict([meta_data_history_, performance_history_,
                                           model_choice_history_, action_history_])[0][-1]
                if value > best_value:
                    best_value = value
                    chosen_action = a
            '''
            proposal = []
            for _ in range(100):
                proposal.append(generate())
            proposal = np.array(proposal)
            proposal = np.expand_dims(proposal, axis=1)
            if step == 0:
                action_history_ = proposal
            else:
                a_h = np.tile(np.array(action_history[:]), (100,1, 1))
                action_history_ = np.concatenate((a_h, proposal), axis=1)
            print('predicting values')
            print(action_history_.shape)
            print(performance_history_.shape)
            value = qsa_model.predict([np.tile(meta_data_history_, (100,1,1)), np.tile(performance_history_,(100,1,1)),
                                          np.tile(model_choice_history_, (100,1,1)), np.tile(action_history_, (100,1,1))])
            value = value[:,-1,0]
            print('prediction done')
            chosen_action = proposal[np.argmax(value)][0]
        else:
            chosen_action = generate()
        model_chosen = sample_model_choice_from_prob(chosen_action)
        action_history.append(chosen_action) 
        #Sample a model choice based on a
        model_choice_history.append(model_chosen)
        #get Performance
        performance_history.append(get_performance_of_encoded_model((X, y), model_chosen))
        #calculate reward
        accuracy = performance_history[-1][1]
        result_acc.append(accuracy)
        reward_history.append(max(0, accuracy - max_acc_now))
    qsa_model = None
    q_history = [0] * num_of_steps
    q_history[-1] = reward_history[-1]
    for _ in range(2, num_of_steps + 1):
        q_history[num_of_steps - _] = reward_history[num_of_steps - _] + GAMMA * q_history[num_of_steps + 1 - _]
    return_dict[thread_idx] = (np.tile(meta_data, (num_of_steps, 1)).reshape((1, num_of_steps, 38)), np.array([performance_history[:num_of_steps]]), np.array([model_choice_history[:num_of_steps]]),
                        np.array([action_history]), q_history, result_acc)
    return

def reinforcement_learning(num_iterations):
    process = multiprocessing.Process(target=create_Q_network, args=(16, 38, 17, 4, 17, 10))
    process.start()
    process.join()
    #qsa_model.save('model_%d' % 0)
    for iteration_idx in range(num_iterations):
        m, p, c, a, q = evaluate(iteration_idx, 20, 100)
        process = multiprocessing.Process(target=train_network, args=(iteration_idx, m, c, p, a, q))
        process.start()
        process.join()
        #train_network(iteration_idx, m, c, p, a, q)

reinforcement_learning(100)

