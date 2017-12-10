import matplotlib.pyplot as plt
import numpy as np

#input [(str,[]),...]
def plot(title,input):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Model Attempted')
    ax.set_ylabel('Test Accuracy')
    for line in input:
        ax.plot(line[1],label=line[0])
    ax.legend(loc = 4)
    fig.savefig(title+'.png')
    #plt.show()

    
def maxresult(array):
    for i in range(1,array.shape[0]):
        array[i] = max(array[i],array[i-1])
    return array

def plotresult():
    randoms = [0]*10
    fig = plt.figure()
    fig.suptitle('Searching the Best Model', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Model Attempted')
    ax.set_ylabel('Test Accuracy')
    for name in 'rnn_balance','rnn_base','rnn_bernoulli_nb','rnn_imputation_choice','rnn_model_choice', 'rnn_pca','rnn_preprocessor_choice','rnn_qda','rnn_rescale':
        rnn = maxresult(np.load('./rnn_models/'+name+'_rnn_result.npy'))
        random = maxresult(np.load('./rnn_models/'+name+'_random_result.npy'))
        for i in range(random.shape[0]):
            randoms[i] += random[i]/9
        ax.plot(rnn,label = name[4:]+' loss weight incresed')
    ax.plot(randoms,label = 'random')
    ax.legend(loc = 4)
    fig.savefig('Together.png')
    #plt.show()
    
if __name__ == "__main__":
    #plot('test',[('x',[1,2,3,4,5]),('y',[5,4,3,2,1])])
    plotresult()
