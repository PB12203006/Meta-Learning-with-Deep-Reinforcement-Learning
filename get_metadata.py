from get_metadata_from_log import load_metadata,encode_without_read
import autosklearn.classification
import sklearn.model_selection
from Generate_Data_Set import generate_data_set
import numpy as np

# input tuple (x,y)
def run_autosklearn(input,threadidx, verbose=False):
    X = input[0]
    y = input[1]
    path = './log_rnn'+str(threadidx)
    #N = int(np.random.randint(100) * 10 + 100)
    #D = int(np.random.random() * 0.8 * N + 5)
    #X, y, probas = generate_data_set(N, D)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=15, per_run_time_limit=4,
            tmp_folder=path,
            output_folder=path,
            include_estimators=['qda', 'bernoulli_nb'],
            exclude_estimators = None,
            include_preprocessors=["no_preprocessing", "pca"], exclude_preprocessors=None,
            ml_memory_limit= 20480,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            ensemble_size=1, initial_configurations_via_metalearning=38)
    automl.fit(X_train, y_train, dataset_name="test")
    if verbose:
        print(automl.show_models())
    return path+'/AutoML(1):test.log'

def get_metadata(input, threadidx, verbose=False):
    path = run_autosklearn(input, threadidx, verbose)
    metadatadict = load_metadata(path)
    metadatavector = encode_without_read(metadatadict)
    if verbose:
        print(metadatadict)
        print(metadatavector)
    return np.array(metadatavector)

if __name__ == "__main__":
    X = np.loadtxt('Data_Set/X_1')
    y = np.loadtxt('Data_Set/y_1')
    input = (X,y)
    get_metadata(input,0)