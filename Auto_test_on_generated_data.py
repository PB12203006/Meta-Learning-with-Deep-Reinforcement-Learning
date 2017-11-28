import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from Generate_Data_Set import generate_data_set 
import autosklearn.classification
import numpy as np
import sys

generate_range = range(int(sys.argv[1]), int(sys.argv[2]))
for data_set_index in generate_range:
	N = int(np.random.randint(100) * 10 + 100)
	D = int(np.random.random() * 0.8 * N + 5)
	X, y, probas = generate_data_set(N, D)
	np.savetxt('Data_Set/X_' + str(data_set_index), X)
	np.savetxt('Data_Set/y_' + str(data_set_index), y)
	np.savetxt('Data_Set/probas_' + str(data_set_index), probas)	
	feature_types = (['numerical'] * D)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)	
	
	automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=150, per_run_time_limit=20,
        tmp_folder='./log/classifier_log' + str(data_set_index),
        output_folder='./log/classifier_out' + str(data_set_index),
        include_estimators=['qda', 'bernoulli_nb', 'libsvm_svc'],
        exclude_estimators = None,
        include_preprocessors=["no_preprocessing"], exclude_preprocessors=None,
        ml_memory_limit= 20480,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        ensemble_size=30, initial_configurations_via_metalearning=0)
	automl.fit(X_train, y_train, dataset_name='simulated' + str(data_set_index),)
	print(automl.show_models())
	
