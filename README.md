# Meta-Learning-with-Deep-Reinforcement-Learning

Dependencies: autosklearn, tensorflow, Keras, Jupyter Notebook

Use the command "python3 Reinfocement_Learning_Three.py" to run the reinforcement Learning algorithm (for round 3).

Then using jupyter notebook to open "Visualize_improved_RL.ipynb" to visualize the RL progress.

* __Auto_test_on_generated_data.py__  Processes Dataset with Auto-sklearn

* __Generate_Data_Set.py__   Generates DataSet

* __generate_random_model.py__  generate a random model, uniform distributed

* __get_autosklean_tried_models_hyperparameters.py__   Parses the log of Auto-sklearn

* __get_metadata_from_log.py__ get the metadata of a given dataset index (with log file)

* __get_metadata.py__  get the metadata of a given dataset

* __get_performance_of_encoded_model.py__ get performance dictionary/encoded matrix from encoded model choice matrix/vector

* __integrate_rnn_input_matrix.py__ integrate metadata, model choices, model performance arrays

* __plot.py__  plot the result

* __randomlearning.py__  generate a random model and implement it on dataset

* __reproduce_performance_of_AutoML_tried_models.py__   Reproduces the path of Auto-sklearn by calling sklearn directly

* __RNN.py__ generate, train and evaluate RNN model (policy network)

* __sample_model_choice.py__ sample and get predicted model choice from rnn prediction/probability

* __summary.ipynb__   Summarizes the statistics to produce input to neural network

* __Reinforcement_Learning_Run__ : first round of Reinforcement learning.

* __Reinforcement_Learning_Two__ : second round of Reinforcement learning.

* __Reinforcement_Learning_Three__ : third round of Reinforcement learning.
