from randomlearning import random_model
import get_autosklean_tried_models_hyperparameters 
import numpy as np

def generate():
    #generate a model in dictionary class
    modeldict = random_model()
    #encode the dictionary to array
    modelvector = np.array(get_autosklean_tried_models_hyperparameters.encode_model(modeldict))
    return modelvector

if __name__ == "__main__":
    print(generate())
