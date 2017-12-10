from randomlearning import random_model
import get_autosklean_tried_models_hyperparameters 
import numpy as np

def generate():
    modeldict = random_model()
    modelvector = np.array(get_autosklean_tried_models_hyperparameters.encode_model(modeldict))
    return modelvector

if __name__ == "__main__":
    print(generate())
