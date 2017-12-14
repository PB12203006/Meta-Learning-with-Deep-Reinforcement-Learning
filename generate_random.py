from numpy.random import dirichlet
import numpy as np

def generate_action():
    array = []
    array.append(np.random.random())
    imp = dirichlet((0.05, 0.05, 0.05))
    array.append(imp[0])
    array.append(imp[1])
    array.append(imp[2])
    rescale = dirichlet((0.05, 0.05, 0.05, 0.05))
    array.append(rescale[0])
    array.append(rescale[1])
    array.append(rescale[2])
    array.append(rescale[3])
    array.append(dirichlet((0.1, 0.1))[0])
    array.append(dirichlet((0.1, 0.1))[0])
    array.append(dirichlet((0.1, 0.1))[0])
    array.append(0.1)
    array.append(np.random.random())
    array.append(np.random.random())
    array.append(np.random.random())
    array.append(np.random.random())
    array.append(np.random.random())
    return np.array(array)

a = generate_action()
from sample_model_choice import sample_model_choice_from_prob
print(a)
choice = sample_model_choice_from_prob(a)
print(choice)