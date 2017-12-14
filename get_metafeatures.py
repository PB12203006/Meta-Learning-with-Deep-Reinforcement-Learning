from metafeatures import calculate_all_metafeatures_with_labels
import numpy as np

def get_meta(dataset):
    X, y = dataset
    dim = X.shape[1]
    t = calculate_all_metafeatures_with_labels(X, y, np.zeros((dim,)), 'whatever')
    v = [t.metafeature_values[key].value for key in t.metafeature_values]
    v = [s for s in v if type(s) == type(0.0) or type(s) == np.float64 or type(s) == type(1)]
    return np.array(v)

