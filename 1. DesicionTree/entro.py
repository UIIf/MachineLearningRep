import numpy as np

def entropy(y: np.ndarray) -> float:
    v = np.unique(y, return_counts=True)[1]/y.shape[0]
    return -(v * np.log(v)).sum()