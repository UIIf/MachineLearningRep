import numpy as np


def gini(y: np.ndarray) -> float:
    v = np.unique(y, return_counts=True)[1] / y.shape[0]
    return 1- (v**2).sum()