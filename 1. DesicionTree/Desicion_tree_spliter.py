import numpy as np
def var(y: np.ndarray) -> float:
    return np.power((y - y.mean()),2).sum() / y.shape[0]

def gini(y: np.ndarray) -> float:
    v = np.unique(y, return_counts=True)[1] / y.shape[0]
    return 1 - (v**2).sum()

def entropy(y: np.ndarray) -> float:
    v = np.unique(y, return_counts=True)[1] / y.shape[0]
    return -(v * np.log(v)).sum()



def tree_split(X, y, criterion):

    criterion = {"var": var, "gini": gini, "entropy": entropy}[criterion]
    best_score = np.inf
    best_ind = (0, 0)

    for col_ind, col in enumerate(X.T):
        for cell_ind, cell in enumerate(col):
            mask = (col <= cell)
            left, right = (y[mask], y[~mask])
            c = criterion(left) * left.shape[0] + criterion(right) * right.shape[0]
            if c < best_score:
                best_score, best_ind = c, (col_ind, cell_ind)
    return best_ind


print(tree_split(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]), "gini"))