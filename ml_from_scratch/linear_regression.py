import numpy as np
import math


class LinearRegression:
    """
    Linear Regression Model

    Parameters
    ----------
    """
    def __init__(self):
        self.weight = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.weight = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        # insert bias as the first column
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.weight)
        return y_pred
