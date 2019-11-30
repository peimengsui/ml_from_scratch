import numpy as np
from ml_from_scratch.utils import normalize
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


class Regularization:
    """
    l1 and l2 regularization and gradient caculation

    Parameters
    ----------
    reg_type: string
        Regularization type: l1 for Lasso and l2 for Ridge
    reg_lambda: float
        Regularization parameter lambda
    """
    def __init__(self, reg_type, reg_lambda):
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda

    def __call__(self, w):
        if self.reg_type == 'l1':
            return self.reg_lambda * np.linalg.norm(w)
        if self.reg_type == 'l2':
            return self.reg_lambda * 0.5 * w.T.dot(w)

    def grad(self, w):
        if self.reg_type == 'l1':
            return self.reg_lambda * np.sign(w)
        if self.reg_type == 'l2':
            return self.reg_lambda * w


class ShrinkageRegression:
    """
    Shrinkage Regression Model

    Parameters
    ----------
    reg_type: string
        Regularization type: l1 for Lasso and l2 for Ridge
    reg_lambda: float
        Regularization parameter lambda
    n_iters: int
        Number of iterations running gradient descent, default is 1000
    lr: float:
        learning rate for gradient descent, default is 0.001
    normalize: Booleane
        whether normalize samples, default is True
    """
    def __init__(self, reg_type, reg_lambda,
                 n_iters=1000,
                 lr=0.001,
                 normalize=False):
        self.regularization = Regularization(reg_type, reg_lambda)
        self.n_iters = n_iters
        self.lr = lr
        self.weight = None
        self.normalize = normalize

    def initialize_weights(self, n_features):
        """ Initialize weights randomly"""
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        X = X - X.mean(axis=0)
        if self.normalize:
            X = normalize(X)
        X = np.insert(X, 0, 1, axis=1)

        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iters):
            y_pred = X.dot(self.weight)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.weight))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.weight)
            # Update the weights
            self.weight -= self.lr * grad_w

        # closed from solution, non-singular matrix inverse
        # self.weight = np.linalg.inv(X.T.dot(X)+np.identity(X.shape[1])*self.reg_lambda).dot(X.T).dot(y)
        # self.bias = np.mean(y)

    def predict(self, X):
        X = X - X.mean(axis=0)
        if self.normalize:
            X = normalize(X)
        X = np.insert(X, 0, 1, axis=1)

        y_pred = X.dot(self.weight)
        return y_pred
