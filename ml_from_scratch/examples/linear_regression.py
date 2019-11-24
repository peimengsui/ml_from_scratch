import numpy as np
from sklearn.datasets import make_regression
import sklearn.linear_model

from ml_from_scratch.linear_regression import LinearRegression
from ml_from_scratch.utils import train_test_split, mean_squared_error


def main():
    X, y = make_regression(n_samples=100, n_features=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    model = LinearRegression()
    model.fit(X_train, y_train)

    model_sklearn = sklearn.linear_model.LinearRegression()
    model_sklearn.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean squared error: %s' % mse)

    y_pred_sklearn = model_sklearn.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    print('Mean squared error from sklearn: %s' % mse_sklearn)


if __name__ == '__main__':
    main()
