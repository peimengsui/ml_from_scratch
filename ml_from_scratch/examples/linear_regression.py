import numpy as np
from sklearn.datasets import make_regression
import sklearn.linear_model
import matplotlib.pyplot as plt

from ml_from_scratch.linear_regression import LinearRegression, ShrinkageRegression
from ml_from_scratch.utils import train_test_split, mean_squared_error


def main():
    X, y = make_regression(n_samples=100, n_features=10, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    # Linear Regression
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

    # Ridge Regression
    model_ridge = ShrinkageRegression(reg_type='l2', reg_lambda=1, lr=0.001, n_iters=1000)
    model_ridge.fit(X_train, y_train)
    # training, = plt.plot(range(1000), model_ridge.training_errors, label="Training Error")
    # plt.legend(handles=[training])
    # plt.title("Error Plot")
    # plt.ylabel('Mean Squared Error')
    # plt.xlabel('Iterations')
    # plt.show()

    model_ridge_sklearn = sklearn.linear_model.Ridge()
    model_ridge_sklearn.fit(X_train, y_train)

    y_pred_ridge = model_ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    print('Mean squared error Ridge: %s' % mse_ridge)

    y_pred_ridge_sklearn = model_ridge_sklearn.predict(X_test)
    mse_ridge_sklearn = mean_squared_error(y_test, y_pred_ridge_sklearn)
    print('Mean squared error from sklearn Ridge: %s' % mse_ridge_sklearn)

    # Lasso Regression
    model_lasso = ShrinkageRegression(reg_type='l1', reg_lambda=1, lr=0.001, n_iters=1000)
    model_lasso.fit(X_train, y_train)
    # training, = plt.plot(range(1000), model_ridge.training_errors, label="Training Error")
    # plt.legend(handles=[training])
    # plt.title("Error Plot")
    # plt.ylabel('Mean Squared Error')
    # plt.xlabel('Iterations')
    # plt.show()

    model_lasso_sklearn = sklearn.linear_model.Lasso()
    model_lasso_sklearn.fit(X_train, y_train)

    y_pred_lasso = model_lasso.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    print('Mean squared error Lasso: %s' % mse_lasso)

    y_pred_lasso_sklearn = model_lasso_sklearn.predict(X_test)
    mse_lasso_sklearn = mean_squared_error(y_test, y_pred_lasso_sklearn)
    print('Mean squared error from sklearn Lasso: %s' % mse_lasso_sklearn)


if __name__ == '__main__':
    main()
