import numpy as np
from sklearn.datasets import make_classification
import sklearn.linear_model

from ml_from_scratch.logistic_regression import LogisticRegression
from ml_from_scratch.utils import train_test_split, accuracy_score


def main():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    model_sklearn = sklearn.linear_model.LogisticRegression(penalty='none', solver='sag')
    model_sklearn.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %s' % accuracy)

    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print('Accuracy from sklearn: %s' % accuracy_sklearn)


if __name__ == '__main__':
    main()
