import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from ml_from_scratch.knn import KNN
from ml_from_scratch.utils import train_test_split, accuracy_score, normalize


def main():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = KNN()

    model_sklearn = KNeighborsClassifier()
    model_sklearn.fit(X_train, y_train)

    y_pred = model.predict(X_test, X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %s' % accuracy)

    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print('Accuracy from sklearn: %s' % accuracy_sklearn)


if __name__ == '__main__':
    main()
