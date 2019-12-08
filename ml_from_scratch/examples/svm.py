import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from ml_from_scratch.svm import SupportVectorMachine
from ml_from_scratch.utils import train_test_split, accuracy_score, normalize


def main():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    label_mapping = {1: 1, 0: -1}
    model = SupportVectorMachine()
    model.fit(normalize(X_train), np.array([label_mapping.get(k) for k in y_train]))

    model_sklearn = SVC(gamma='scale')
    model_sklearn.fit(X_train, y_train)

    y_pred = model.predict(normalize(X_test))
    accuracy = accuracy_score(np.array([label_mapping.get(k) for k in y_test]), y_pred)
    print('Accuracy: %s' % accuracy)

    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print('Accuracy from sklearn: %s' % accuracy_sklearn)


if __name__ == '__main__':
    main()
