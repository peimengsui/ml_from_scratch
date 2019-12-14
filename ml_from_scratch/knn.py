import numpy as np

from ml_from_scratch.utils import euclidean_distance


class KNN:
    """ K Nearest Neighbors classifier.
        Parameters:
        -----------
        k: int
            The number of closest neighbors that will determine the class of the
            sample that we wish to predict.
        """

    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        return float(np.mean(neighbor_labels) > 0.5)

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            idx = np.argsort(euclidean_distance(test_sample, X_train))[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred
