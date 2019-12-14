import numpy as np
from ml_from_scratch.utils import standardize, euclidean_distance


class KMeans:
    """A simple clustering method that forms k clusters by iteratively reassigning
        samples to the closest centroids and after that moves the centroids to the center
        of the new formed clusters.
        Parameters:
        -----------
        k: int
            The number of clusters the algorithm will form.
        max_iterations: int
            The number of iterations the algorithm will run for if it does
            not converge before that.
        """

    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None

    def _init_random_centroids(self, X):
        """ Initialize the centroids as k random samples of X"""
        n_samples, n_features = np.shape(X)
        indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[indices, :].copy()
        return centroids

    def _assign_clusters(self, centroids, X):
        clusters = []
        for sample in X:
            clusters.append(np.argmin(euclidean_distance(sample, centroids)))
        return clusters

    def _calculate_centroids(self, clusters, X):
        centroids = []
        for i in range(self.k):
            mask = np.array(clusters) == i

            centroids.append(np.mean(X[mask, :], axis=0))
        return np.array(centroids)

    def fit(self, X):
        X = standardize(X)
        centroids = self._init_random_centroids(X)
        for i in range(self.max_iterations):
            clusters = self._assign_clusters(centroids, X)
            prev_centroids = centroids.copy()
            centroids = self._calculate_centroids(clusters, X)
            diff = centroids - prev_centroids
            if not diff.any():
                print('break')
                break
        self.centroids = centroids
        self.clusters = clusters

    def predict(self, X):
        X = standardize(X)
        y_preds = self._assign_clusters(self.centroids, X)
        return y_preds
