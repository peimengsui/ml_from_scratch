import numpy as np
from ml_from_scratch.utils import calculate_covariance_matrix


class PCA:
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.
    """
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X, n_components):
        """ Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset """
        covariance_matrix = calculate_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx][:n_components]
        self.eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

    def transform(self, X):
        # Project the data onto principal components
        X_transformed = X.dot(self.eigenvectors)

        return X_transformed
