from sklearn import datasets
import numpy as np

from ml_from_scratch.kmeans import KMeans
from ml_from_scratch.plot import Plot


def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Means
    clf = KMeans(k=3)
    clf.fit(X)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="K-Means Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()
