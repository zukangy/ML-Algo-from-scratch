# Version 1.0

import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        """
        K-Nearest Neighbors algorithm
        :param n_neighbors: (int, default 5) number of neighbors used taken into account
        for predictions.
        """
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        :param X: (array-like or sparse matrix) shape = (n_samples n_features)
        :param y: (array-like or sparse matrix) shape = (n_samples n_features)
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
        :param X: (array-like or sparse matrix) shape = (n_samples n_features)

        return: (1D-array) predictions
        """
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            # Calculate Euclidean distances |((x1, x2), (x3, x4))|^2 = (x1 - x3)**2 + (x2 - x4)**2
            dists = np.sum((self.X - X[i]) ** 2, axis=1)

            # Sort and record the indices with the smallest distances
            closest_indices = dists.argsort()[:self.n_neighbors]

            # Count the most frequent label as the prediction of the row
            y_hat[i] = np.bincount(self.y[closest_indices]).argmax()

        return y_hat


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, n_clusters_per_class=1)

    print(f"Feature shape: {X.shape}")
    print(f"target shape: {y.shape}")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    predictions = knn.predict(X)

    print(f'Accuracy: {accuracy_score(y, predictions)}')