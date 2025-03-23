import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.Y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]

        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        most_common = np.bincount(k_nearest_labels).argman()
        return most_common
