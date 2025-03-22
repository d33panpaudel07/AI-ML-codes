from __future__ import division, print_function
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # Visualize the results
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis")
    plt.title("K-Means Clustering Results")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()  # Display the plot


if __name__ == "__main__":
    main()
