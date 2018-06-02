from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import RandomizedPCA


def plot_eigenvalues_hist(n_components=150):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))

    cov_matrix = np.dot(X.T, X) / n_samples
    eigenvalues = list()
    for eigenvector in pca.components_:
        eigenvalues.append(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print("%d eigenvalues have been computed" % len(eigenvalues))

    # Create histograms.
    plt.hist(eigenvalues, bins=20)
    plt.title("Histogram of " +str(len(eigenvalues)) + " Eigenvalues")
    plt.show()


if __name__ == "__main__":
    pca = plot_eigenvalues_hist(n_components=150)
