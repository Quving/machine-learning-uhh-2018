from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import RandomizedPCA


def plot_eigenvalues_hist(n_components=150, bins=20):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))

    cov_matrix = np.dot(X.T, X) / n_samples
    
    eigenvalues = pca.explained_variance_

    # Create histograms.
    hist, bins_ = np.histogram(eigenvalues)

    # Scale Y values
    freq = hist / n_components

    print(freq)

    x_ticks = rearrange_x_ticks(bins_[:-1])

    # Plot histogram
    # plt.bar(x_ticks, freq)
    # plt.xlabel("Number of eigenvalues")
    # plt.ylabel("Eigenvalues")
    # plt.title("Histogram of " +str(len(eigenvalues)) + " Eigenvalues")
    # plt.grid(True)
    # plt.show()
    return np.amax(freq)

def rearrange_x_ticks(x_ticks):
    x_ticks_i = [i for i in range(0, len(x_ticks))]
    plt.xticks(x_ticks_i, x_ticks)
    return x_ticks_i

if __name__ == "__main__":

    ncs = np.arange(5,500,25)

    p_variances = np.array([])

    for nc in ncs:
        pca = plot_eigenvalues_hist(n_components=nc, bins=20)
        p_variances = np.append(p_variances, pca)


    x_ticks = rearrange_x_ticks(ncs)

    plt.plot(x_ticks, p_variances, 'g.-')
    plt.title("Data Variances of the most frequent eigenvector")
    plt.xlabel("Number of components")
    plt.ylabel("% of data variances of the first eigenvector")
    plt.grid(True)
    plt.show()
