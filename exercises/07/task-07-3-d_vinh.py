#!/usr/bin/python3

from __future__ import print_function
from time import time
from threading import Thread
import numpy as np
import logging
import operator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def face_recognition(n_components,results):
    t0 = time()
    print("Start computing", n_components, "n_compoments.")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    pca = PCA(svd_solver="randomized", n_components=n_components, whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    report = classification_report(y_test, y_pred, target_names=target_names)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=range(n_classes))

    mse_r2_score = mean_squared_error(y_test, y_pred), r2_score(y_test,y_pred)
    result =  {"report": report,
            "confusion_matrix": confusion_mat,
            "n_components": n_components,
            "mse": mse_r2_score[0],
            "r2_score": mse_r2_score[1]}

    results[n_components] = result
    print("Finished computing for", n_components,"n_components:", n_components, "done in %0.3fs" % (time() - t0))


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


if __name__ == "__main__":
    results = dict()
    threads = list()

    for n in np.arange(25,1000,10):
        thread = Thread(target=face_recognition, args=(n, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    x_axes, loss_y, r2_score_y = list(), list(), list()
    for key,value in sorted(results.items(), key=operator.itemgetter(0)):
        x_axes.append(key)
        loss_y.append(value["mse"])
        r2_score_y.append(value["r2_score"])

    # Plot mse and r2
    plt.gcf().clear()
    plt.title("MSE and R2_score by increasing n_components.")
    plt.grid(True)
    plt.plot(x_axes, loss_y, label="MSE")
    plt.plot(x_axes, r2_score_y, label="R2_Score")
    leg = plt.legend(loc='upper right', ncol=1, mode="None", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.savefig('task-07-3-d_plot.png')
