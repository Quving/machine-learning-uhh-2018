#!/usr/bin/python3

from __future__ import print_function
from time import time
from threading import Thread
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


print(__doc__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def face_recognition(n_components,results):
    print("Bla")
    t0 = time()
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    report = classification_report(y_test, y_pred, target_names=target_names)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=range(n_classes))

    print("done in %0.3fs" % (time() - t0))
    result =  {"report": report,
            "confusion_matrix": confusion_mat,
            "n_components": n_components}
    results.append(result)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


if __name__ == "__main__":
    results = list()
    threads = list()
    
    for n in np.arange(25,500,25):
        thread = Thread(target=face_recognition, args=(n, results))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    for res in results:
        print("================================")
        print("N_component", res["n_components"])
        print(res["report"])

