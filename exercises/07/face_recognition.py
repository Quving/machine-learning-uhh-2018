"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset::

                     precision    recall  f1-score   support

  Gerhard_Schroeder       0.91      0.75      0.82        28
    Donald_Rumsfeld       0.84      0.82      0.83        33
         Tony_Blair       0.65      0.82      0.73        34
       Colin_Powell       0.78      0.88      0.83        58
      George_W_Bush       0.93      0.86      0.90       129

        avg / total       0.86      0.84      0.85       282



"""
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


#print(__doc__)

def recognize_faces(n_components=150, min_faces_per_person=70, svm_c=[1e3, 5e3, 1e4, 5e4, 1e5], svm_gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], resize=0.4, plot_histogram=True, show_face_gallery=False):

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


    ###############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)


    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction

    print("Extracting the top %d eigenfaces from %d faces"
        % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

    cov_matrix = np.dot(X.T, X) / n_samples
    for eigenvector in pca.components_:
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))
    print(eigenfaces)
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))


    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': svm_c,
                'gamma': svm_gamma, }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)


    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    end_time = time() - t0
    print("done in {}s".format(end_time))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # Return errors
    return mean_squared_error(y_test, y_pred, multioutput='uniform_average'), r2_score(y_test,y_pred)


    if show_face_gallery:
        prediction_titles = [title(y_pred, y_test, target_names, i)
                        for i in range(y_pred.shape[0])]
        plot_gallery(X_test, prediction_titles, h, w)

        # plot the gallery of the most significative eigenfaces

        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        plot_gallery(eigenfaces, eigenface_titles, h, w)


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
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

# plt.show()

def rearrange_x_ticks(x_ticks):

    x_ticks_i = [i for i in range(0, len(x_ticks))]
    plt.xticks(x_ticks_i, x_ticks)
    return x_ticks_i

if __name__ == "__main__":

    ncs = [1,3,5,10,15,20,30,50,100,150,200,250,300]
    # ncs = [1,3,5,10,15,20]

    losses = np.array([])
    r2_scores = np.array([])
    for nc, nc_i in zip(ncs, np.arange(len(ncs))):

        print('------------------------------------------------')
        print('Starting recognition progress for nc={} ({} of {})'.format(nc, nc_i+1, len(ncs)))

        loss, r2_score = recognize_faces(n_components=nc)

        losses = np.append(losses, loss)
        r2_scores = np.append(r2_scores, r2_score)
    
    x_ticks = rearrange_x_ticks(ncs)

    print('Losses: {}'.format(losses))

    plt.grid(True)
    plt.plot(x_ticks, r2_score, 'g.-')
    plt.title("R^2 per number of components")
    plt.xlabel("Number of components")
    plt.ylabel("R^2")
    plt.show()